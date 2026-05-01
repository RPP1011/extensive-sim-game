# DSL Compute-Graph IR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current ad-hoc kernel emitters (which inherited a pre-T16 kernel topology and treat kernels as primary) with a proper transpiler pipeline:

```
DSL source
  → AST                       (parsing — already exists in dsl_ast)
  → Resolved IR               (already exists — name resolution, type checks)
  → High-level CG IR          (NEW — semantic compute graph)
  → ... multiple IR→IR passes (NEW — normalization, optimization, lowering)
  → Mid-level CG IR           (NEW — fused, scheduled, post-DCE/CSE)
  → ... more IR→IR passes     (NEW — backend-specific lowering)
  → Low-level "MIR"           (NEW — almost isomorphic to WGSL)
  → WGSL + Rust output        (trivial — last pass is near-isomorphic emission)
```

**The bulk of the effort is the IR layers and the passes between them — *not* the emit step.** Real compilers stack many IR→IR transformations: inlining, constant folding, common-subexpression elimination, dead-code elimination, fusion, scheduling, backend-specific lowering. Each pass is a pure function on IR; each is testable in isolation; the pipeline composes. By the time we reach the final emit, the IR is so close to the output that emission has zero decisions left to make.

**Why this matters:** Kernels do not exist in the DSL. The DSL declares `event`, `mask`, `view`, `physics`, `scoring`, `verb`, `entity`, `config`. The pre-T16 emit pipeline chose specific kernel topologies (fused_mask = 1 kernel, scoring = 1 kernel, per-view fold kernels, etc.). Those choices were inherited blindly post-T16; the post-T16 emitters port the topology rather than re-derive it from the DSL. This plan re-derives the topology from the IR.

**Architecture:** A stack of IR layers, each more concrete than the last. Passes are pure functions; the pipeline composes:

| Layer | What it captures | Lives in |
|---|---|---|
| **AST** | Surface syntax | `dsl_ast::ast` (existing) |
| **Resolved IR** | AST + name resolution + types | `dsl_ast::ir::*IR` (existing) |
| **HIR** (high-level CG) | Decomposed compute ops with semantic-level expressions: `mask predicate`, `view fold`, `physics rule`, `scoring argmax`, etc. Reads/writes via typed `DataHandle`s. | `dsl_compiler::cg::hir` (NEW) |
| **MIR** (mid-level) | HIR after normalization passes: lazy views inlined, builtins resolved, constants folded, dead code eliminated, common subexpressions hoisted, fusion analysis run. Still semantic — but normalized. | `dsl_compiler::cg::mir` (NEW) |
| **LIR** (low-level) | MIR after backend lowering: ops grouped into concrete kernels, BGL slots assigned, dispatch shapes resolved, atomic-vs-non-atomic typed, WGSL builtin types resolved. | `dsl_compiler::cg::lir` (NEW) |
| **Output** | WGSL bodies + Rust kernel modules | `engine_gpu_rules/src/*.{rs,wgsl}` |

**Each transformation is a pass.** Passes can be added without touching upstream or downstream code. Test each pass in isolation. The pipeline is:

```
ResolvedIR
   │  lower_to_hir         — DSL semantics → CG ops
   ▼
   HIR
   │  hir::inline_lazy     — replace lazy view calls with inlined exprs
   │  hir::const_fold      — fold known constants (cfg.tick at emit-time? no; cfg layout? yes)
   │  hir::dce             — drop ops whose writes are never read
   │  hir::cse             — common-subexpression hoisting within ops
   │  hir::infer_deps      — populate reads/writes from expr/stmt walks
   │  hir::well_formed     — invariant check (dependencies acyclic, types consistent)
   ▼
   MIR
   │  mir::fusion_analyze  — identify fusable op groups
   │  mir::schedule        — topological sort + kernel-topology decision
   │  mir::lower_views     — view reads → concrete buffer reads with offset arithmetic
   │  mir::lower_rng       — `Rng { purpose }` → concrete `per_agent_u32_glsl(...)` call
   │  mir::lower_events    — typed event reads → raw u32 ring offsets
   ▼
   LIR
   │  lir::assign_bgl_slots — every binding gets a concrete slot number
   │  lir::wgsl_type_resolve — `array<u32>` vs `array<atomic<u32>>` per access mode
   │  lir::workgroup_lower — dispatch shape → workgroup_size + dispatch count
   │  lir::near_isomorphic_check — assert each LIR op corresponds 1:1 to an emit fragment
   ▼
   WGSL + Rust kernel modules
```

The final emit is a near-isomorphic walk: each LIR op produces a fixed WGSL fragment with no decisions left.

**Why this matters:** The single-step "IR → WGSL" approach (v1 of this plan) collapsed too much into one transformation. Real compilers don't do this — LLVM has dozens of passes; rustc has multiple MIR layers; even toy compilers separate semantic IR from machine-near IR. Each pass solves one problem; the composition handles the whole compile. **The DSL → output translation is bounded compiler work, not bounded transcription work.**

**Tech Stack:** Rust, AST/IR pattern, compute-graph IR with topological schedule synthesis. The output side reuses the `KernelBindingIR` + lowerings from the previous refactor (commit `4b65d0a6`) — that work is the bottom of the pipe; this plan builds the layers above it.

**Architectural Impact Statement (P8):**

- **P1 (compiler-first):** strengthened. After this plan, no DSL construct's GPU lowering is hand-coded. Every WGSL body is derived from the Compute-Graph IR via mechanical lowering. The current per-kernel emit modules (`emit_mask_wgsl`, `emit_scoring_wgsl`, etc.) are retired in favor of one IR-walking emitter.
- **P2 (schema-hash):** the Compute-Graph IR introduces new structured forms but doesn't change SoA layout / event variants. `engine/.schema_hash` unaffected. `engine_gpu_rules/.schema_hash` regenerates per emit cycle as kernel boundaries change.
- **P3 (cross-backend parity):** preserved. The IR is backend-agnostic; CPU emit (`engine_rules`) and GPU emit (`engine_gpu_rules`) both lower from the same IR. Parity becomes a property of the IR being correct, not of two emitters happening to agree.
- **P4 (`EffectOp` size):** untouched.
- **P5 (deterministic RNG):** the IR carries `per_agent_u32(seed, agent, tick, purpose)` as a typed primitive — emit strategy is uniform across backends.
- **P6 (events as mutation channel):** strengthened. The IR makes the events-only invariant explicit: every `writes` field on a compute op is an event ring, never a direct field write outside the event-fold dispatcher.
- **P7 (replayability):** the IR carries replayability flags through to emission; non-replayable events are tagged at the IR level.
- **P10 (no runtime panic):** the IR's well-formedness checks catch the failure modes that previously surfaced as `unimplemented!()` panics. If a DSL construct can't be lowered, the compiler reports it as a structured error at IR-construction time — not a runtime stub.
- **P11 (reduction determinism):** the IR explicitly tags reduction ops (atomic-append, sort-then-fold) so the emitter can enforce the deterministic-ordering pattern at lowering time.

- **Runtime gate:** `cargo test -p engine_gpu --features gpu --test gpu_pipeline_smoke` passes (already a gate). New gate: `cargo test -p engine_gpu --features gpu --test parity_with_cpu` passes — without the CPU forward inside `step_batch`. The IR is correct iff CPU + GPU produce byte-equal output from the same input.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design (after task list stabilises).

---

## Effort distribution (honest, revised)

The bulk is in IR design + IR→IR passes. Final emit is small because LIR is near-isomorphic to output.

- **Phase 1 (IR data model — HIR + MIR + LIR types):** ~20% of total effort.
- **Phase 2 (AST → HIR lowering — semantic compilation per DSL construct):** ~25% of total effort.
- **Phase 3 (HIR → MIR passes — inlining, const-fold, DCE, CSE, dependency inference, well-formedness):** ~15% of total effort.
- **Phase 4 (MIR → LIR passes — fusion, schedule, view lowering, RNG lowering, event-ring lowering, BGL slot assignment, WGSL type resolution):** ~25% of total effort.
- **Phase 5 (LIR → WGSL+Rust emission — near-isomorphic):** ~5% of total effort.
- **Phase 6 (Migration + retire legacy emitters):** ~5% of total effort.
- **Phase 7 (Validation — parity_with_cpu green without CPU forward):** ~5% of total effort.

The pass framework itself is part of Phase 1: every pass is `(InputIR, &mut Diagnostics) -> OutputIR`; passes register against pipeline stages. Adding a new pass is a small, contained change.

---

## Phase 1: Compute-Graph IR data model

The IR is the contract every later phase walks. Define it precisely; resist the urge to "figure it out as we go."

**Files (all NEW):**
- `crates/dsl_compiler/src/cg/mod.rs` — module root, re-exports
- `crates/dsl_compiler/src/cg/data_handle.rs` — `DataHandle` enum + invariants
- `crates/dsl_compiler/src/cg/expr.rs` — `CgExpr` (compute-graph expression tree)
- `crates/dsl_compiler/src/cg/op.rs` — `ComputeOp` + `ComputeOpKind`
- `crates/dsl_compiler/src/cg/dispatch.rs` — `DispatchShape` + dispatch-rule helpers
- `crates/dsl_compiler/src/cg/program.rs` — `CgProgram` (the top-level container)
- `crates/dsl_compiler/src/cg/well_formed.rs` — invariant-checking pass

### Task 1.1: `DataHandle` — typed references to state

Every read/write in the IR refers to a `DataHandle` — never a string. The handle captures *what kind* of data this is and *which* instance.

**Files:** `crates/dsl_compiler/src/cg/data_handle.rs`

```rust
/// Typed reference to a piece of simulation state. The compiler
/// tracks every "where does this data live" via these handles —
/// never via raw names. Naming becomes an emit-time concern: the
/// lowering decides what binding slot or struct field corresponds
/// to a given handle.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum DataHandle {
    /// Per-agent SoA field. `field` is a typed enum that names
    /// every field the DSL can read or write (hp, pos, alive,
    /// shield_hp, attack_damage, …). The agent itself is implicit
    /// in the dispatch shape (PerAgent → "current agent slot").
    AgentField { field: AgentFieldId, target: AgentRef },

    /// Materialized view storage. `view` is the view's stable id;
    /// `slot` is which of the view's storage layers (primary,
    /// anchor, ids — depends on storage hint).
    ViewStorage { view: ViewId, slot: ViewStorageSlot },

    /// Event ring — either the apply-path A-ring or the cascade-
    /// physics ring, identified by `ring`. The IR tracks producer/
    /// consumer relationships through these handles.
    EventRing { ring: EventRingId, kind: EventRingAccess },

    /// Configuration constant — `sim_cfg.tick`, `sim_cfg.move_speed`,
    /// ability registry slots, etc. `id` resolves through the
    /// config emit pipeline.
    ConfigConst { id: ConfigConstId },

    /// Mask bitmap output (one bit per agent per mask). `mask` is
    /// the mask's stable id.
    MaskBitmap { mask: MaskId },

    /// Scoring output — per-agent (action, target, score) tuple.
    ScoringOutput,

    /// Spatial-grid storage (cells, offsets, query results).
    SpatialStorage { kind: SpatialStorageKind },

    /// The deterministic RNG primitive: `per_agent_u32(seed, agent,
    /// tick, purpose)`. Emit-time becomes a function call — but at
    /// the IR level it's a typed read.
    Rng { purpose: RngPurpose },
}

/// Which agent does the field reference?
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum AgentRef {
    /// The dispatch's current agent (PerAgent shape).
    Self_,
    /// A target identified by an expression (typically read from a
    /// candidate buffer or scoring output).
    Target(CgExprId),
    /// The actor of the current event (PerEvent shape).
    Actor,
    /// The target of the current event.
    EventTarget,
}
```

- [ ] **Step 1: Enumerate every AgentField the DSL can name**

Walk the existing `crates/engine/src/state/agent.rs` + `crates/engine_gpu/src/sync_helpers.rs::GpuAgentSlot` for the field list. Write `AgentFieldId` as a fielded enum, not a string-keyed lookup. Each variant carries its WGSL/Rust type info (`f32` vs `u32` vs Vec3, etc.).

- [ ] **Step 2: Enumerate every ViewStorageSlot**

For each storage hint (PairMap, PairMapDecay, SymmetricPairTopK, PerEntityRing, PerEntityTopK, SlotMap), declare which storage layers exist (primary, anchor, cursor, etc.). Encode this as a typed enum keyed by storage hint.

- [ ] **Step 3: Define ConfigConstId, EventRingId, MaskId, ViewId, etc.**

These are stable opaque ids assigned during the AST → IR lowering. Use newtype wrappers (`#[derive(Copy, Clone, Eq, Hash)] pub struct ViewId(u32);`) so they can't be mixed up.

- [ ] **Step 4: Implement `Display` + serialization**

Each handle prints a stable, human-readable form ("agent.self.hp", "view[standing].slots", "event_ring[cascade_a].read") for debugging. Serialize via `serde` — needed for IR snapshots in tests.

- [ ] **Step 5: Unit tests**

For each handle variant, assert: round-trip via serde preserves identity; `Display` produces the expected form; equality is structural (two handles with the same fields are `==`).

```bash
git commit -m "feat(dsl_compiler/cg): DataHandle — typed references to sim state"
```

### Task 1.2: `CgExpr` — compute-graph expression tree

The CG expression tree is the lowered form of DSL expressions (predicates, utility scores, fold-body expressions). Type-checked, side-effect-free, references state via `DataHandle`.

**Files:** `crates/dsl_compiler/src/cg/expr.rs`

```rust
#[derive(Debug, Clone)]
pub enum CgExpr {
    /// Read a piece of state.
    Read(DataHandle),
    /// Numeric / boolean literal.
    Lit(LitValue),
    /// Binary op. `op` carries the type of operands + result.
    Binary { op: BinaryOp, lhs: CgExprId, rhs: CgExprId, ty: CgTy },
    /// Unary op (negate, not, abs, sqrt, normalize).
    Unary { op: UnaryOp, arg: CgExprId, ty: CgTy },
    /// Built-in call — distance, dot, cross, normalize_or_zero,
    /// is_hostile, can_attack, etc. Each builtin is a typed enum
    /// variant (not a name string).
    Builtin { fn_id: BuiltinId, args: Vec<CgExprId>, ty: CgTy },
    /// Per-agent RNG draw. `purpose` differentiates streams.
    Rng { purpose: RngPurpose, ty: CgTy },
    /// Conditional select (if-then-else as expression, not stmt).
    Select { cond: CgExprId, then: CgExprId, else_: CgExprId, ty: CgTy },
}
```

- [ ] **Step 1: Define `CgTy`**

Compute-graph types: `Bool`, `U32`, `I32`, `F32`, `Vec3F32`, `AgentId`, `ViewKey { view: ViewId }`, `Tick`, etc. Type-check is a property the lowering enforces.

- [ ] **Step 2: Define `BuiltinId`**

Enumerate every builtin the DSL can call: `distance`, `dot`, `cross`, `normalize`, `length`, `min`, `max`, `clamp`, `abs`, `sqrt`, `is_hostile`, `kin_count_within`, `view_<name>_get` (parameterized by `ViewId`), etc. Typed signature per builtin.

- [ ] **Step 3: Pretty-printer + structural equality + serde**

- [ ] **Step 4: Unit tests**

Round-trip serialization. Type-check happy-path expressions. Reject ill-typed expressions (e.g., `bool + f32`).

```bash
git commit -m "feat(dsl_compiler/cg): CgExpr — typed compute-graph expression tree"
```

### Task 1.3: `ComputeOp` — the unit of compute

```rust
#[derive(Debug, Clone)]
pub struct ComputeOp {
    pub id: OpId,
    pub kind: ComputeOpKind,
    /// Every DataHandle this op reads. Used for fusion analysis +
    /// BGL synthesis (the union becomes the kernel's bind set).
    pub reads: Vec<DataHandle>,
    /// Every DataHandle this op writes. Same use as reads.
    pub writes: Vec<DataHandle>,
    /// Dispatch shape — determines workgroup count + per-thread
    /// data fetching at emit time.
    pub shape: DispatchShape,
    /// Source location for diagnostics.
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ComputeOpKind {
    /// Per-agent predicate evaluation. Lowered from `mask` decls.
    /// The predicate is a `CgExpr<Bool>`; emit walks it to produce
    /// the mask-bit set logic.
    MaskPredicate { mask: MaskId, predicate: CgExprId },

    /// Per-agent utility computation + argmax. Lowered from
    /// `scoring` decls. `rows` is the action-keyed list of
    /// (utility expr, target expr) pairs.
    ScoringArgmax { scoring: ScoringId, rows: Vec<ScoringRowOp> },

    /// Per-event handler. Lowered from `physics` rules. `body` is a
    /// `CgStmtList` (sibling tree to CgExpr — assignments + emits).
    PhysicsRule { rule: PhysicsRuleId, on_event: EventKindId, body: CgStmtListId },

    /// Per-event view-storage update. Lowered from `view fold {
    /// on Event => ... }` handlers.
    ViewFold { view: ViewId, on_event: EventKindId, body: CgStmtListId },

    /// Spatial query — dispatch shape determines hash-build vs
    /// query-walk.
    SpatialQuery { kind: SpatialQueryKind },

    /// One-shot scratch ops — agents pack/unpack, alive bitmap pack,
    /// event-ring drain, sim_cfg upload, snapshot kick. These are
    /// emit-strategy plumbing the schedule synthesis can choose to
    /// inline or split.
    Plumbing { kind: PlumbingKind },
}
```

- [ ] **Step 1: Implement `ComputeOpKind` variants**

Each variant carries its specific data in typed fields. No string keys.

- [ ] **Step 2: Define `CgStmt` + `CgStmtList`**

Statements are the "body" form (assignments, emits, control flow). Sibling to `CgExpr`. A statement can read+write `DataHandle`s.

- [ ] **Step 3: Auto-derive `reads` + `writes` from kind**

For each kind, write a function `kind.compute_dependencies() -> (Vec<DataHandle>, Vec<DataHandle>)` that walks the embedded expression/statement trees and collects every read/write. The `ComputeOp::reads`/`writes` fields are populated at construction by calling this — never set manually.

- [ ] **Step 4: Unit tests**

For each `ComputeOpKind`, construct a synthetic op and assert the auto-derived dependencies match expectations.

```bash
git commit -m "feat(dsl_compiler/cg): ComputeOp — typed compute-graph op + auto-derived deps"
```

### Task 1.4: `DispatchShape` — how an op runs

```rust
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DispatchShape {
    /// One thread per agent slot. agent_cap → workgroup count.
    PerAgent,
    /// One thread per event in the source ring. count read from
    /// the ring's tail buffer (indirect dispatch).
    PerEvent { source_ring: EventRingId },
    /// One thread per (agent, target) pair within a spatial radius.
    /// count = agents × candidates_per_agent.
    PerPair { source: PerPairSource },
    /// Single-threaded (workgroup_size 1, dispatch 1×1×1) —
    /// used for sim_cfg.tick bumps, indirect-args seeding.
    OneShot,
    /// Per-word output (alive bitmap pack, mask compaction). Count
    /// = ceil(agent_cap / 32).
    PerWord,
}
```

- [ ] **Step 1: Implement + tests**

Each shape carries enough info that emit can derive the workgroup size + dispatch count + indexing. No hard-coded constants.

```bash
git commit -m "feat(dsl_compiler/cg): DispatchShape — typed dispatch-rule descriptors"
```

### Task 1.5: `CgProgram` — top-level container

```rust
pub struct CgProgram {
    /// Every compute op the DSL produced. Indexed by `OpId`.
    pub ops: Vec<ComputeOp>,
    /// Every CgExpr — flattened arena indexed by `CgExprId`.
    pub exprs: Vec<CgExpr>,
    /// Every CgStmtList — flattened arena indexed by `CgStmtListId`.
    pub stmt_lists: Vec<CgStmtList>,
    /// Stable interned ids → human-readable names (for diagnostics).
    pub interner: Interner,
    /// IR-level diagnostics (warnings, lints, fusion suggestions).
    pub diagnostics: Vec<CgDiagnostic>,
}
```

- [ ] **Step 1: Implement arena allocation + id newtype wrappers**

- [ ] **Step 2: Construction API — `CgProgramBuilder`**

Lowering passes use the builder to add ops/exprs/stmts; the builder enforces invariants (op ids contiguous, expr arena no dangling refs).

- [ ] **Step 3: Pretty-printer**

Walk the program; print every op with its kind, reads, writes, dispatch shape. Used for snapshot tests.

- [ ] **Step 4: Unit tests + commit**

```bash
git commit -m "feat(dsl_compiler/cg): CgProgram — arena container + builder"
```

### Task 1.6: Well-formedness pass

```rust
pub fn check_well_formed(prog: &CgProgram) -> Result<(), Vec<CgError>> { ... }
```

Properties checked:
- Every `OpId` reference resolves.
- Every `CgExprId` reference resolves.
- Every `DataHandle` is internally consistent (e.g., `MaskBitmap { mask }` references a real `MaskId`).
- No circular dependencies in the read/write graph (a cycle is the IR-level form of "this can't be scheduled").
- Type checks: every expression's claimed `ty` matches its operands' types.
- Every `writes` field on a non-event-fold op references an `EventRing` (P6 invariant).

- [ ] **Step 1-2: Implement + tests**

```bash
git commit -m "feat(dsl_compiler/cg): well_formed — IR invariants checker"
```

---

## Phase 2: AST → Compute-Graph IR lowering passes

This is where the real compilation happens. Each pass takes a Compilation (the resolved IR) and produces a piece of CgProgram.

**Files (all NEW):**
- `crates/dsl_compiler/src/cg/lower/mod.rs`
- `crates/dsl_compiler/src/cg/lower/expr.rs`
- `crates/dsl_compiler/src/cg/lower/mask.rs`
- `crates/dsl_compiler/src/cg/lower/view.rs`
- `crates/dsl_compiler/src/cg/lower/physics.rs`
- `crates/dsl_compiler/src/cg/lower/scoring.rs`
- `crates/dsl_compiler/src/cg/lower/spatial.rs`
- `crates/dsl_compiler/src/cg/lower/plumbing.rs`

### Task 2.1: Expression lowering

`lower_expr(ast: &IrExprNode, ctx: &mut LoweringCtx) -> CgExprId`

Walks resolved IR expressions, produces CG expressions. Type-checks at every node. Emits diagnostics for unsupported constructs.

- [ ] **Step 1: Lower each `IrExprNode` variant**

Dotted field access (`agent.hp`) → `Read(AgentField { field: Hp, target: AgentRef::Self_ })`. Comparisons → `Binary`. Builtins (`distance(a, b)`) → `Builtin { fn_id: Distance, args: [...] }`.

- [ ] **Step 2: Type-check each lowered expression**

If the AST says `agent.alive < 5`, produce a typed error: `bool < u32 has no defined ordering`. Don't silently lower to a typed mismatch.

- [ ] **Step 3: Tests — lower every example expression in the DSL spec**

For each example in `docs/spec/dsl.md`, lower it and assert the resulting CgExpr matches a snapshot. This is the regression gate for expression lowering correctness.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(dsl_compiler/cg/lower): expression lowering pass"
```

### Task 2.2: Mask lowering

`lower_mask(ir: &MaskIR, ctx: &mut LoweringCtx) -> ComputeOp`

Each `mask` decl becomes one `ComputeOp { kind: MaskPredicate { ... } }`.

- [ ] **Step 1: Lower mask predicate**

Use `lower_expr` to lower the mask body. Wrap it in a `MaskPredicate` op. Dispatch shape = `PerAgent` for self-only masks, `PerPair { source: SpatialQuery(...) }` for target-bound masks.

- [ ] **Step 2: Encode `from <source>` semantics**

The `from` clause selects which spatial query feeds candidates. The IR captures this as a typed `PerPair::source` — emit decides whether to fuse with another mask sharing the same source.

- [ ] **Step 3: Snapshot tests**

For each mask in `assets/sim/masks.sim`, lower it; assert the resulting op matches a snapshot. Snapshots live in `crates/dsl_compiler/tests/cg_snapshots/mask/<name>.snap`.

- [ ] **Step 4: Commit**

### Task 2.3: View lowering

`lower_view(ir: &ViewIR, ctx: &mut LoweringCtx) -> Vec<ComputeOp>`

Each materialized view becomes one or more `ComputeOp { kind: ViewFold { ... } }` — one per fold handler. Lazy views become `CgExpr` builtin definitions, not compute ops.

- [ ] **Step 1: Lower each fold handler**

Walk the handler body's statements, lower to `CgStmt`. Wrap as `ViewFold { view, on_event, body }`. Dispatch shape = `PerEvent { source_ring: <view's source ring> }`.

- [ ] **Step 2: Lower lazy views**

`@lazy` views with `Expr` body become inline-able expressions; register them in the `BuiltinId` set so other lowerings can reference them as `Builtin { fn_id: LazyView(view_id), args: ... }`.

- [ ] **Step 3: Storage-hint tracking**

The view's storage hint (PairMap / SymmetricPairTopK / PerEntityRing / etc.) determines which `ViewStorageSlot` variants are valid. The lowering attaches this info to every `Read(ViewStorage)` in the fold body so emit knows which storage layer to read.

- [ ] **Step 4: Snapshot tests + commit**

### Task 2.4: Physics rule lowering

`lower_physics(ir: &PhysicsIR, ctx: &mut LoweringCtx) -> ComputeOp`

Each `physics` rule becomes one `ComputeOp { kind: PhysicsRule { ... } }`. The rule body's statements become a `CgStmtList`.

- [ ] **Step 1: Lower the rule body**

Walk every IR stmt: assignments, emits, conditionals, ability-effect dispatches. Each lowered stmt's reads + writes become part of the op's dependency set.

- [ ] **Step 2: Effect-op match handling**

`cast` and similar rules dispatch on `EffectOp` variant. The IR captures this as a `CgStmt::Match { scrutinee, arms }` — typed pattern match on a typed scrutinee.

- [ ] **Step 3: Encode replayability flag**

Per-rule `replayable` flag (P7) propagates from the IR to the lowered op so emit can sort the rule into the right ring.

- [ ] **Step 4: Snapshot tests + commit**

### Task 2.5: Scoring lowering

Each `scoring` decl becomes one `ComputeOp { kind: ScoringArgmax { ... } }` — one op per scoring (typically one per agent kind, since scoring is per-action argmax).

- [ ] **Step 1-3: Implement + tests + commit**

### Task 2.6: Spatial query lowering

`@spatial query` annotations on masks and physics rules become `ComputeOp { kind: SpatialQuery { ... } }`. The schedule synthesis later decides whether they share a hash-build with other queries.

- [ ] **Step 1-3: Implement + tests + commit**

### Task 2.7: Plumbing lowering

The "plumbing" ops (agents pack/unpack, alive bitmap, event-ring drain, sim_cfg upload, snapshot kick) are not DSL constructs — they're emit-time plumbing. But they're still ops in the compute graph because the schedule needs them.

- [ ] **Step 1: Define `PlumbingKind` enum**

`PackAgents`, `UnpackAgents`, `AliveBitmap`, `DrainEvents`, `UploadSimCfg`, `KickSnapshot`, `SeedIndirectArgs`. Each has a fixed read/write set + dispatch shape.

- [ ] **Step 2: Synthesize plumbing ops at IR-construction time**

The CgProgram builder generates plumbing ops automatically based on the user-facing ops it sees. (E.g., if any op writes `AgentField { Alive, ... }`, an `AliveBitmap` plumbing op gets emitted to refresh the bitmap.)

- [ ] **Step 3: Tests + commit**

### Task 2.8: End-to-end lowering driver

`lower_compilation_to_cg(comp: &Compilation) -> CgProgram`

The single entry point. Walks the entire Compilation; emits one CgProgram covering every DSL construct.

- [ ] **Step 1: Implement the driver**

- [ ] **Step 2: Run on the actual project DSL**

`cargo test -p dsl_compiler --test lower_full_compilation` — lowers every `assets/sim/*.sim` file. Asserts the resulting program is well-formed (`check_well_formed` passes).

**Cycle gating (carry-forward from Task 2.7):**

`synthesize_plumbing_ops` produces structurally cyclic dependencies — `PackAgents`
reads every `AgentField` and `UnpackAgents` writes every `AgentField`, with
`AgentScratch{Packed}` as the intermediate. The cycle is real at the IR layer
(the IR doesn't model tick phases or the host/device boundary) but is resolved
at runtime by the schedule layer (PackAgents at tick start, UnpackAgents at tick
end).

The driver's `check_well_formed` gate runs on the **user-op-only program** —
before `synthesize_plumbing_ops` is invoked. Post-plumbing well-formedness is a
Phase 3 (schedule synthesis) concern.

Phase 3 must address this either:
- (a) Structurally: introduce a tick-phase or host/device tag on
  `DataHandle::AgentField` so Pack reads and Unpack writes resolve to distinct
  handles (cleanest).
- (b) Procedurally: define a phase-aware cycle detector that respects schedule
  barriers (Pack at start, Unpack at end → no cycle within a phase).

- [ ] **Step 3: Snapshot the full program**

Pretty-print the lowered CgProgram; commit the snapshot. Future changes that perturb lowering surface as snapshot diffs in code review.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(dsl_compiler/cg/lower): full DSL → CgProgram lowering"
```

**Source-ring symmetry obligation (carry-forward from Tasks 2.3, 2.4):**

After `builder.finish()`, the driver MUST iterate `prog.ops` and, for each
`ComputeOpKind::ViewFold` / `PhysicsRule` op, invoke
`op.record_read(DataHandle::EventRing { ring: source_ring, kind: EventRingAccess::Read })`
to symmetrically wire the source-ring read. Similarly, for every `CgStmt::Emit`
inside any op's body, the driver invokes
`op.record_write(DataHandle::EventRing { ring: dest_ring, kind: EventRingAccess::Append })`.

The per-op lowerings (Tasks 2.3 view, 2.4 physics) deliberately stop at the
dispatch-shape representation — `DispatchShape::PerEvent { source_ring }`
captures the ring identity, but the auto-walker (Task 1.3) only synthesises
*structural* reads/writes from the body's `Assign` targets and expression
reads, not registry-resolved ring edges. The driver is the natural place to
add the explicit `EventRing` handles because it has the full registry view.

Without this symmetry, `well_formed::detect_cycles` (which consults only
`op.reads` / `op.writes`) silently misses event-ring producer/consumer cycles —
no false positives, no true positives. Both sides must be wired together; wiring
only one creates an asymmetric edge graph that produces *worse* diagnostics
than wiring neither.

---

## Phase 3: Schedule synthesis

The CgProgram is a DAG of compute ops. Schedule synthesis assigns ops to compute pipelines (kernels) and decides dispatch order.

**Files (NEW):**
- `crates/dsl_compiler/src/cg/schedule/mod.rs`
- `crates/dsl_compiler/src/cg/schedule/topology.rs`
- `crates/dsl_compiler/src/cg/schedule/fusion.rs`

### Task 3.1: Topological sort + read/write dependency graph

`fn dependency_graph(prog: &CgProgram) -> DepGraph` — produces a graph where edges encode "op B reads what op A writes."

- [ ] **Step 1-2: Implement + tests**

### Task 3.2: Fusion analysis

Two ops are fusion candidates if:
- They share a dispatch shape.
- Their read/write sets don't conflict (no RW-after-W to the same handle without an intervening barrier).
- The schedule has them adjacent.

`fn fusion_candidates(prog: &CgProgram, deps: &DepGraph) -> Vec<FusionGroup>`.

- [ ] **Step 1: Implement fusion analysis**

- [ ] **Step 2: Default fusion strategy**

For each fusion group of size ≥ 2, choose `KernelTopology::Fused`. Singletons get `KernelTopology::Split`. Indirect dispatches (event-rings) get `KernelTopology::Indirect`.

- [ ] **Step 3: Decision-explanation diagnostics**

For each fusion decision, emit a diagnostic explaining why ops were/weren't fused. Aids in debugging emit shape questions.

- [ ] **Step 4: Tests + commit**

### Task 3.3: Megakernel option

The schedule synthesis can choose to fuse the entire program into one megakernel. T14's megakernel scaffold becomes a real emit target.

- [ ] **Step 1: Implement megakernel as an alternate schedule strategy**

- [ ] **Step 2: Configure: per-kernel vs megakernel**

`CgProgramBuilder::build_with_strategy(strategy: ScheduleStrategy)`. Strategies: `Conservative` (no fusion), `Default` (fusion within stages), `Megakernel` (one fused kernel for the whole tick).

- [ ] **Step 3: Tests + commit**

### Task 3.4: ComputeSchedule output type

```rust
pub struct ComputeSchedule {
    pub stages: Vec<ComputeStage>,
}

pub struct ComputeStage {
    pub kernels: Vec<KernelTopology>,
}

pub enum KernelTopology {
    Fused { ops: Vec<OpId>, dispatch: DispatchShape },
    Split { op: OpId, dispatch: DispatchShape },
    Indirect { producer: OpId, consumers: Vec<OpId> },
}
```

- [ ] **Step 1: Implement; commit**

```bash
git commit -m "feat(dsl_compiler/cg/schedule): synthesis + fusion + megakernel option"
```

---

## Phase 4: WGSL + Rust emission

The IR is rich. Emission walks it. **No string concatenation builds new bindings; everything goes through the existing `KernelBindingIR` + lowerings (commit `4b65d0a6`).**

**Files (all NEW):**
- `crates/dsl_compiler/src/cg/emit/mod.rs`
- `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` — lowers `CgExpr` + `CgStmt` to WGSL
- `crates/dsl_compiler/src/cg/emit/kernel.rs` — given a `KernelTopology`, builds a `KernelSpec` (from Phase 1 of the prior refactor) + WGSL body
- `crates/dsl_compiler/src/cg/emit/program.rs` — emits the full set of kernel modules from a `ComputeSchedule`

### Task 4.1: Expression / statement → WGSL

`lower_cg_expr_to_wgsl(expr: &CgExpr, prog: &CgProgram, ctx: &EmitCtx) -> String`

Pure walks. No decisions. Each `CgExpr` variant maps to a fixed WGSL form.

- [ ] **Step 1-3: Implement + snapshot tests + commit**

### Task 4.2: KernelTopology → KernelSpec

For each `KernelTopology`, derive:
- The set of `DataHandle`s read/written by all ops in the kernel → `KernelBinding[]`
- The dispatch shape → workgroup count + entry-point body shape
- The WGSL body — concatenation of each op's lowered statements, with shared per-thread setup hoisted

- [ ] **Step 1-3: Implement + tests**

The output is a `KernelSpec`. Existing `KernelBindingIR` lowerings (`lower_wgsl_bindings`, `lower_rust_bgl_entries`, etc.) produce the rest. Drift is structurally impossible.

- [ ] **Step 4: Commit**

### Task 4.3: Full-program emission

`emit_cg_program(schedule: &ComputeSchedule, prog: &CgProgram) -> EmittedArtifacts`

Walks every kernel in the schedule, calls `KernelTopology → KernelSpec`, calls the existing lowerings, produces the file set xtask writes.

- [ ] **Step 1-2: Implement + tests + commit**

```bash
git commit -m "feat(dsl_compiler/cg/emit): WGSL + Rust emission from ComputeSchedule"
```

---

## Phase 5: Migration

Replace the legacy per-kernel emit modules with the CG pipeline.

> **Acceptance gate is `parity_with_cpu`, NOT byte-equal source files.**
>
> Originally Phase 5 was framed around byte-for-byte parity with legacy emit. A 2026-04-29 audit (memo in this commit's notes) revealed:
> - On `assets/sim/*.sim` the CG pipeline lowers 21 ops total (9 view_fold + 12 plumbing) and **0** mask/scoring/physics/spatial ops. 45 typed deferral diagnostics hit AST shapes the per-construct lowerings still defer (target binding, fielded `Emit`, namespace setters, `Let`, etc.).
> - Byte-equality vs legacy emit cannot be evaluated against missing output. The plan's own qualifier ("where the existing emit was correct") was already conceding the legacy had bugs (fused_mask BGL drift, scoring drift, fold entry-name drift).
> - Phase 6's actual gate is `parity_with_cpu`: GPU output equals CPU output. That is **behavioral parity**, not source-file parity.
>
> Phase 5 is reframed around two real constraints:
>
> 1. **API compatibility.** `engine_gpu_rules`'s public exports (the `Kernel` trait, per-kernel structs like `FusedMaskKernel`, cross-cutting types like `BindingSources`/`PingpongContext`/`Pool`/`ResidentContext`/`ExternalBuffers`/`TransientHandles`) must match what `engine_gpu` imports. Internal WGSL bodies and Rust impl details may differ.
> 2. **Behavioral parity.** `gpu_pipeline_smoke` + `parity_with_cpu` (Phase 6) pass.

### Task 5.1: side-channel + parity harness

Wire `compile-dsl` to ALSO emit via the CG pipeline into a side directory, NOT overwriting `engine_gpu_rules/src/`. Add an xtask subcommand that runs the CG-pipeline output through a behavioral parity check.

1. Update `crates/xtask/src/compile_dsl_cmd.rs` to accept `--cg-emit-into <dir>`. When set: run the CG pipeline (lower → synthesize_schedule(Default) → emit_cg_program), write `EmittedArtifacts` into the side directory.
2. Add `cargo run --bin xtask -- compile-dsl-parity --cg-out <dir>` that:
   - Builds `engine_gpu_rules` against the side-directory output instead of the legacy `engine_gpu_rules/src/`.
   - Runs `parity_with_cpu` (or a CG-side variant) under `--features gpu`.
   - Reports pass/fail.
3. **No diff-to-legacy step.** The CG output may diverge textually; what matters is whether it runs and matches CPU.

- [ ] **Step 1: Implement `--cg-emit-into <dir>` flag.**
- [ ] **Step 2: Implement parity harness command.**
- [ ] **Step 3: Tests — verify the side-channel emits without panicking; the parity command surfaces typed errors when the CG output won't compile.**
- [ ] **Step 4: Commit.** `feat(xtask): CG-pipeline side-channel + parity harness`

### Task 5.2: Bucket-1 blockers (API compatibility)

Without these, the CG pipeline's emitted Rust files won't compile against `engine_gpu_rules`'s extant runtime contract.

1. Emit `Kernel` trait impl per kernel module (struct + `impl crate::Kernel for <Pascal>Kernel { type Bindings; type Cfg; new; build_cfg; bind; record; }`).
2. Emit `bgl_storage`/`bgl_uniform` cross-kernel helpers (or move to a shared helpers file).
3. Emit `lib.rs` (kernel registry: `mod <name>;` + `pub use` per kernel + `pub trait Kernel { … }`).
4. Replace structural kernel names (`cg_fused_mask_predicate_2`) with semantic names (`fused_mask`, `scoring`, `fold_<view>`, etc.) so file paths match.

- [ ] **Step 1-4: Implement + tests + commit.** `feat(dsl_compiler/cg/emit): API-compatible kernel module shape`

### Task 5.3: ViewFold parity

The 9 view_fold ops are the only kind that lower from real DSL today. Close their parity gap (storage-hint-aware bodies, anchor/ids fallback in `bind()`, cfg shape per view). After this, the side-channel build of `engine_gpu_rules` should compile and the fold kernels should behaviorally match legacy.

- [ ] **Step 1-3: Implement + tests + commit.** `feat(dsl_compiler/cg/emit): ViewFold body parity`

### Task 5.4: Cross-cutting modules

`BindingSources`, `ExternalBuffers`, `TransientHandles`, `PingpongContext`, `Pool`, `ResidentContext`, `Schedule`. These aggregate state across the program, are not per-kernel, and are imported by every kernel + by `engine_gpu`. Decide ownership:
- **(a)** Port to the CG pipeline using the schedule's union-of-handles.
- **(b)** Keep legacy emitters and accept the hybrid (CG owns kernels, legacy owns aggregates).

(a) is the cleanest endpoint. (b) is faster but locks in a long-term hybrid. **Choose (a).**

- [ ] **Step 1-3: Implement + tests + commit.** `feat(dsl_compiler/cg/emit): cross-cutting module emission`

### Task 5.5: AST coverage cleanup

The largest single batch. Without these, mask/scoring/physics/spatial ops cannot lower from real DSL.

1. `target` binding in `lower_field` (extends `LoweringCtx`).
2. Fielded `Emit` lowering (driver-supplied event-field schema).
3. `Let` binding (new `CgStmt::Let` variant or expression hoisting).
4. Namespace setter call lowering (`agents.set_hp(t, x)` → `Emit { event, fields }`).
5. `For` loop (the `cast` rule's `for op in abilities.effects(ab)`).
6. Lazy view inlining at call sites.
7. `NamespaceField` (`config.<…>` → `DataHandle::ConfigConst`).
8. `EngagementQuery` driver routing (today only `KinQuery` selected).
9. PerUnit modifier in scoring.
10. Match (top-level expression) + Quantifier + Fold.

- [ ] **Step 1-N: Implement + tests + commit.** Likely 5-10 commits, one per AST shape closed.

### Task 5.6: Per-kind body lowering

Implement real WGSL bodies for op kinds that today emit `// TODO(task-4.x)` comments:
1. `MaskPredicate` (real predicate evaluation + per-mask bitmap word write — replace `atomicOr` placeholder).
2. `ScoringArgmax` (per-row utility + per-action argmax + target resolution + output write).
3. `SpatialQuery` (3 kinds: BuildHash, KinQuery, EngagementQuery — distinct bodies).
4. `Plumbing` (per-`PlumbingKind` body — many of these don't yet have variants).

- [ ] **Step 1-N: Implement + tests + commit.** Likely 4-8 commits, one per body.

### Task 5.7: Switchover

Replace xtask's legacy-emitter calls with `emit_cg_program` writes. Run `parity_with_cpu` (Phase 6 gate). If parity fails, fix the IR/lowering/schedule — NOT the test.

- [ ] **Step 1-2: Switchover + parity gate.**
- [ ] **Step 3: Commit.** `refactor(xtask): canonical compile-dsl emit is CG pipeline`

### Task 5.8: Retire legacy emitters

After switchover holds:

- [ ] **Step 1: Mark every per-kernel emit module `#[deprecated]`.**

`emit_mask_kernel`, `emit_mask_wgsl`, `emit_scoring_kernel`, `emit_scoring_wgsl`, `emit_view_fold_kernel`, `emit_view_wgsl`'s fold emitters, `emit_movement_wgsl`, `emit_apply_actions_wgsl`, etc.

- [ ] **Step 2: Verify no callers remain.** `grep -rn "dsl_compiler::emit_<old>" crates/` outside the deprecated modules' own files.
- [ ] **Step 3: Delete.** The remaining emit code lives in `dsl_compiler::cg::emit` — one tree, structured.
- [ ] **Step 4: Commit.** `refactor(dsl_compiler): retire per-kernel emit modules; CG pipeline is canonical`

---

## Phase 6: Validation

The compute graph IR is correct iff the emitted GPU pipeline produces output byte-equal **to the CPU reference** (NOT to the legacy GPU emit). Phase 6 is the binary acceptance gate; Phase 5 textual divergence from legacy emit is acceptable as long as Phase 6 passes.

- [ ] **Step 1: Run `gpu_pipeline_smoke`**

Must PASS — every kernel instantiates against its derived BGL.

- [ ] **Step 2: Run `parity_with_cpu` under `--features gpu`**

Must PASS without the CPU forward inside `step_batch`. **This is the binary acceptance gate** — CPU output equals GPU output, byte-equal.

If parity fails, the IR is missing something — fix the IR (or the lowering / schedule), NOT the test.

- [ ] **Step 3: Run the full test sweep**

`cargo test --workspace` at parity with HEAD. Pre-existing failures (engine probe tests, dsl_compiler spec_snippets, tactical_sim lints) are OK; new failures are not.

- [ ] **Step 4: Final commit + plan-level closeout**

```bash
git commit -m "test: cg-pipeline produces byte-equal CPU/GPU parity (acceptance)"
```

---

## Final verification

After all phases, the following invariants hold:

1. `parity_with_cpu_n4_t1`, `_t10`, `_t100` all PASS under `--features gpu` — no CPU forward, real byte-equality.
2. `gpu_pipeline_smoke` PASSES.
3. `cargo build --workspace` clean. `cargo build -p engine_gpu --features gpu` clean.
4. **No per-kernel emit modules remain in `dsl_compiler/src/`.** The directory contains: `cg/` (the IR + lowerings + emit), AST/IR re-exports, schema_hash, kernel_binding_ir + kernel_lowerings (the shared bottom-of-pipe).
5. The DSL emit pipeline is the canonical source of every WGSL byte in `engine_gpu_rules/src/`. Adding a new mask / view / physics rule to `assets/sim/*.sim` produces correct GPU compute via the IR alone — no per-kernel emitter touch.
6. The schedule synthesis can choose any kernel topology (per-kernel, fused-stage, megakernel) without IR changes — the topology decision is centralized.

---

## What this plan deliberately does NOT do

- **Does NOT change the DSL surface.** `assets/sim/*.sim` files are unchanged. This is a compiler internals refactor.
- **Does NOT touch the CPU emit pipeline (`engine_rules`).** That backend doesn't go through GPU kernels; it stays on its current emit path. (Future work could route CPU emit through the same CG IR for cross-backend symmetry, but it's out of scope here.)
- **Does NOT modernize the runtime side.** `engine_gpu`'s host-side code (step_batch, ResidentPathContext, BindingSources) is unchanged except where xtask's emitted output structure requires it.
- **Does NOT pre-optimize.** The schedule synthesis ships with a Default strategy that produces correct output. Megakernel mode is exposed but not wired as default. Performance work is downstream.

---

## Honest scope estimate

| Phase | Scope | Estimate |
|---|---|---|
| 1 | IR data model (DataHandle, CgExpr, ComputeOp, DispatchShape, CgProgram, well-formed) | 4-6 days |
| 2 | AST → CG-IR lowering passes (expr, mask, view, physics, scoring, spatial, plumbing, end-to-end) | **8-12 days** (the bulk) |
| 3 | Schedule synthesis (deps, fusion, megakernel, ComputeSchedule) | 2-3 days |
| 4 | WGSL + Rust emission (mechanical from IR) | 1-2 days |
| 5 | Migration (xtask + retire legacy emitters) | 1-2 days |
| 6 | Validation (parity_with_cpu must PASS without CPU forward) | 1-3 days (depends on how many IR bugs surface here) |
| **Total** | **— DSL → CG-IR → WGSL pipeline, parity green —** | **17-28 days** |

This is multi-week. Each phase is independently committable; each commit lands runnable code. Phase 6's parity-without-CPU-forward is the binary acceptance gate.

The investment buys: a single source of truth for GPU compute, drift impossible by construction, kernel topology as a *compiler decision* instead of an inherited assumption, future kernel work mechanical.
