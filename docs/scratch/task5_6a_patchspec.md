# Task 5.6a Patch Specification — `MaskPredicate` real WGSL body

**HEAD at dispatch:** `b50a6744`.
**Goal:** Replace the current `MaskPredicate` body in
`cg::emit::kernel::lower_op_body` (today: a placeholder
`let mask_<id>_value: bool = …; if (…) { atomicOr(…, …); }` that
hard-codes `agent_id` regardless of dispatch shape) with a
dispatch-shape-aware body that:

1. Resolves the per-thread agent index from the kernel's
   [`DispatchShape`].
2. Lowers the predicate via Task 4.1's
   [`lower_cg_expr_to_wgsl`].
3. Atomic-ORs a single bit into the per-agent
   `mask_<id>_bitmap` word.

The body must be exhaustive over **every** `DispatchShape` variant a
`MaskPredicate` op can land under — today **`PerAgent`** (self-only
mask) and **`PerPair { source }`** (mask with a `from
query.nearby_agents(…)` clause). `PerEvent` / `OneShot` /
`PerWord` are not produced for `MaskPredicate` by the lowering
(`crates/dsl_compiler/src/cg/lower/mask.rs:248-281`); the body
returns a typed structural error if they appear so the assumption
is checked at emit time rather than producing wrong WGSL.

---

## Files allowed to touch

- `crates/dsl_compiler/src/cg/emit/kernel.rs`
  — `lower_op_body`'s `MaskPredicate` arm. Signature change:
    `lower_op_body(op: &ComputeOp, ctx: &EmitCtx<'_>)` →
    `lower_op_body(op: &ComputeOp, dispatch: &DispatchShape, ctx: &EmitCtx<'_>)`.
    Caller `build_wgsl_body` (kernel.rs:1219) already has
    `dispatch` in scope (param at kernel.rs:1204) — pass it through.

- `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`
  — Add a small private helper
    `mask_bitmap_atomic_or_snippet(mask_id: MaskId, predicate_wgsl: &str, agent_var: &str) -> String`
    that wraps the "compute (word, bit), atomicOr if predicate" tail.
    Keeps the per-`DispatchShape` body templates in `kernel.rs`
    short and reuses the same atomic-or shape across both PerAgent
    and PerPair templates. Pure string formatting; no new IR
    surface.

- **No changes** to `compose_kernel_trait_impl`
  (`cg/emit/program.rs:478`). The host-side `record()` /
  `bind()` paths read `DispatchShape` from `topology` and emit
  the right `pass.dispatch_workgroups(...)` call — that wiring
  was already exhaustive over `PerPair` at program.rs:891-897.
  The MaskPredicate body change is WGSL-only.

- **No changes** to `KernelSpec` / `BindingMetadata`. Bindings
  for a MaskPredicate kernel are still
  `[agent_<field>… , mask_<id>_bitmap (atomic), cfg]` — that
  derivation runs in `kernel_topology_to_spec_and_body`
  (kernel.rs:209) before `build_wgsl_body` and is correct for
  both `PerAgent` and `PerPair` (the bitmap is per-agent, sized
  by `agent_cap`, regardless of dispatch shape — see
  `DataHandle::MaskBitmap` at
  `cg/data_handle.rs:696-698`).

---

## The two body templates

### Shared atomic-or tail

The `if (predicate) { atomicOr(...) }` tail is identical across
both shapes once an `agent_id` local is in scope. Extract to a
helper to keep the templates readable:

```text
let mask_{ID}_value: bool = {PREDICATE_WGSL};
if (mask_{ID}_value) {
    let mask_{ID}_word = {AGENT_VAR} >> 5u;
    let mask_{ID}_bit  = 1u << ({AGENT_VAR} & 31u);
    atomicOr(&mask_{ID}_bitmap[mask_{ID}_word], mask_{ID}_bit);
}
```

`{ID}` is the `MaskId.0` numeric value; `{PREDICATE_WGSL}` is
the result of `lower_cg_expr_to_wgsl(predicate, ctx)`;
`{AGENT_VAR}` is the WGSL local name the surrounding template
binds to the per-thread agent slot (`agent_id` for PerAgent,
`mask_{ID}_agent` for PerPair — see below for why a per-template
name).

The legacy emitter (`emit_mask_wgsl.rs:386-390`) uses
`self_id / 32u` and `1u << bit_idx`. Both emit forms are
equivalent under WGSL's u32 semantics; the shift form (`agent
>> 5u`) is a verbatim port of the existing CG placeholder
(`kernel.rs:1343-1344`) and avoids re-introducing the
`_idx` / `bit_idx` local-name shape. Either is acceptable — the
spec freezes `>> 5u` / `& 31u` because that's what the
placeholder body already produces and the existing
`mask_predicate_uses_interner_name_when_present` /
`body_includes_thread_indexing_preamble_and_op_comment` tests
(kernel.rs:2117-2133, 2172-2200) pin that shape implicitly via
`mask_5_bitmap` substring assertions.

### Template A — `DispatchShape::PerAgent`

Preamble (already emitted by `thread_indexing_preamble` at
kernel.rs:1683) binds `agent_id` and gates on
`cfg.agent_cap`. The body fragment uses `agent_id` directly:

```text
let mask_{ID}_value: bool = {PREDICATE_WGSL};
if (mask_{ID}_value) {
    let mask_{ID}_word = agent_id >> 5u;
    let mask_{ID}_bit  = 1u << (agent_id & 31u);
    atomicOr(&mask_{ID}_bitmap[mask_{ID}_word], mask_{ID}_bit);
}
```

This matches the current placeholder behavior at
kernel.rs:1340-1348, **except** the local-name prefix changes
from `word`/`bit` to `mask_{ID}_word`/`mask_{ID}_bit`. The rename
is required so a Fused kernel with two `MaskPredicate` ops
(today's `fused_three_mask_ops_share_agent_hp_binding` test at
kernel.rs:1911-1932) doesn't shadow `word`/`bit` between
fragments. WGSL `let` is block-scoped but each op's fragment is
emitted at the top level of the kernel function — without the
per-mask suffix the second op's `let word` redeclares the first
op's, which naga rejects as `redefinition of 'word'`.

### Template B — `DispatchShape::PerPair { source }`

Preamble (`thread_indexing_preamble` at kernel.rs:1693) binds
`pair = gid.x;` and emits a comment but **does not** split
`agent` / `cand` or bound-check. The MaskPredicate body must
do both:

```text
// PerPair MaskPredicate — derive (agent, cand) from `pair`.
//
// `k` is the per-agent candidate count. Today the cfg struct
// (build_generic_cfg_struct_decl at kernel.rs:976-982) carries
// only `agent_cap`; the per-pair candidate count is wired by
// Task 5.5+ when MaskPredicate kernels start flowing real
// dispatch counts. The placeholder uses `1u` so the body
// compiles + gives correct semantics for k=1; the typed
// `DispatchCtx::per_pair_candidates` (cg/dispatch.rs:206) is
// the runtime source.
let mask_{ID}_k = 1u; // TODO(task-5.7): read from cfg.per_pair_candidates.
let mask_{ID}_agent = pair / mask_{ID}_k;
let mask_{ID}_cand  = pair % mask_{ID}_k;
if (mask_{ID}_agent >= cfg.agent_cap) { return; }

let mask_{ID}_value: bool = {PREDICATE_WGSL};
if (mask_{ID}_value) {
    let mask_{ID}_word = mask_{ID}_agent >> 5u;
    let mask_{ID}_bit  = 1u << (mask_{ID}_agent & 31u);
    atomicOr(&mask_{ID}_bitmap[mask_{ID}_word], mask_{ID}_bit);
}
```

Notes:

- The bitmap is **per-agent** (one bit per agent), not
  per-pair. Every thread for the same `agent` writes into the
  same bit; `atomicOr` collapses concurrent writes — any
  matching candidate sets the agent's bit. This matches the
  legacy `AgentTarget` mask semantics
  (`emit_mask_wgsl.rs:439-501`) which sets `found = true` once
  any candidate matches and writes one bit per agent.
- The predicate may reference `target.<field>` —
  `lower_cg_expr_to_wgsl` resolves
  `Read(AgentField { target: AgentRef::PerPairCandidate, … })`
  to `agent_per_pair_candidate_<field>`
  (wgsl_body.rs:185-193). That's a placeholder identifier
  pending Task 4.2's runtime resolution; the MaskPredicate body
  emit does **not** need to wire that — it inherits whatever
  Task 4.1 produces for the target read.
- `mask_{ID}_cand` is bound but unused inside the bitmap
  write — keeping the local around documents the index split
  and gives the predicate WGSL access to a `cand` variable if
  Task 4.2's per-pair lookup expands `agent_per_pair_candidate_<field>`
  to a buffer access keyed on `cand`. WGSL allows unused
  `let`s; no warning fires.

### Template C — every other `DispatchShape` variant

`MaskPredicate` ops can only land under `PerAgent` or
`PerPair` per
`crates/dsl_compiler/src/cg/lower/mask.rs::resolve_dispatch_shape`
(248-281). If `PerEvent` / `OneShot` / `PerWord` ever appears,
fail loudly:

```rust
DispatchShape::PerEvent { .. }
| DispatchShape::OneShot
| DispatchShape::PerWord => {
    return Err(KernelEmitError::InvalidDispatchForOpKind {
        op_kind: "MaskPredicate",
        dispatch: format!("{dispatch}"),
    });
}
```

The error variant `InvalidDispatchForOpKind { op_kind:
&'static str, dispatch: String }` does not exist today — add it
to `KernelEmitError`. This is a small expansion of the typed
error surface; the same variant is reusable by 5.6b
(ScoringArgmax) and any future op-kind that pins its valid
dispatch shape. If the apply agent prefers to keep the error
surface frozen for this patch, fall back to `unreachable!()`
guarded by a `debug_assert_eq!(matches!(dispatch,
DispatchShape::PerAgent | DispatchShape::PerPair { .. }), true)`
(less typed, but doesn't ripple beyond `lower_op_body`). **Spec
prefers the typed error**; apply agent escalates if the variant
addition conflicts with a parallel patch.

---

## Patch breakdown (sequential, conflict-free with 5.6b)

### Patch 1 — extend `lower_op_body` signature

**File:** `crates/dsl_compiler/src/cg/emit/kernel.rs`

**Current** (line 1333):

```rust
fn lower_op_body(op: &ComputeOp, ctx: &EmitCtx<'_>) -> Result<String, KernelEmitError> {
```

**New:**

```rust
fn lower_op_body(
    op: &ComputeOp,
    dispatch: &DispatchShape,
    ctx: &EmitCtx<'_>,
) -> Result<String, KernelEmitError> {
```

**Caller update** (line 1219):

```rust
let fragment = lower_op_body(op, ctx)?;
```

→

```rust
let fragment = lower_op_body(op, dispatch, ctx)?;
```

`dispatch` is the `&DispatchShape` already in scope as
`build_wgsl_body`'s second parameter (line 1204). No other
caller; `lower_op_body` is private to `kernel.rs`.

### Patch 2 — replace the MaskPredicate arm body

**File:** `crates/dsl_compiler/src/cg/emit/kernel.rs`

**Replace** lines 1335-1349 (the existing
`ComputeOpKind::MaskPredicate` arm) with the dispatch-aware
body builder. Pseudocode (the apply agent expands to real Rust
strings):

```rust
ComputeOpKind::MaskPredicate { mask, predicate } => {
    let predicate_wgsl = lower_cg_expr_to_wgsl(*predicate, ctx)
        .map_err(KernelEmitError::from)?;
    match dispatch {
        DispatchShape::PerAgent => {
            Ok(mask_predicate_per_agent_body(*mask, &predicate_wgsl))
        }
        DispatchShape::PerPair { .. } => {
            Ok(mask_predicate_per_pair_body(*mask, &predicate_wgsl))
        }
        DispatchShape::PerEvent { .. }
        | DispatchShape::OneShot
        | DispatchShape::PerWord => {
            Err(KernelEmitError::InvalidDispatchForOpKind {
                op_kind: "MaskPredicate",
                dispatch: format!("{dispatch}"),
            })
        }
    }
}
```

Where `mask_predicate_per_agent_body` /
`mask_predicate_per_pair_body` are two new `fn` bodies that
expand the templates above into a single `String`. Locate the
two helpers immediately below `lower_op_body`
(insertion point: just before
`// ---------------------------------------------------------------------------`
at kernel.rs:1384, the start of the Plumbing-body-templates
section).

Each helper composes the format using the existing layout
convention (4-space-indented WGSL inside the kernel body — the
caller `build_wgsl_body` does **not** indent fragments today;
the fragment is dropped flush-left after the `// op#…` comment
line, see kernel.rs:1221-1223). Match that — every line of the
helper output is **flush-left** so the produced kernel reads as:

```text
let agent_id = gid.x;
if (agent_id >= cfg.agent_cap) { return; }

// op#0 (mask_predicate)
let mask_5_value: bool = (agent_self_hp < 5.0);
if (mask_5_value) {
    let mask_5_word = agent_id >> 5u;
    let mask_5_bit  = 1u << (agent_id & 31u);
    atomicOr(&mask_5_bitmap[mask_5_word], mask_5_bit);
}
```

The 4-space indent inside the `if` block keeps the body
WGSL-pretty without disturbing the kernel-level flush-left
convention.

### Patch 3 — extend `KernelEmitError`

**File:** `crates/dsl_compiler/src/cg/emit/kernel.rs` (or
wherever the enum lives — apply agent locates via grep for
`enum KernelEmitError`).

**Add variant:**

```rust
/// An op-kind landed under a [`DispatchShape`] its lowering does
/// not support. Surfaced by per-op body templates (today:
/// `MaskPredicate` admits `PerAgent` + `PerPair`; everything else
/// is a typed mismatch).
InvalidDispatchForOpKind {
    op_kind: &'static str,
    dispatch: String,
},
```

Plus matching `Display` impl arm:

```rust
KernelEmitError::InvalidDispatchForOpKind { op_kind, dispatch } => {
    write!(f, "{op_kind} op cannot lower under dispatch shape {dispatch}")
}
```

If `KernelEmitError` derives `Eq` / `PartialEq` (likely — for
test assertions), the embedded `String` is fine; no derive
breakage. The apply agent verifies before adding.

### Patch 4 — `wgsl_body.rs` shared helper (optional)

**File:** `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`

If the apply agent prefers a single source-of-truth for the
atomic-or tail, add a `pub(crate) fn
mask_bitmap_atomic_or_tail(mask_id: u32, predicate_wgsl: &str,
agent_var: &str) -> String` near the top of the
"Statement emission" block (line 511). Otherwise inline the tail
inside the two kernel.rs helpers — both shapes are
structurally identical and the inline form keeps the body in
one file.

**Spec preference:** inline. Two helpers in `kernel.rs`, no
new wgsl_body.rs surface. The atomic-or shape is short enough
(four lines) that extraction loses more in indirection than it
saves in dedup. If 5.6b (ScoringArgmax) needs the same shape,
extract then.

---

## Tests

All new tests go in the existing `tests` module at
`crates/dsl_compiler/src/cg/emit/kernel.rs:1729+`, following
the pattern of
`body_includes_thread_indexing_preamble_and_op_comment`
(kernel.rs:2117) and `mask_op` test fixture (kernel.rs:1769).

### Test 1 — PerAgent body shape (synthetic op, `Lit(Bool(true))` predicate)

```rust
#[test]
fn mask_predicate_per_agent_body_emits_atomic_or_at_agent_id() {
    let mut prog = CgProgram::default();
    let lit_true = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
    let kind = ComputeOpKind::MaskPredicate {
        mask: MaskId(7),
        predicate: lit_true,
    };
    let op = ComputeOp::new(
        OpId(0),
        kind,
        DispatchShape::PerAgent,
        Span::dummy(),
        &prog, &prog, &prog,
    );
    let op_id = push_op(&mut prog, op);
    let topology = KernelTopology::Split {
        op: op_id,
        dispatch: DispatchShape::PerAgent,
    };
    let ctx = EmitCtx::structural(&prog);
    let (_spec, body) = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

    // Preamble.
    assert!(body.contains("let agent_id = gid.x;"), "body: {body}");
    assert!(body.contains("if (agent_id >= cfg.agent_cap)"), "body: {body}");
    // Predicate substitution — Lit(Bool(true)) lowers to `true`.
    assert!(body.contains("let mask_7_value: bool = true;"), "body: {body}");
    // Atomic-OR shape.
    assert!(body.contains("let mask_7_word = agent_id >> 5u;"), "body: {body}");
    assert!(body.contains("let mask_7_bit  = 1u << (agent_id & 31u);"), "body: {body}");
    assert!(
        body.contains("atomicOr(&mask_7_bitmap[mask_7_word], mask_7_bit);"),
        "body: {body}"
    );
}
```

`CgExpr::Lit(LitValue::Bool(true))` lowers via
`lower_cg_expr_to_wgsl::CgExpr::Lit` arm at wgsl_body.rs:463;
the literal renderer for `Bool(true)` produces `"true"`
(verify against `lower_literal` in wgsl_body.rs — apply agent
greps if uncertain).

**Note on the existing `body_includes_thread_indexing_preamble_and_op_comment`
test (kernel.rs:2117):** that test uses `mask_op` with
`MaskId(5)` and a `(self.hp < 5.0)` predicate; the assertions
on `agent_id = gid.x` / `mask_5_bitmap` / `agent_self_hp <
5.0` continue to hold under the new body. The local-name
change from `word`/`bit` to `mask_5_word`/`mask_5_bit`
**breaks no existing assertions** — the existing test does not
inspect those locals.

### Test 2 — PerPair body shape (synthetic op, `Lit(Bool(true))` predicate)

```rust
#[test]
fn mask_predicate_per_pair_body_derives_agent_from_pair_and_writes_per_agent_bit() {
    use crate::cg::dispatch::PerPairSource;
    use crate::cg::data_handle::SpatialQueryKind;

    let mut prog = CgProgram::default();
    let lit_true = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
    let kind = ComputeOpKind::MaskPredicate {
        mask: MaskId(11),
        predicate: lit_true,
    };
    let pair_dispatch = DispatchShape::PerPair {
        source: PerPairSource::SpatialQuery(SpatialQueryKind::KinQuery),
    };
    let op = ComputeOp::new(
        OpId(0),
        kind,
        pair_dispatch,
        Span::dummy(),
        &prog, &prog, &prog,
    );
    let op_id = push_op(&mut prog, op);
    let topology = KernelTopology::Split {
        op: op_id,
        dispatch: pair_dispatch,
    };
    let ctx = EmitCtx::structural(&prog);
    let (_spec, body) = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

    // PerPair preamble (from thread_indexing_preamble).
    assert!(body.contains("let pair = gid.x;"), "body: {body}");
    // Per-mask agent/cand split.
    assert!(body.contains("let mask_11_k = 1u;"), "body: {body}");
    assert!(body.contains("let mask_11_agent = pair / mask_11_k;"), "body: {body}");
    assert!(body.contains("let mask_11_cand  = pair % mask_11_k;"), "body: {body}");
    // Bound-check uses cfg.agent_cap.
    assert!(
        body.contains("if (mask_11_agent >= cfg.agent_cap) { return; }"),
        "body: {body}"
    );
    // Predicate substitution.
    assert!(body.contains("let mask_11_value: bool = true;"), "body: {body}");
    // Bitmap write keyed on the AGENT (not the pair) — atomic-OR
    // collapses concurrent writes from different cands for the
    // same agent.
    assert!(body.contains("let mask_11_word = mask_11_agent >> 5u;"), "body: {body}");
    assert!(body.contains("let mask_11_bit  = 1u << (mask_11_agent & 31u);"), "body: {body}");
    assert!(
        body.contains("atomicOr(&mask_11_bitmap[mask_11_word], mask_11_bit);"),
        "body: {body}"
    );
}
```

### Test 3 — invalid dispatch shape errors typed

```rust
#[test]
fn mask_predicate_under_unsupported_dispatch_shape_errors() {
    let mut prog = CgProgram::default();
    let lit_true = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
    let kind = ComputeOpKind::MaskPredicate {
        mask: MaskId(0),
        predicate: lit_true,
    };
    // Construct with an illegal dispatch — synthesis would never
    // produce this, but the body emit must reject it loudly.
    let op = ComputeOp::new(
        OpId(0),
        kind,
        DispatchShape::OneShot,
        Span::dummy(),
        &prog, &prog, &prog,
    );
    let op_id = push_op(&mut prog, op);
    let topology = KernelTopology::Split {
        op: op_id,
        dispatch: DispatchShape::OneShot,
    };
    let ctx = EmitCtx::structural(&prog);
    let err = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap_err();
    assert!(
        matches!(
            err,
            KernelEmitError::InvalidDispatchForOpKind { op_kind: "MaskPredicate", .. }
        ),
        "got {err:?}"
    );
}
```

### Test 4 — fused two-op kernel doesn't shadow locals

```rust
#[test]
fn fused_two_mask_predicates_emit_distinct_local_names() {
    // The new body suffixes `word`/`bit` with the mask id so two
    // MaskPredicate ops in the same kernel don't redeclare locals.
    let mut prog = CgProgram::default();
    let m0 = mask_op(&mut prog, MaskId(0));
    let m1 = mask_op(&mut prog, MaskId(1));
    let topology = KernelTopology::Fused {
        ops: vec![m0, m1],
        dispatch: DispatchShape::PerAgent,
    };
    let ctx = EmitCtx::structural(&prog);
    let (_spec, body) = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

    // Both masks contribute their own atomic-or block.
    assert!(
        body.contains("atomicOr(&mask_0_bitmap[mask_0_word], mask_0_bit);"),
        "body: {body}"
    );
    assert!(
        body.contains("atomicOr(&mask_1_bitmap[mask_1_word], mask_1_bit);"),
        "body: {body}"
    );
    // No bare `let word = …` — the suffix is required.
    assert!(!body.contains("\n    let word ="), "body: {body}");
    assert!(!body.contains("\nlet word ="), "body: {body}");
}
```

### Test 5 — predicate uses `target` reads (PerPair smoke test)

Tests that a predicate with an `AgentRef::PerPairCandidate`
read produces a body where the placeholder identifier flows
through unchanged from `lower_cg_expr_to_wgsl`. Construct a
mask_op-style fixture but with the comparison rhs reading
`AgentField { target: AgentRef::PerPairCandidate, field: Hp }`
instead of a literal. Assert the body contains
`agent_per_pair_candidate_hp` (the structural placeholder name
from wgsl_body.rs:191).

This pins the contract that the MaskPredicate body emitter
**delegates** target resolution to `lower_cg_expr_to_wgsl` and
does not paper over the placeholder. When Task 4.2 wires the
real per-pair lookup, only `lower_cg_expr_to_wgsl` changes;
this test then needs an updated assertion (expected drift —
documented in the test docstring).

---

## Cross-patch interactions

- **5.6b (ScoringArgmax body):** also touches `lower_op_body`'s
  `ScoringArgmax` arm (kernel.rs:1356-1363). Different
  match arm; no edit overlap. The signature change
  (`lower_op_body` taking `&DispatchShape`) introduced here
  benefits 5.6b too — ScoringArgmax's body may also vary by
  dispatch shape. **Apply 5.6a first**; 5.6b inherits the new
  signature without re-editing it.

- **5.5d (bare-local resolution in mask predicates):** lands
  before any real `MaskPredicate` op flows from AST. Until 5.5d
  closes, no DSL produces `MaskPredicate` ops — the only
  exercise of this body is the synthetic tests above.
  Independent: 5.5d edits `cg/lower/expr.rs` /
  `cg/lower/mask.rs`, never `cg/emit/`. No conflict.

- **`KernelEmitError::InvalidDispatchForOpKind` variant:** new.
  If 5.6b reuses the same variant, both patches add it
  identically — apply agent resolves the redundant addition by
  keeping one. Spec recommends **5.6a adds it** since it
  surfaces the typed-error pattern first; 5.6b's spec then
  references the existing variant.

- **Existing `body_includes_thread_indexing_preamble_and_op_comment`
  test (kernel.rs:2117):** continues to pass under the new body.
  No assertion in that test inspects `word` / `bit` local names;
  the new `mask_5_word` / `mask_5_bit` names don't trip it. The
  predicate substitution (`agent_self_hp < 5.0`) is preserved
  because the predicate-lowering call site is unchanged.

- **Snapshot test `snapshot_split_mask_predicate_bindings`
  (kernel.rs:2018):** asserts on `KernelSpec.bindings`, not on
  the body string. Unaffected.

---

## Risk + drift notes

- **`mask_{ID}_k = 1u` placeholder:** the per-pair candidate
  count `k` is not yet plumbed through cfg. The placeholder
  produces semantically-correct output for `k = 1` (i.e. a
  single-candidate per-pair walk degenerates to per-agent).
  When real `k > 1` arrives (Task 5.7's dispatch-time
  population, mirroring the cfg.event_count wiring at
  kernel.rs:993-1001), the helper body changes one line
  (`let mask_{ID}_k = cfg.per_pair_candidates;`) and the
  generic cfg struct gains a `per_pair_candidates: u32` field.
  Documented inline as a `TODO(task-5.7)` comment in the
  emitted WGSL.

- **`mask_{ID}_cand` is unbound for use today:** the predicate
  cannot yet reference the candidate's runtime slot —
  `agent_per_pair_candidate_<field>` is a placeholder
  identifier. Task 4.2 wires the real per-pair lookup. When
  that lands, the lookup will likely use `mask_{ID}_cand` as
  an index into a per-pair candidate list buffer; the body
  helper does not need updating for that landing — only
  `lower_cg_expr_to_wgsl` changes.

- **Naga validation:** the produced WGSL must round-trip
  through naga without errors when the bindings declared by
  `KernelSpec` are wired in. The placeholder body
  (kernel.rs:1340-1348) was already naga-clean; the new body
  uses the same primitives (`atomicOr`, `>> 5u`, `& 31u`) and
  the new `let mask_{ID}_*` shadowing-safe locals. No new
  naga risk.

---

## Summary of edits

| File | Change | Approx. line range (HEAD `b50a6744`) |
|---|---|---|
| `crates/dsl_compiler/src/cg/emit/kernel.rs` | `lower_op_body` signature + `MaskPredicate` arm body + 2 helper fns + `InvalidDispatchForOpKind` variant + `Display` arm | 1219, 1333-1349 (replace), insert near 1384, plus enum site |
| `crates/dsl_compiler/src/cg/emit/kernel.rs` (tests mod) | 5 new tests | append after kernel.rs:2200 (mask test cluster) |
| `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` | (optional) shared atomic-or helper — **spec recommends inline, no edit** | n/a |

No `compose_kernel_trait_impl` change. No `KernelSpec` change.
No binding-aggregation change. No host-side `record()` /
`bind()` change.
