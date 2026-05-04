# Task 5.6b — Patch Specification: ScoringArgmax WGSL body

**HEAD at dispatch:** `b50a6744`
**Author of spec:** dispatch agent (no code changes)
**Apply target:** a separate "apply" agent reads this file and applies the edits sequentially.

---

## 1. Goal

Replace the `ComputeOpKind::ScoringArgmax` arm in
`crates/dsl_compiler/src/cg/emit/kernel.rs::lower_op_body` (currently a
`// TODO(task-4.x)` placeholder, lines **1356–1363**) with a real
per-agent argmax body that:

1. Iterates each `ScoringRowOp` in source order.
2. Optionally evaluates `row.guard` (skip row when `false`).
3. Evaluates `row.utility` (always F32) and `row.target` (when `Some`,
   AgentId u32).
4. Tracks the row with the maximum utility.
5. Writes `(best_action, best_target)` into the `scoring_output` flat
   `array<u32>` buffer at the agent's slot, using the legacy stride of
   4 u32 per agent.

All expression substitution is done through Task 4.1's
`lower_cg_expr_to_wgsl` helper (re-exported from
`crates/dsl_compiler/src/cg/emit/wgsl_body.rs:453`).

---

## 2. Files modified

| File | Function / region | Action |
|---|---|---|
| `crates/dsl_compiler/src/cg/emit/kernel.rs` | `lower_op_body` — `ScoringArgmax` arm (lines 1356–1363) | Replace placeholder with real argmax body builder. |
| `crates/dsl_compiler/src/cg/emit/kernel.rs` | new private helper `lower_scoring_argmax_body` placed immediately after `lower_op_body` (i.e. just above the `// SpatialQuery body templates (Task 5.6c)` divider near line 1384) | Add helper that builds the per-agent argmax fragment from `(scoring, rows, ctx)`. |
| `crates/dsl_compiler/src/cg/emit/kernel.rs` | tests block (`mod tests`, around line 1729) | Replace existing `scoring_op_emits_todo_placeholder_not_panic` test (lines 2137–2167) with two new tests: `scoring_argmax_emits_argmax_skeleton` and `scoring_argmax_emits_per_row_blocks_with_guard`. |

No edits required in `cg/emit/wgsl_body.rs` — the `lower_cg_expr_to_wgsl`
contract is sufficient. (Listed as "possibly add a helper" in the
charter, but on inspection the row-evaluation block is small enough to
keep inline in `kernel.rs` and avoid bloating `wgsl_body.rs`.)

---

## 3. Body template — single PerAgent shape

`ScoringArgmax` only ever lowers under `DispatchShape::PerAgent` (the
preamble at `kernel.rs:1683` already declares `let agent_id = gid.x;`
and bounds-checks against `cfg.agent_cap`). The fragment emitted by
`lower_op_body` runs **after** that preamble, so it can assume
`agent_id` is in range.

### 3.1 Whole-fragment shape

```text
// scoring_argmax: scoring=#<scoring.0>, <N> rows.
var best_utility: f32 = -3.4028235e38;
var best_action: u32 = 0u;
var best_target: u32 = 0xFFFFFFFFu;

// row 0: action=#<action.0>
{
    <guard_test_or_unconditional_open>
    let utility_0: f32 = <lower(row.utility)>;
    if (utility_0 > best_utility) {
        best_utility = utility_0;
        best_action = <action.0 as u32 literal>u;
        best_target = <target_value>;
    }
    <guard_close_or_nothing>
}
// row 1: ...
// ...

let scoring_base: u32 = agent_id * 4u;
scoring_output[scoring_base + 0u] = best_action;
scoring_output[scoring_base + 1u] = best_target;
scoring_output[scoring_base + 2u] = bitcast<u32>(best_utility);
scoring_output[scoring_base + 3u] = 0u;
```

Notes that drive each piece:

- **Sentinel `best_utility = -3.4028235e38`.** Approximation of
  `f32::MIN`, matching the literal form already in
  `emit_scoring_wgsl.rs:2119` (`-1.0e38` is also acceptable; the spec
  picks the more precise constant from the charter pseudocode). The
  legacy emitter uses `0.0` + a `found_any` bool because its scoring
  table can have negative scores; CG IR tests today can rely on the
  strict sentinel because there is no `found_any` companion in the
  schema. This is intentional: the spec keeps the body simple and
  matches the charter pseudocode literally; if a future row produces
  `f32::NEG_INFINITY`-style scores the emitter can promote to a
  `found_any` variant — out of scope for 5.6b.

- **`best_target` sentinel `0xFFFFFFFFu`.** Matches the legacy
  `NO_TARGET` constant from `emit_scoring_wgsl.rs:239` and the
  apply-actions WGSL contract at
  `crates/engine_gpu_rules/src/apply_actions.wgsl`. Inlined as a
  hex literal to avoid introducing a `const` that would conflict with
  any other op's preamble.

- **Stride 4 u32 per agent.** Matches the legacy
  `SCORING_STRIDE_U32 = 4u` (see
  `crates/engine_gpu_rules/src/scoring.wgsl:13`,
  `crates/dsl_compiler/src/emit_scoring_wgsl.rs:237`,
  `crates/dsl_compiler/src/emit_apply_actions_wgsl.rs:68`). Slots:
  - `+0u` chosen action (u32)
  - `+1u` chosen target (u32; `0xFFFFFFFFu` = no target)
  - `+2u` `bitcast<u32>(best_utility)` for debuggability and parity
  - `+3u` `0u` debug slot (production zero, matches legacy stub)

- **Binding name `scoring_output`.** Verified at:
  - `kernel.rs:587–591` (binding metadata: `wgsl_ty: "array<u32>"`,
    `AccessMode::ReadWriteStorage`, resident source `scoring_table`)
  - `wgsl_body.rs:164` (structural handle name)
  - `wgsl_body.rs:1173` (cross-cutting context registration)

  The CG path uses the structural name `scoring_output` (not the
  legacy `scoring_out` from `emit_scoring_wgsl.rs:231`). The
  `KernelSpec` pipeline declares the binding as `array<u32>` so the
  body MUST index it as a flat u32 array; index-assignment is fine
  because the access mode is `ReadWriteStorage`, not atomic.

### 3.2 Per-row block — guarded

When `row.guard.is_some()`:

```text
// row <i>: action=#<action.0> (guarded)
{
    let guard_<i>: bool = <lower(row.guard)>;
    if (guard_<i>) {
        let utility_<i>: f32 = <lower(row.utility)>;
        if (utility_<i> > best_utility) {
            best_utility = utility_<i>;
            best_action = <action.0>u;
            best_target = <target_value>;
        }
    }
}
```

### 3.3 Per-row block — unguarded

When `row.guard.is_none()`:

```text
// row <i>: action=#<action.0>
{
    let utility_<i>: f32 = <lower(row.utility)>;
    if (utility_<i> > best_utility) {
        best_utility = utility_<i>;
        best_action = <action.0>u;
        best_target = <target_value>;
    }
}
```

### 3.4 `<target_value>` resolution

- `row.target.is_some()` → `lower_cg_expr_to_wgsl(target_id, ctx)?`.
  The well-formed pass guarantees this lowers to a `u32`-shaped
  WGSL expression (via `CgTy::AgentId`, see `op.rs:412–415`).
- `row.target.is_none()` → literal `0xFFFFFFFFu`. Matches the
  legacy `NO_TARGET` semantic for "self-only" rows and lines up
  with the apply-actions WGSL no-target check.

### 3.5 Indentation and formatting

Match the existing style in `lower_op_body`'s `MaskPredicate` arm
(lines 1340–1348): use `\n` with 4-space indents inside the block
braces. Each row block is wrapped in its own `{ ... }` to scope the
`utility_<i>` / `guard_<i>` `let` bindings — same idiom the legacy
emitter uses for nested score loops (`emit_scoring_wgsl.rs:1782+`).

---

## 4. Helper function — `lower_scoring_argmax_body`

Add a new private function in `kernel.rs` directly after
`lower_op_body` (i.e. between the closing brace of `lower_op_body`
near line 1382 and the `// Plumbing body templates` divider near line
1384):

```rust
/// Lower a `ScoringArgmax` op to its WGSL body fragment.
///
/// Body shape — single `DispatchShape::PerAgent` form (the only
/// dispatch the CG IR emits for scoring; see
/// `crates/dsl_compiler/src/cg/dispatch.rs`):
///
/// 1. Initialise sentinel best (`best_utility = f32::MIN`,
///    `best_action = 0u`, `best_target = NO_TARGET`).
/// 2. Per row in source order: optional guard test, evaluate
///    utility, optional target, strictly-greater compare-and-swap
///    against the running best.
/// 3. Write `(best_action, best_target, bitcast(best_utility), 0u)`
///    into `scoring_output[agent_id * SCORING_STRIDE_U32 + …]`.
///
/// Stride / sentinel constants are inlined (no module-scope `const`)
/// so the fragment composes cleanly with sibling op bodies in a
/// fused topology — naga rejects duplicate top-level consts. The
/// legacy emitter (`emit_scoring_wgsl.rs`) emits a `const
/// SCORING_STRIDE_U32 = 4u` at module scope; the CG path keeps the
/// stride literal to dodge that hazard.
///
/// Expression lowering routes through Task 4.1's
/// [`lower_cg_expr_to_wgsl`]. Per-row `let` bindings (`utility_<i>`,
/// `guard_<i>`) are wrapped in their own `{ ... }` block so the
/// names don't leak into sibling rows.
fn lower_scoring_argmax_body(
    scoring: ScoringId,
    rows: &[ScoringRowOp],
    ctx: &EmitCtx<'_>,
) -> Result<String, KernelEmitError> {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(
        out,
        "// scoring_argmax: scoring=#{}, {} rows.",
        scoring.0,
        rows.len()
    ).expect("write to String never fails");
    out.push_str("var best_utility: f32 = -3.4028235e38;\n");
    out.push_str("var best_action: u32 = 0u;\n");
    out.push_str("var best_target: u32 = 0xFFFFFFFFu;\n\n");

    for (i, row) in rows.iter().enumerate() {
        let utility_wgsl = lower_cg_expr_to_wgsl(row.utility, ctx)?;
        let target_wgsl = match row.target {
            Some(target_id) => lower_cg_expr_to_wgsl(target_id, ctx)?,
            None => "0xFFFFFFFFu".to_string(),
        };
        let action_lit = format!("{}u", row.action.0);

        match row.guard {
            Some(guard_id) => {
                let guard_wgsl = lower_cg_expr_to_wgsl(guard_id, ctx)?;
                writeln!(
                    out,
                    "// row {i}: action=#{} (guarded)",
                    row.action.0
                ).unwrap();
                writeln!(out, "{{").unwrap();
                writeln!(out, "    let guard_{i}: bool = {guard_wgsl};").unwrap();
                writeln!(out, "    if (guard_{i}) {{").unwrap();
                writeln!(out, "        let utility_{i}: f32 = {utility_wgsl};").unwrap();
                writeln!(out, "        if (utility_{i} > best_utility) {{").unwrap();
                writeln!(out, "            best_utility = utility_{i};").unwrap();
                writeln!(out, "            best_action = {action_lit};").unwrap();
                writeln!(out, "            best_target = {target_wgsl};").unwrap();
                writeln!(out, "        }}").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out, "}}").unwrap();
            }
            None => {
                writeln!(out, "// row {i}: action=#{}", row.action.0).unwrap();
                writeln!(out, "{{").unwrap();
                writeln!(out, "    let utility_{i}: f32 = {utility_wgsl};").unwrap();
                writeln!(out, "    if (utility_{i} > best_utility) {{").unwrap();
                writeln!(out, "        best_utility = utility_{i};").unwrap();
                writeln!(out, "        best_action = {action_lit};").unwrap();
                writeln!(out, "        best_target = {target_wgsl};").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out, "}}").unwrap();
            }
        }
        out.push('\n');
    }

    out.push_str("let scoring_base: u32 = agent_id * 4u;\n");
    out.push_str("scoring_output[scoring_base + 0u] = best_action;\n");
    out.push_str("scoring_output[scoring_base + 1u] = best_target;\n");
    out.push_str("scoring_output[scoring_base + 2u] = bitcast<u32>(best_utility);\n");
    out.push_str("scoring_output[scoring_base + 3u] = 0u;");
    Ok(out)
}
```

Imports already in scope at the top of `kernel.rs`:
- `ComputeOpKind`, `OpId`, `PlumbingKind`, `SpatialQueryKind` from
  `crate::cg::op` (line 99). The new helper additionally needs
  `ScoringId` and `ScoringRowOp` from the same path — confirm these
  are imported; if not, extend the existing `use crate::cg::op::{…}`
  list at line 99.
- `lower_cg_expr_to_wgsl` is **not** in scope at the top of
  `kernel.rs` (only `EmitError` is, via `wgsl_body::EmitError as
  InnerEmitError` at line 106). The MaskPredicate arm calls it
  without a `super::` prefix because there is an existing top-level
  `use super::wgsl_body::lower_cg_expr_to_wgsl;` — verify this with
  a grep at apply time and add the `use` line if missing.

  **Apply-time check:** `grep -n "lower_cg_expr_to_wgsl" crates/dsl_compiler/src/cg/emit/kernel.rs`
  should show the import already present (the MaskPredicate arm at
  line 1336 calls it bare). If grep returns only call sites (no
  `use` line), add `use super::wgsl_body::lower_cg_expr_to_wgsl;`
  alongside line 106.

---

## 5. Replacement edit in `lower_op_body`

Replace (`kernel.rs:1356–1363`):

```rust
ComputeOpKind::ScoringArgmax { scoring, rows } => Ok(format!(
    "// TODO(task-4.x): scoring_argmax kernel body — \
     scoring_id={0}, {1} rows.\n\
     // The legacy emitter (emit_scoring_wgsl.rs) computes per-row \
     utility, runs argmax, and writes (action, target, score) into scoring_output.",
    scoring.0,
    rows.len()
)),
```

with:

```rust
ComputeOpKind::ScoringArgmax { scoring, rows } => {
    lower_scoring_argmax_body(*scoring, rows, ctx)
}
```

The deref of `scoring` (a `ScoringId` `Copy` newtype) and the borrow
of `rows` (a `&Vec<ScoringRowOp>`, slice-coerced) match the existing
pattern in the `MaskPredicate` arm at line 1335 (`mask` is also a
`Copy` newtype dereffed at the call site).

---

## 6. Tests

The existing test `scoring_op_emits_todo_placeholder_not_panic`
(`kernel.rs:2137–2167`) asserts the placeholder text. After this
patch the placeholder text is gone, so the test must be replaced
with two real tests.

### 6.1 Replace `scoring_op_emits_todo_placeholder_not_panic`

Delete lines 2137–2167 (the existing test plus its `// ---- 9.
Scoring placeholder body ----` divider stays; rename the divider to
`// ---- 9. Scoring argmax body ----`).

### 6.2 Add `scoring_argmax_emits_argmax_skeleton`

Single-row, unguarded, no target. Verifies the preamble + sentinel +
single-row block + tail-write are all present.

```rust
#[test]
fn scoring_argmax_emits_argmax_skeleton() {
    let mut prog = CgProgram::default();
    let utility = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(2.5)));
    let kind = ComputeOpKind::ScoringArgmax {
        scoring: ScoringId(7),
        rows: vec![ScoringRowOp {
            action: ActionId(3),
            utility,
            target: None,
            guard: None,
        }],
    };
    let probe = ComputeOp::new(
        OpId(0),
        kind,
        DispatchShape::PerAgent,
        Span::dummy(),
        &prog,
        &prog,
        &prog,
    );
    let op_id = push_op(&mut prog, probe);
    let topology = KernelTopology::Split {
        op: op_id,
        dispatch: DispatchShape::PerAgent,
    };
    let ctx = EmitCtx::structural(&prog);
    let (_spec, body) =
        kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

    // Preamble + scoping + sentinel
    assert!(body.contains("agent_id = gid.x"), "body: {body}");
    assert!(
        body.contains("// scoring_argmax: scoring=#7, 1 rows."),
        "body: {body}"
    );
    assert!(
        body.contains("var best_utility: f32 = -3.4028235e38;"),
        "body: {body}"
    );
    assert!(body.contains("var best_action: u32 = 0u;"), "body: {body}");
    assert!(
        body.contains("var best_target: u32 = 0xFFFFFFFFu;"),
        "body: {body}"
    );

    // Row block — utility lowering visible, action = #3
    assert!(body.contains("// row 0: action=#3"), "body: {body}");
    assert!(
        body.contains("let utility_0: f32 = 2.5"),
        "body: {body}"
    );
    assert!(
        body.contains("if (utility_0 > best_utility)"),
        "body: {body}"
    );
    assert!(body.contains("best_action = 3u;"), "body: {body}");
    // No target -> sentinel literal.
    assert!(
        body.contains("best_target = 0xFFFFFFFFu;"),
        "body: {body}"
    );

    // Tail write into scoring_output, stride 4.
    assert!(
        body.contains("let scoring_base: u32 = agent_id * 4u;"),
        "body: {body}"
    );
    assert!(
        body.contains("scoring_output[scoring_base + 0u] = best_action;"),
        "body: {body}"
    );
    assert!(
        body.contains("scoring_output[scoring_base + 1u] = best_target;"),
        "body: {body}"
    );
    assert!(
        body.contains(
            "scoring_output[scoring_base + 2u] = bitcast<u32>(best_utility);"
        ),
        "body: {body}"
    );
    assert!(
        body.contains("scoring_output[scoring_base + 3u] = 0u;"),
        "body: {body}"
    );

    // Placeholder text MUST be gone.
    assert!(
        !body.contains("TODO(task-4.x): scoring_argmax"),
        "body still has placeholder: {body}"
    );
}
```

### 6.3 Add `scoring_argmax_emits_per_row_blocks_with_guard`

Two rows: row 0 unguarded; row 1 guarded with a target. Verifies the
per-row scoping and the guard branch.

```rust
#[test]
fn scoring_argmax_emits_per_row_blocks_with_guard() {
    let mut prog = CgProgram::default();
    // Row 0: unguarded, utility = 1.0, no target
    let util_0 = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
    // Row 1: guarded `(true && false)` (deliberately constant so the
    // lowered text is stable + grep-able), utility = 4.0,
    // target literal -> agent slot expression. We use a u32 lit to
    // stand in for an AgentId expr; the well-formed pass would
    // reject this in production but lower_cg_expr_to_wgsl itself is
    // type-erased and just walks the arena.
    let util_1 = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(4.0)));
    let target_1 = push_expr(&mut prog, CgExpr::Lit(LitValue::U32(2)));
    let t = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
    let f = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(false)));
    let guard_1 = push_expr(
        &mut prog,
        CgExpr::Binary {
            op: crate::cg::expr::BinaryOp::And,
            lhs: t,
            rhs: f,
            ty: CgTy::Bool,
        },
    );
    let kind = ComputeOpKind::ScoringArgmax {
        scoring: ScoringId(0),
        rows: vec![
            ScoringRowOp {
                action: ActionId(0),
                utility: util_0,
                target: None,
                guard: None,
            },
            ScoringRowOp {
                action: ActionId(5),
                utility: util_1,
                target: Some(target_1),
                guard: Some(guard_1),
            },
        ],
    };
    let probe = ComputeOp::new(
        OpId(0),
        kind,
        DispatchShape::PerAgent,
        Span::dummy(),
        &prog,
        &prog,
        &prog,
    );
    let op_id = push_op(&mut prog, probe);
    let topology = KernelTopology::Split {
        op: op_id,
        dispatch: DispatchShape::PerAgent,
    };
    let ctx = EmitCtx::structural(&prog);
    let (_spec, body) =
        kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

    // Row 0 — unguarded, sentinel target.
    assert!(body.contains("// row 0: action=#0"), "body: {body}");
    assert!(
        body.contains("let utility_0: f32 = 1"),
        "body: {body}"
    );
    assert!(body.contains("best_action = 0u;"), "body: {body}");
    // Row 0 carries the sentinel target literal AT LEAST once
    // (the var-init line also contains the literal; assert the
    // assign line specifically).
    assert!(
        body.contains("best_target = 0xFFFFFFFFu;"),
        "body: {body}"
    );

    // Row 1 — guarded, explicit target.
    assert!(
        body.contains("// row 1: action=#5 (guarded)"),
        "body: {body}"
    );
    assert!(
        body.contains("let guard_1: bool = (true && false);"),
        "body: {body}"
    );
    assert!(body.contains("if (guard_1)"), "body: {body}");
    assert!(
        body.contains("let utility_1: f32 = 4"),
        "body: {body}"
    );
    assert!(
        body.contains("if (utility_1 > best_utility)"),
        "body: {body}"
    );
    assert!(body.contains("best_action = 5u;"), "body: {body}");
    // Row 1 target lowers to the u32 literal "2u".
    assert!(body.contains("best_target = 2u;"), "body: {body}");

    // Per-row scope check — the helper wraps each row in its own
    // `{ ... }` block, so two open-brace lines should appear at
    // column 0. Grep for the count.
    let row_open_count = body.matches("\n{\n").count();
    assert_eq!(row_open_count, 2, "expected 2 row blocks; body: {body}");
}
```

> **Apply-time check on `LitValue::U32` and `LitValue::Bool`.** The
> existing tests use `LitValue::F32` only (e.g. lines 1778, 2140).
> Confirm `LitValue::U32` and `LitValue::Bool` are valid variants by
> grepping `crates/dsl_compiler/src/cg/expr.rs` for `pub enum LitValue`
> before applying. If the variants are spelled differently
> (`LitValue::Uint`, `LitValue::B`, etc.), adjust the test literals;
> the assertions only depend on the lowered WGSL text shape, not on
> the Rust spelling.
>
> If `BinaryOp::And` is named differently (e.g. `BinaryOp::AndBool`),
> adjust similarly. Search anchor: `BinaryOp::LtF32` is used by the
> mask test fixture at line 1782, so the enum is in scope.

### 6.4 Re-running tests

`cargo test -p dsl_compiler` should pass with 757 lib + 10 xtask tests
(2 new tests, 1 deleted, net +1). Existing tests that grepped for the
TODO placeholder are limited to the one deleted in §6.1 — verify with
`grep -rn "scoring_argmax kernel body" crates/dsl_compiler/`. Any
other call site that referenced the placeholder text needs the same
replacement; expected count is **zero**.

---

## 7. Cross-patch interactions

- **Task 5.6a (MaskPredicate body refinement)** also touches
  `lower_op_body` in the same file. It edits the
  `ComputeOpKind::MaskPredicate` arm (lines 1335–1349); this patch
  edits the `ComputeOpKind::ScoringArgmax` arm (lines 1356–1363).
  The two arms are non-overlapping textually and there is no shared
  helper between them — apply order is irrelevant.

- **Task 5.5c (NamespaceField) / 5.5d (bare-local + Match/Quantifier
  /Fold)** expand the AST → CG expression-lowering surface. They
  affect `lower_cg_expr_to_wgsl`'s coverage but not its signature.
  As long as `lower_cg_expr_to_wgsl` continues to return
  `Result<String, EmitError>`, this patch's call sites need no
  change.

- **Task 5.6c (SpatialQuery bodies)** and **Task 5.6d (Plumbing
  bodies)** are already merged at HEAD `b50a6744` and live below
  the `lower_op_body` switch. The new helper added by this patch
  sits between `lower_op_body` and the `// SpatialQuery body
  templates` divider — no overlap with those constants.

- **Stride / sentinel choices vs. `emit_apply_actions_wgsl.rs`.**
  The legacy apply-actions kernel reads
  `scoring_out[slot * SCORING_STRIDE_U32 + SCORING_OFFSET_*]` with
  stride `4u` (`emit_apply_actions_wgsl.rs:68`). The CG-emitted
  scoring kernel must agree on stride or apply-actions reads the
  wrong slot — the spec keeps stride 4 and offsets `0`/`1` to
  preserve compatibility. Future task 5.x can pull this constant
  into a shared module so the CG and legacy paths can't drift; out
  of scope here.

---

## 8. Constraints checklist

- [x] Markdown only — no code changes by the dispatch agent.
- [x] All line-number citations are against HEAD `b50a6744`.
- [x] Expression lowering uses Task 4.1's `lower_cg_expr_to_wgsl`.
- [x] Test plan covers a synthetic 2-row scoring fixture (one
      guarded, one unguarded), per the charter §"Synthetic test
      fixtures".
- [x] Cross-patch interactions called out in §7 (5.6a same fn, 5.5c/d
      AST expansion).
- [x] PerAgent dispatch only — no PerPair / PerEvent variants
      considered.

---

## 9. Apply-time pre-flight summary

For the apply agent, in order:

1. `grep -n "lower_cg_expr_to_wgsl" crates/dsl_compiler/src/cg/emit/kernel.rs`
   — confirm the helper is already in scope (likely via a `use
   super::wgsl_body::lower_cg_expr_to_wgsl;` somewhere near line 106).
   If not, add the `use`.
2. `grep -n "use crate::cg::op::" crates/dsl_compiler/src/cg/emit/kernel.rs`
   — confirm `ScoringId` and `ScoringRowOp` are importable in the
   non-test scope. If absent (only present in the test `use`),
   extend the file-level `use crate::cg::op::{…}` list.
3. Locate `pub enum LitValue` in `crates/dsl_compiler/src/cg/expr.rs`
   and confirm `U32(u32)` + `Bool(bool)` variants exist; if spelled
   differently, adjust the §6.3 test fixture only.
4. Apply the §4 helper, the §5 arm replacement, and the §6 test
   block edits.
5. Run `cargo test -p dsl_compiler` — expect all to pass.
6. Run `cargo build` from the workspace root — expect a clean build.
7. Run `cargo test` — expect the previous 755 lib + 10 xtask totals
   to advance by `+2 -1 = +1` lib test (one test deleted, two
   added). Final expected: 756 lib + 10 xtask.
