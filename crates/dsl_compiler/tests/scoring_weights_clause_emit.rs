//! Pin: `scoring { row X per_target { base: <const>, weights: <expr> } }`
//! lowers `weights:` into the row's utility expression as `base + weights`
//! (instead of parse-and-discarding the `weights:` clause).
//!
//! Closes Gap C from `docs/architecture/gaps_observed.md` (squad_skirmish
//! discovery, 2026-05-11): pre-fix the parser captured the `base:`
//! literal as the row's score and dropped `weights:` on the floor, so
//! the personality-weighted scoring rows in `assets/sim/squad_skirmish.sim`
//! contributed nothing to argmax — the emitted scoring kernel's
//! utility expression was `let utility_N: f32 = config_K;` (the base
//! literal alone), not `base + weights * personality + ...` as the
//! .sim authored.
//!
//! The fix lowers `base + weights` as a `CgExpr::Binary { AddF32 }`
//! node in `cg::lower::scoring::lower_per_ability_row`, so the
//! emitted WGSL utility expression contains BOTH the base literal AND
//! the per-agent SoA load referenced by `weights:`.
//!
//! Approach (B) from the gap-fix prompt: implement weights lowering
//! (rather than rejecting the clause).

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::cg::expr::{BinaryOp, CgExpr, CgTy, LitValue};
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::op::ComputeOpKind;
use dsl_compiler::cg::program::CgProgram;

/// Drive an inline `.sim` source through parse → resolve → CG-lower
/// → schedule → emit. Surfaces lowering diagnostics as `eprintln!` so
/// failures pinpoint the first defect instead of an opaque panic.
fn compile_inline(src: &str) -> (CgProgram, EmittedArtifacts) {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(outcome) => {
            for diag in &outcome.diagnostics {
                eprintln!("[lower diagnostic] {diag}");
            }
            panic!(
                "lower_compilation_to_cg returned {} diagnostic(s) — \
                 see stderr above",
                outcome.diagnostics.len()
            );
        }
    };
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit");
    (cg, art)
}

/// Round-trip an inline scoring decl with both `base:` and `weights:`
/// clauses through to the CG IR. Asserts the row's utility lowers to
/// a `CgExpr::Binary { AddF32, lhs, rhs }` whose:
///   - `lhs` traces back to the `base:` constant (10.0)
///   - `rhs` traces back to a chain that ultimately reads the
///     `Altruism` agent SoA field (the body of the `weights:` clause)
///
/// This is the IR-level evidence that the weights clause is no longer
/// parse-and-discarded; the WGSL emit assertion below covers the
/// downstream side.
#[test]
fn weights_clause_lowers_to_base_plus_weights_binary() {
    // Minimal scoring decl: a single per-ability row with both `base:`
    // and `weights:` clauses. Uses `agents.altruism(self)` as the
    // weights body — a registered F32 SoA column that exists in the
    // shipped agent shape, so it lowers without reaching for an
    // exotic field. The `* 5.0` multiplier proves the weights body
    // can be a non-trivial expression (matches squad_skirmish's
    // `agents.altruism(self) * 30.0` shape).
    let src = r#"
event Tick { }
entity Soldier : Agent {
  pos: vec3,
  alive: bool,
}
scoring Soldier {
  row Strike per_target {
    base:    10.0,
    weights: agents.altruism(self) * 5.0,
  }
}
verb Strike(self) =
  action Strike
  when  self.alive
  score 0.0
"#;
    let (cg, _art) = compile_inline(src);

    // Locate the scoring op and the `Strike` row.
    let rows = cg
        .ops
        .iter()
        .find_map(|op| match &op.kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => Some(rows),
            _ => None,
        })
        .expect("expected one ScoringArgmax op");

    // The scoring decl has one user-authored `row Strike per_target`
    // PLUS the verb-injected `verb_Strike` entry (the verb's `score
    // 0.0` clause). The user-authored row is the one with a non-zero
    // base; find it explicitly so a future row-order shuffle can't
    // false-positive on the verb entry.
    let strike_row = rows
        .iter()
        .find(|row| {
            // Walk the row's utility looking for the base literal 10.0.
            // Per-ability rows post-fix lower as Binary(AddF32, base,
            // weights); the user-authored row is the one with a
            // matching literal in its expression sub-tree.
            walk_expr_contains_lit_f32(&cg, row.utility, 10.0)
        })
        .expect(
            "expected at least one scoring row whose utility tree contains \
             the base literal 10.0 — the user-authored `row Strike` row \
             from the inline DSL",
        );

    // Assertion 1: the row's utility is a `Binary { op: AddF32, .. }`
    // node — the composed `base + weights`. Pre-fix this was the bare
    // `Lit(F32(10.0))` directly (weights dropped on the floor).
    let utility_expr = cg
        .exprs
        .get(strike_row.utility.0 as usize)
        .expect("strike row utility resolves to an arena expr");
    let CgExpr::Binary { op, lhs, rhs, ty } = utility_expr else {
        panic!(
            "expected the row's utility to be CgExpr::Binary (the \
             composed `base + weights`); got {utility_expr:?}",
        );
    };
    assert_eq!(
        *op,
        BinaryOp::AddF32,
        "expected the composed utility to be AddF32 (base + weights), \
         got {op:?}",
    );
    assert_eq!(
        *ty,
        CgTy::F32,
        "expected the composed utility to type-check as F32, got {ty:?}",
    );

    // Assertion 2: `lhs` traces the `base:` literal 10.0.
    assert!(
        walk_expr_contains_lit_f32(&cg, *lhs, 10.0),
        "expected the AddF32's LHS sub-tree to contain the `base:` \
         literal 10.0 — the LHS is supposed to be the lowered `score:` \
         (a.k.a. `base:`) expression",
    );

    // Assertion 3: `rhs` traces the `weights:` expression — must
    // contain BOTH the `* 5.0` multiplier AND a read of the
    // `agents.altruism(...)` SoA field. Pre-fix the weights body
    // was discarded entirely so neither would appear anywhere in
    // the lowered program.
    assert!(
        walk_expr_contains_lit_f32(&cg, *rhs, 5.0),
        "expected the AddF32's RHS sub-tree to contain the `weights:` \
         expression's `* 5.0` multiplier literal",
    );
    assert!(
        walk_expr_contains_altruism_read(&cg, *rhs),
        "expected the AddF32's RHS sub-tree to contain a Read of the \
         Altruism SoA field — i.e., the lowered `agents.altruism(self)` \
         from the `weights:` clause body",
    );
}

/// WGSL emit-side gate: the scoring kernel emits the composed `base +
/// weights` expression as a binary `+` over the two sub-expressions,
/// AND the kernel body references the per-agent SoA load that the
/// `weights:` clause names. Pre-fix the emitted WGSL was the bare
/// base literal (e.g. `let utility_N: f32 = config_K;`) with no SoA
/// reference whatsoever.
#[test]
fn weights_clause_emits_into_scoring_wgsl_kernel() {
    // Same fixture shape as the IR test above. Picks `agents.altruism`
    // (a known shipping personality SoA field; lowers as
    // `agent_altruism[index]` per `cg::emit::wgsl_body`'s
    // `agent_<field_snake>[index]` convention).
    let src = r#"
event Tick { }
entity Soldier : Agent {
  pos: vec3,
  alive: bool,
}
scoring Soldier {
  row Strike per_target {
    base:    10.0,
    weights: agents.altruism(self) * 5.0,
  }
}
verb Strike(self) =
  action Strike
  when  self.alive
  score 0.0
"#;
    let (_cg, art) = compile_inline(src);

    // The scoring op may emit as a standalone `scoring*.wgsl` kernel
    // OR be fused into a `fused_*` kernel that batches multiple
    // per-agent ops in one dispatch. Either path proves the
    // `weights:` clause survived lowering — the load-bearing check
    // is that the `agents.altruism(self)` SoA read appears in
    // *some* emitted kernel that also references the row's argmax
    // shape (`best_utility` or `scoring_output` are the canonical
    // markers; `cg::emit::kernel::lower_scoring_argmax_body`
    // synthesises both).
    let kernel_with_argmax_body: Vec<(&str, &str)> = art
        .wgsl_files
        .iter()
        .filter(|(_, body)| body.contains("best_utility") || body.contains("scoring_output"))
        .map(|(n, b)| (n.as_str(), b.as_str()))
        .collect();

    assert!(
        !kernel_with_argmax_body.is_empty(),
        "expected at least one emitted WGSL kernel to carry the scoring \
         argmax body (markers: `best_utility` / `scoring_output`); \
         emitted files: {:?}",
        art.wgsl_files.keys().collect::<Vec<_>>(),
    );

    // The `weights:` body reads `agents.altruism(self)`, which lowers
    // through `DataHandle::AgentField { field: Altruism, .. }` and
    // emits as `agent_altruism[<index>]` (see
    // `cg::emit::wgsl_body::lower_read` formatting). Pre-fix the
    // weights clause was dropped, so the scoring kernel never even
    // declared an altruism read.
    let altruism_in_scoring = kernel_with_argmax_body
        .iter()
        .any(|(_, body)| body.contains("agent_altruism"));
    assert!(
        altruism_in_scoring,
        "expected at least one scoring-bearing WGSL kernel to read \
         `agent_altruism` (the WGSL form of `agents.altruism(self)` \
         from the `weights:` clause); pre-fix the weights body was \
         parse-and-discarded and never reached the kernel.\n\
         --- scoring-bearing kernels ---\n{}\n",
        kernel_with_argmax_body
            .iter()
            .map(|(n, b)| format!("=== {n} ===\n{b}\n"))
            .collect::<String>(),
    );

    // The `* 5.0` weights multiplier must also reach the same
    // kernel — proves the weights body lowered as a structured
    // expression (not just a name reference that got constant-folded
    // away). Look for the WGSL float-literal forms `5.0` / `5f` /
    // `5.0f` that the emitter produces for an f32 literal.
    let weights_mul_in_scoring = kernel_with_argmax_body.iter().any(|(_, body)| {
        body.contains("5.0") || body.contains("5f")
    });
    assert!(
        weights_mul_in_scoring,
        "expected the scoring-bearing WGSL kernel to reference the \
         weights expression's `* 5.0` multiplier literal\n\
         --- scoring-bearing kernels ---\n{}\n",
        kernel_with_argmax_body
            .iter()
            .map(|(n, b)| format!("=== {n} ===\n{b}\n"))
            .collect::<String>(),
    );
}

// ---------------------------------------------------------------------------
// Helpers — walk a CgExpr tree looking for a specific shape.
// ---------------------------------------------------------------------------

/// True when the sub-tree rooted at `id` contains a `CgExpr::Lit(F32(v))`
/// equal to `target` (under bit-exact comparison — both literals here
/// are exactly representable, no rounding concerns).
fn walk_expr_contains_lit_f32(cg: &CgProgram, id: dsl_compiler::cg::data_handle::CgExprId, target: f32) -> bool {
    let Some(node) = cg.exprs.get(id.0 as usize) else {
        return false;
    };
    match node {
        CgExpr::Lit(LitValue::F32(v)) => v.to_bits() == target.to_bits(),
        CgExpr::Binary { lhs, rhs, .. } => {
            walk_expr_contains_lit_f32(cg, *lhs, target)
                || walk_expr_contains_lit_f32(cg, *rhs, target)
        }
        CgExpr::Unary { arg, .. } => walk_expr_contains_lit_f32(cg, *arg, target),
        CgExpr::Builtin { args, .. } => {
            args.iter().any(|a| walk_expr_contains_lit_f32(cg, *a, target))
        }
        CgExpr::Select { cond, then, else_, .. } => {
            walk_expr_contains_lit_f32(cg, *cond, target)
                || walk_expr_contains_lit_f32(cg, *then, target)
                || walk_expr_contains_lit_f32(cg, *else_, target)
        }
        _ => false,
    }
}

/// True when the sub-tree rooted at `id` contains a
/// `CgExpr::Read(DataHandle::AgentField { field: Altruism, .. })`.
/// Used to confirm the `weights:` clause's `agents.altruism(self)`
/// body lowered into the row's utility expression (rather than being
/// dropped on the floor by the pre-fix parse-and-discard path).
fn walk_expr_contains_altruism_read(cg: &CgProgram, id: dsl_compiler::cg::data_handle::CgExprId) -> bool {
    use dsl_compiler::cg::data_handle::{AgentFieldId, DataHandle};
    let Some(node) = cg.exprs.get(id.0 as usize) else {
        return false;
    };
    match node {
        CgExpr::Read(DataHandle::AgentField { field, .. }) => {
            *field == AgentFieldId::Altruism
        }
        CgExpr::Binary { lhs, rhs, .. } => {
            walk_expr_contains_altruism_read(cg, *lhs)
                || walk_expr_contains_altruism_read(cg, *rhs)
        }
        CgExpr::Unary { arg, .. } => walk_expr_contains_altruism_read(cg, *arg),
        CgExpr::Builtin { args, .. } => {
            args.iter().any(|a| walk_expr_contains_altruism_read(cg, *a))
        }
        CgExpr::Select { cond, then, else_, .. } => {
            walk_expr_contains_altruism_read(cg, *cond)
                || walk_expr_contains_altruism_read(cg, *then)
                || walk_expr_contains_altruism_read(cg, *else_)
        }
        _ => false,
    }
}
