//! Rust emission for compiler milestone 4 — `mask` predicates.
//!
//! Every `mask <Name>(<args>) when <predicate>` declaration produces:
//!
//! 1. A standalone module file `mask/<snake_case>.rs` carrying a pure
//!    predicate fn `pub fn mask_<name>(state: &SimState, self_id: AgentId,
//!    <args>: AgentId) -> bool`. The body is the mechanical lowering of
//!    `<predicate>` as a short-circuiting chain of `if !<clause> { return
//!    false; }` statements.
//! 2. An entry in the aggregate `mask/mod.rs`'s `register()` stub (no-op
//!    for milestone 4 — the engine imports the per-mask fns directly).
//!
//! The emitter is deliberately verbose — `writeln!` into a `String`, no
//! macros, no helper traits beyond lowering utilities. Reviewers should be
//! able to skim the emitted output without tracing abstractions.
//!
//! Body lowering surface (milestone 4): `&&`-chains of boolean clauses,
//! `agents.*` accessor calls (with `agents.pos(x)` hoisted into a prelude
//! `let x_pos = ...` binding so guard lines stay short), `distance`
//! builtin, binary comparisons and arithmetic, and `ViewCall` routed to
//! `crate::generated::views::<name>(state, args...)` for DSL-declared
//! views. An `UnresolvedCall` falls through to `crate::rules::<name>(...)`
//! — but the `rules` module was retired in task 140, so any new
//! `UnresolvedCall` will fail the engine link. Prefer declaring a `view`
//! in `assets/sim/views.sim` over reviving the shim. Any IR construct
//! outside this surface raises `EmitError::Unsupported`.
//!
//! ## Rustfmt stability
//!
//! The xtask's scaffolded-kinds write path (`write_scaffolded_kinds` in
//! `src/bin/xtask/compile_dsl_cmd.rs`) does *not* run rustfmt on mask
//! files; only the events + physics paths are rustfmt'd. But the `--check`
//! CI guard rustfmts the in-memory emission before comparing. That means
//! the emitter has to produce rustfmt-stable output on its own: multi-line
//! `if` bodies, hoisted pos bindings, balanced paren usage. If rustfmt
//! reformats the emission, `--check` reports a spurious mismatch. Keep
//! changes here round-trippable through `rustfmt --edition=2021`.

use std::fmt::Write;

use crate::ast::{BinOp, UnOp};
use crate::ir::{
    Builtin, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, IrType, MaskIR, NamespaceId,
    ViewIR, ViewKind, ViewRef,
};

/// Emission context shared across mask lowering. Carries the enclosing
/// `views: &[ViewIR]` slice so `ViewCall(ViewRef, args)` can dispatch to
/// the correct generated symbol (`crate::generated::views::<name>(...)`
/// for `@lazy`, `state.views.<name>.get(...)` for `@materialized`).
#[derive(Debug, Clone, Copy)]
pub struct EmitContext<'a> {
    pub views: &'a [ViewIR],
}

impl<'a> EmitContext<'a> {
    pub const fn empty() -> Self {
        Self { views: &[] }
    }
    fn view(&self, ViewRef(idx): ViewRef) -> Option<&ViewIR> {
        self.views.get(idx as usize)
    }
}

/// Errors raised during mask emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// Some IR construct isn't supported by the milestone-4 emitter yet.
    /// Carries a short diagnostic string with the offending shape.
    Unsupported(String),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Unsupported(s) => write!(f, "mask emission: {s}"),
        }
    }
}

impl std::error::Error for EmitError {}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Emit a single mask module file (`mask/<snake_case>.rs`). Called once per
/// `MaskIR`; result is written to the engine's generated `mask/` directory.
///
/// Returns `Err(String)` (not `EmitError`) so the xtask call site stays
/// symmetric with `emit_scoring` / `emit_entity`. Internally we still use
/// the typed `EmitError` for precise diagnostics.
pub fn emit_mask(mask: &MaskIR, source_file: Option<&str>) -> Result<String, String> {
    emit_mask_with_ctx(mask, source_file, EmitContext::empty())
}

/// Like [`emit_mask`], but with a populated `EmitContext` carrying the
/// view slice. Used by the compile-all path so mask predicates can call
/// `view::<name>(...)` against the real view registry.
pub fn emit_mask_with_ctx(
    mask: &MaskIR,
    source_file: Option<&str>,
    ctx: EmitContext<'_>,
) -> Result<String, String> {
    emit_mask_result(mask, source_file, ctx).map_err(|e| e.to_string())
}

fn emit_mask_result(
    mask: &MaskIR,
    source_file: Option<&str>,
    ctx: EmitContext<'_>,
) -> Result<String, EmitError> {
    let mut out = String::new();
    emit_header(&mut out, source_file);
    emit_imports(&mut out, mask);
    emit_predicate_fn(&mut out, mask, ctx)?;
    // Task 138: when a target-bound mask carries a `from <source>`
    // clause, emit an additional `mask_<name>_candidates` enumerator
    // that walks the source and pushes every candidate passing the
    // predicate into the target-mask buffer. Self-masks and target-
    // bound masks without a `from` clause emit nothing extra.
    if mask.candidate_source.is_some() {
        writeln!(&mut out).unwrap();
        emit_candidate_enumerator_fn(&mut out, mask, ctx)?;
    }
    Ok(out)
}

/// Emit the aggregate `mask/mod.rs`. The aggregator re-exports each
/// per-mask fn (`pub use <stem>::mask_<stem>;`) and exposes a `pub fn
/// register()` stub. Milestone 4 keeps the stub a no-op: the engine's
/// mask-build path calls the per-mask fns by name. `register()` is kept
/// on the API surface so SPIR-V / GPU-side mask dispatch can grow here
/// later without adding a new entry point.
pub fn emit_mask_mod(masks: &[MaskIR]) -> String {
    let mut sorted: Vec<&MaskIR> = masks.iter().collect();
    sorted.sort_by(|a, b| a.head.name.cmp(&b.head.name));

    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();

    if sorted.is_empty() {
        writeln!(out, "// No `mask` declarations in scope. `register()` is a no-op.").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "/// Called from the engine's mask-build path. With zero masks in").unwrap();
        writeln!(out, "/// scope this is a no-op; the per-mask fns are imported by the").unwrap();
        writeln!(out, "/// call site directly when any mask does land.").unwrap();
        writeln!(out, "pub fn register() {{}}").unwrap();
        return out;
    }

    for m in &sorted {
        writeln!(out, "pub mod {};", snake_case(&m.head.name)).unwrap();
    }
    writeln!(out).unwrap();
    for m in &sorted {
        let stem = snake_case(&m.head.name);
        writeln!(out, "pub use {stem}::mask_{stem};").unwrap();
        // Task 138: target-bound masks with a `from` clause also
        // expose a candidate-enumerator re-export so the engine's
        // mask-build path can call it by name.
        if m.candidate_source.is_some() {
            writeln!(out, "pub use {stem}::mask_{stem}_candidates;").unwrap();
        }
    }
    writeln!(out).unwrap();

    writeln!(out, "/// Called from the engine's mask-build path. For milestone 4 this").unwrap();
    writeln!(out, "/// is a no-op shim — the engine imports each per-mask fn directly").unwrap();
    writeln!(out, "/// via the `pub use` re-exports above. `register()` is kept on the").unwrap();
    writeln!(out, "/// surface so SPIR-V / GPU-side mask dispatch can grow here later").unwrap();
    writeln!(out, "/// without adding new scaffolding.").unwrap();
    writeln!(out, "pub fn register() {{}}").unwrap();
    out
}

// ---------------------------------------------------------------------------
// Header / imports
// ---------------------------------------------------------------------------

fn emit_header(out: &mut String, source_file: Option<&str>) {
    match source_file {
        Some(path) => writeln!(out, "// GENERATED by dsl_compiler from {}.", path).unwrap(),
        None => writeln!(out, "// GENERATED by dsl_compiler.").unwrap(),
    }
    writeln!(
        out,
        "// Edit the .sim source; rerun `cargo run --bin xtask -- compile-dsl`."
    )
    .unwrap();
    writeln!(out, "// Do not edit by hand.").unwrap();
    writeln!(out).unwrap();
}

fn emit_imports(out: &mut String, mask: &MaskIR) {
    writeln!(out, "use crate::ids::AgentId;").unwrap();
    writeln!(out, "use crate::state::SimState;").unwrap();
    // Candidate-enumerator fns push into `TargetMask`; pull the import
    // in only when this mask actually emits one. Task 138.
    if mask.candidate_source.is_some() {
        writeln!(out, "use crate::mask::TargetMask;").unwrap();
    }
    writeln!(out).unwrap();
}

// ---------------------------------------------------------------------------
// Predicate fn
// ---------------------------------------------------------------------------

/// Indent depth for top-level statements inside the predicate body: 4 spaces.
const BASE_INDENT: usize = 4;

fn emit_predicate_fn(
    out: &mut String,
    mask: &MaskIR,
    ctx: EmitContext<'_>,
) -> Result<(), EmitError> {
    let fn_name = format!("mask_{}", snake_case(&mask.head.name));

    // Param list: `state: &SimState`, `self_id: AgentId` plus one
    // positional param per action-head slot. Milestone 4 only supports
    // the positional shape — `Named` heads would require a richer call
    // convention that we don't need for the attack mask.
    let mut params = vec!["state: &SimState".to_string(), "self_id: AgentId".to_string()];
    match &mask.head.shape {
        IrActionHeadShape::None => {}
        IrActionHeadShape::Positional(binds) => {
            for (name, _local, ty) in binds {
                if name == "_" {
                    return Err(EmitError::Unsupported(format!(
                        "`_` placeholder arg in mask `{}` head; name the arg",
                        mask.head.name
                    )));
                }
                params.push(format!("{name}: {}", render_head_type(ty, &mask.head.name)?));
            }
        }
        IrActionHeadShape::Named(_) => {
            return Err(EmitError::Unsupported(format!(
                "named-field action head on mask `{}` not supported in milestone 4",
                mask.head.name
            )));
        }
    }

    writeln!(
        out,
        "/// Predicate: can `self_id` issue this mask's action head against the given target?"
    )
    .unwrap();
    writeln!(
        out,
        "/// Lowered from `mask {}` in `assets/sim/masks.sim`.",
        mask.head.name
    )
    .unwrap();
    writeln!(out, "pub fn {fn_name}({}) -> bool {{", params.join(", ")).unwrap();
    emit_predicate_body(out, &mask.predicate, ctx)?;
    writeln!(out, "}}").unwrap();
    Ok(())
}

/// Emit a candidate-enumerator fn. Task 138.
///
/// Signature: `pub fn mask_<name>_candidates(state, self_id, out: &mut TargetMask)`.
/// Walks the `from` expression (a
/// `query.nearby_agents(<pos>, <radius>)` call — the only recognised
/// shape at v1), filters out `self_id`, runs the mask's predicate, and
/// pushes each passing candidate into `out` keyed on the mask's
/// action-head `MicroKind`.
fn emit_candidate_enumerator_fn(
    out: &mut String,
    mask: &MaskIR,
    ctx: EmitContext<'_>,
) -> Result<(), EmitError> {
    let source = mask.candidate_source.as_ref().expect(
        "emit_candidate_enumerator_fn called with mask.candidate_source = None",
    );
    let target_binding = match &mask.head.shape {
        IrActionHeadShape::Positional(binds) if binds.len() == 1 => &binds[0].0,
        IrActionHeadShape::Positional(_) => {
            return Err(EmitError::Unsupported(format!(
                "mask `{}` has `from` clause but multiple target bindings; only single-target heads are supported at v1",
                mask.head.name
            )));
        }
        IrActionHeadShape::None => {
            return Err(EmitError::Unsupported(format!(
                "mask `{}` has `from` clause but no target binding",
                mask.head.name
            )));
        }
        IrActionHeadShape::Named(_) => {
            return Err(EmitError::Unsupported(format!(
                "mask `{}` has `from` clause and a named action head; only positional heads are supported at v1",
                mask.head.name
            )));
        }
    };

    // v1: only `query.nearby_agents(<pos>, <radius>)` recognised.
    let (pos_expr, radius_expr) = match &source.kind {
        IrExpr::NamespaceCall { ns, method, args }
            if *ns == NamespaceId::Query && method == "nearby_agents" && args.len() == 2 =>
        {
            (&args[0].value, &args[1].value)
        }
        _ => {
            return Err(EmitError::Unsupported(format!(
                "mask `{}` `from` clause: expected `query.nearby_agents(<pos>, <radius>)`",
                mask.head.name
            )));
        }
    };

    let fn_name = format!("mask_{}_candidates", snake_case(&mask.head.name));
    let predicate_fn = format!("mask_{}", snake_case(&mask.head.name));
    let micro_variant = &mask.head.name;

    let mut hoisted: Vec<String> = Vec::new();
    collect_pos_hoists(pos_expr, &mut hoisted);
    collect_pos_hoists(radius_expr, &mut hoisted);

    let pad = " ".repeat(BASE_INDENT);
    writeln!(
        out,
        "/// Candidate enumerator: walk `from {}` and push every agent that",
        source_shape_summary(source)
    )
    .unwrap();
    writeln!(out, "/// satisfies the mask predicate into `out`. Task 138.").unwrap();
    writeln!(
        out,
        "pub fn {fn_name}(state: &SimState, self_id: AgentId, out: &mut TargetMask) {{"
    )
    .unwrap();
    for local in &hoisted {
        let arg = if local == "self" { "self_id".to_string() } else { local.clone() };
        let binding = pos_binding_name(local);
        writeln!(
            out,
            "{pad}let {binding} = state.agent_pos({arg}).unwrap_or(glam::Vec3::ZERO);"
        )
        .unwrap();
    }
    let pos_lowered = lower_expr_with_hoist(pos_expr, &hoisted, ctx)?;
    let radius_lowered = lower_expr_with_hoist(radius_expr, &hoisted, ctx)?;
    writeln!(out, "{pad}let pos = {pos_lowered};").unwrap();
    writeln!(out, "{pad}let radius = {radius_lowered};").unwrap();
    writeln!(out, "{pad}let spatial = state.spatial();").unwrap();
    writeln!(
        out,
        "{pad}for {target_binding} in spatial.within_radius(state, pos, radius) {{"
    )
    .unwrap();
    writeln!(
        out,
        "{pad}    if {target_binding} == self_id {{ continue; }}"
    )
    .unwrap();
    writeln!(
        out,
        "{pad}    if !{predicate_fn}(state, self_id, {target_binding}) {{ continue; }}"
    )
    .unwrap();
    writeln!(
        out,
        "{pad}    out.push(self_id, crate::mask::MicroKind::{micro_variant}, {target_binding});"
    )
    .unwrap();
    writeln!(out, "{pad}}}").unwrap();
    writeln!(out, "}}").unwrap();
    Ok(())
}

/// Short human-readable summary of the `from` clause for the generated
/// doc comment. Keeps the emission stable against rustfmt by avoiding
/// embedded code fragments.
fn source_shape_summary(expr: &IrExprNode) -> &'static str {
    match &expr.kind {
        IrExpr::NamespaceCall { ns, method, .. }
            if *ns == NamespaceId::Query && method == "nearby_agents" =>
        {
            "query.nearby_agents(...)"
        }
        _ => "<unsupported>",
    }
}

/// Lower the predicate expression as a chain of `if !(<clause>) { return
/// false; }` statements, split on top-level `&&` for short-circuit
/// readability. Anything that doesn't fit that shape is emitted as a
/// single clause guard. The final `true` fallthrough means "every clause
/// passed".
///
/// Before emitting the guards we walk the predicate once to hoist every
/// `agents.pos(<local>)` call into a prelude `let <local>_pos = state.
/// agent_pos(<local>).unwrap_or(Vec3::ZERO);` binding. The hoist keeps
/// per-guard lines short enough that rustfmt treats the emission as
/// already-formatted — critical because the xtask's scaffolded-kinds
/// write path doesn't run rustfmt on mask files.
fn emit_predicate_body(
    out: &mut String,
    predicate: &IrExprNode,
    ctx: EmitContext<'_>,
) -> Result<(), EmitError> {
    let mut clauses: Vec<&IrExprNode> = Vec::new();
    flatten_and(predicate, &mut clauses);

    let mut hoisted: Vec<String> = Vec::new();
    for clause in &clauses {
        collect_pos_hoists(clause, &mut hoisted);
    }

    let pad = " ".repeat(BASE_INDENT);
    let inner_pad = " ".repeat(BASE_INDENT + 4);
    for local in &hoisted {
        let arg = if local == "self" { "self_id".to_string() } else { local.clone() };
        let binding = pos_binding_name(local);
        writeln!(
            out,
            "{pad}let {binding} = state.agent_pos({arg}).unwrap_or(glam::Vec3::ZERO);"
        )
        .unwrap();
    }
    for clause in &clauses {
        let cond = lower_clause(clause, &hoisted, ctx)?;
        writeln!(out, "{pad}if !{cond} {{").unwrap();
        writeln!(out, "{inner_pad}return false;").unwrap();
        writeln!(out, "{pad}}}").unwrap();
    }
    writeln!(out, "{pad}true").unwrap();
    Ok(())
}

/// Walk a binary `&&` tree and collect every leaf clause in source order.
/// Non-`&&` expressions are treated as a single-leaf clause.
fn flatten_and<'a>(node: &'a IrExprNode, out: &mut Vec<&'a IrExprNode>) {
    if let IrExpr::Binary(BinOp::And, lhs, rhs) = &node.kind {
        flatten_and(lhs, out);
        flatten_and(rhs, out);
    } else {
        out.push(node);
    }
}

/// Walk the predicate AST and record every `agents.pos(<Local>)` call's
/// source-level local name, de-duplicated and in first-seen order. Only
/// `agents.pos` of a bare `Local` argument is hoisted — more complex
/// expressions stay inline.
fn collect_pos_hoists(node: &IrExprNode, out: &mut Vec<String>) {
    match &node.kind {
        IrExpr::NamespaceCall { ns, method, args }
            if *ns == NamespaceId::Agents && method == "pos" =>
        {
            if args.len() == 1 {
                if let IrExpr::Local(_, name) = &args[0].value.kind {
                    if !out.contains(name) {
                        out.push(name.clone());
                    }
                }
            }
            for a in args {
                collect_pos_hoists(&a.value, out);
            }
        }
        IrExpr::NamespaceCall { args, .. }
        | IrExpr::BuiltinCall(_, args)
        | IrExpr::UnresolvedCall(_, args)
        | IrExpr::ViewCall(_, args) => {
            for a in args {
                collect_pos_hoists(&a.value, out);
            }
        }
        IrExpr::Binary(_, l, r) => {
            collect_pos_hoists(l, out);
            collect_pos_hoists(r, out);
        }
        IrExpr::Unary(_, r) => collect_pos_hoists(r, out),
        _ => {}
    }
}

/// Name of the hoisted `let` binding for `agents.pos(<local>)`. `self` maps
/// to `self_pos`; other locals map to `<name>_pos`.
fn pos_binding_name(local: &str) -> String {
    if local == "self" {
        "self_pos".into()
    } else {
        format!("{local}_pos")
    }
}

/// Lower a single top-level clause: the negation target inside `if !(...)`.
/// Callers place the result inside `()` — we strip one layer of outer
/// parens from the recursive lowering so the `!(...)` form doesn't
/// produce a double-paren `!((...))` that rustfmt would collapse later.
fn lower_clause(
    node: &IrExprNode,
    hoisted: &[String],
    ctx: EmitContext<'_>,
) -> Result<String, EmitError> {
    let raw = lower_expr_with_hoist(node, hoisted, ctx)?;
    let stripped = if raw.starts_with('(') && raw.ends_with(')') && balanced(&raw) {
        raw[1..raw.len() - 1].to_string()
    } else {
        raw
    };
    Ok(format!("({stripped})"))
}

/// True iff the string's outer `(` matches the outer `)`. Used by
/// `lower_clause` to avoid stripping parens that don't belong to the
/// outermost pair (e.g. `(a) + (b)` — stripping outer `(` / `)` there
/// would produce `a) + (b`).
fn balanced(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'(') || bytes.last() != Some(&b')') {
        return false;
    }
    let mut depth: i32 = 0;
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 && i != bytes.len() - 1 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

// ---------------------------------------------------------------------------
// Expression lowering
// ---------------------------------------------------------------------------

/// Entry point for non-guard lowering (unit tests); equivalent to
/// `lower_expr_with_hoist(node, &[])`.
#[cfg(test)]
fn lower_expr(node: &IrExprNode) -> Result<String, EmitError> {
    lower_expr_with_hoist(node, &[], EmitContext::empty())
}

/// Lower an expression, substituting hoisted `agents.pos(<local>)` calls
/// with their prelude bindings (`self_pos`, `target_pos`, …).
fn lower_expr_with_hoist(
    node: &IrExprNode,
    hoisted: &[String],
    ctx: EmitContext<'_>,
) -> Result<String, EmitError> {
    lower_expr_kind(&node.kind, hoisted, ctx)
}

fn lower_expr_kind(
    kind: &IrExpr,
    hoisted: &[String],
    ctx: EmitContext<'_>,
) -> Result<String, EmitError> {
    match kind {
        IrExpr::LitBool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        IrExpr::LitInt(v) => Ok(format!("{v}")),
        IrExpr::LitFloat(v) => Ok(render_float(*v)),
        IrExpr::LitString(s) => Ok(format!("{s:?}")),
        IrExpr::Local(_, name) => {
            if name == "self" {
                Ok("self_id".into())
            } else {
                Ok(name.clone())
            }
        }
        IrExpr::NamespaceField { ns, field, .. } => lower_namespace_field(*ns, field),
        IrExpr::NamespaceCall { ns, method, args } => {
            // Hoisted agents.pos(<local>) — use the prelude binding.
            if *ns == NamespaceId::Agents && method == "pos" && args.len() == 1 {
                if let IrExpr::Local(_, name) = &args[0].value.kind {
                    if hoisted.iter().any(|h| h == name) {
                        return Ok(pos_binding_name(name));
                    }
                }
            }
            lower_namespace_call(*ns, method, args, hoisted, ctx)
        }
        IrExpr::ViewCall(view_ref, args) => lower_view_call(*view_ref, args, hoisted, ctx),
        IrExpr::BuiltinCall(b, args) => lower_builtin_call(*b, args, hoisted, ctx),
        IrExpr::UnresolvedCall(name, args) => lower_unresolved_call(name, args, hoisted, ctx),
        IrExpr::Binary(op, lhs, rhs) => {
            // `<expr> == None` / `<expr> != None` → `is_none()` /
            // `is_some()`. Handy for Option-valued stdlib calls that
            // the mask predicate wants to compare against a sentinel
            // without importing `Option::None`. Task 157: the `mask
            // Cast` engagement-lock clause is
            // `agents.engaged_with(self) == None`. Only the right-
            // hand-side `None` variant is detected; swapping sides is
            // uncommon enough in practice that we leave it unsupported
            // (the emitter errors out loudly if it appears, so the
            // DSL file can be fixed upstream).
            if matches!(op, BinOp::Eq | BinOp::NotEq) {
                if let IrExpr::EnumVariant { ty, variant } = &rhs.kind {
                    if ty.is_empty() && variant == "None" {
                        let l = lower_expr_with_hoist(lhs, hoisted, ctx)?;
                        let method = if matches!(op, BinOp::Eq) { "is_none" } else { "is_some" };
                        return Ok(format!("{}.{method}()", maybe_paren(&l)));
                    }
                }
            }
            let l = lower_expr_with_hoist(lhs, hoisted, ctx)?;
            let r = lower_expr_with_hoist(rhs, hoisted, ctx)?;
            Ok(format!("({l} {} {r})", binop_str(*op)))
        }
        IrExpr::Unary(op, rhs) => {
            let r = lower_expr_with_hoist(rhs, hoisted, ctx)?;
            Ok(format!("({}{r})", unop_str(*op)))
        }
        other => Err(EmitError::Unsupported(format!(
            "expression shape {other:?} not supported in milestone 4 mask emission"
        ))),
    }
}

/// Lower a `ViewCall(ref, args)` — either a bare `<name>(...)` or
/// `view::<name>(...)` disambiguation. `@lazy` views route through the
/// generated fn; `@materialized` views call `state.views.<name>.get(...)`.
fn lower_view_call(
    view_ref: ViewRef,
    args: &[IrCallArg],
    hoisted: &[String],
    ctx: EmitContext<'_>,
) -> Result<String, EmitError> {
    let view = ctx.view(view_ref).ok_or_else(|| {
        EmitError::Unsupported(format!(
            "view emission: ViewRef({}) has no matching ViewIR in the context",
            view_ref.0
        ))
    })?;
    let stem = snake_case(&view.name);
    let lowered = lower_positional_args(args, &format!("view::{}", view.name), hoisted, ctx)?;
    match view.kind {
        ViewKind::Lazy => {
            let mut argv = vec!["state".to_string()];
            argv.extend(lowered);
            Ok(format!(
                "crate::generated::views::{stem}({})",
                argv.join(", ")
            ))
        }
        ViewKind::Materialized(_) => {
            // `state.views.<name>.get(args..., tick)` — decay views pick
            // up an extra tick arg. The emitter can't know which without
            // consulting the decay hint; emit with `tick` only when the
            // view has @decay set.
            let mut argv = lowered;
            if view.decay.is_some() {
                argv.push("state.tick".to_string());
            }
            Ok(format!("state.views.{stem}.get({})", argv.join(", ")))
        }
    }
}

/// Lower a `NamespaceField` access. Only `config.<block>.<field>` is wired
/// today (it maps onto `SimState.config.<block>.<field>`). Other namespace
/// fields — `world.tick`, `cascade.iterations`, etc. — aren't yet needed by
/// any DSL-owned mask, so they stay `Unsupported` until a rule asks for them.
fn lower_namespace_field(ns: NamespaceId, field: &str) -> Result<String, EmitError> {
    if ns == NamespaceId::Config {
        if field.contains('.') {
            return Ok(format!("state.config.{field}"));
        }
        return Err(EmitError::Unsupported(format!(
            "bare `config.{field}` is not a value; address a specific field"
        )));
    }
    Err(EmitError::Unsupported(format!(
        "namespace-field `{}.{field}` not supported in milestone 4 mask emission",
        ns.name()
    )))
}

fn lower_namespace_call(
    ns: NamespaceId,
    method: &str,
    args: &[IrCallArg],
    hoisted: &[String],
    ctx: EmitContext<'_>,
) -> Result<String, EmitError> {
    let lowered = lower_positional_args(args, &format!("{}.{method}", ns.name()), hoisted, ctx)?;
    match (ns, method) {
        (NamespaceId::Agents, "alive") => {
            expect_arity(args, 1, "agents.alive")?;
            Ok(format!("state.agent_alive({})", lowered[0]))
        }
        (NamespaceId::Agents, "pos") => {
            expect_arity(args, 1, "agents.pos")?;
            // Mask predicates observe post-decrement state, so `agent_pos`
            // is `Some` for every live slot. Dead slots return `None`; we
            // fall back to `Vec3::ZERO` so the predicate keeps a valid
            // f32 without panicking. Alive-check clauses elsewhere in the
            // predicate gate the meaningful result. Note: this arm only
            // fires for `agents.pos(<non-local>)` — `agents.pos(<local>)`
            // is rewritten to the hoisted binding in `lower_expr_kind`.
            Ok(format!(
                "state.agent_pos({}).unwrap_or(glam::Vec3::ZERO)",
                lowered[0]
            ))
        }
        (NamespaceId::Agents, "hp") => {
            expect_arity(args, 1, "agents.hp")?;
            Ok(format!("state.agent_hp({}).unwrap_or(0.0)", lowered[0]))
        }
        (NamespaceId::Agents, "max_hp") => {
            expect_arity(args, 1, "agents.max_hp")?;
            Ok(format!("state.agent_max_hp({}).unwrap_or(1.0)", lowered[0]))
        }
        (NamespaceId::Agents, "shield_hp") => {
            expect_arity(args, 1, "agents.shield_hp")?;
            Ok(format!("state.agent_shield_hp({}).unwrap_or(0.0)", lowered[0]))
        }
        (NamespaceId::Agents, "hunger") => {
            expect_arity(args, 1, "agents.hunger")?;
            Ok(format!("state.agent_hunger({}).unwrap_or(0.0)", lowered[0]))
        }
        (NamespaceId::Agents, "thirst") => {
            expect_arity(args, 1, "agents.thirst")?;
            Ok(format!("state.agent_thirst({}).unwrap_or(0.0)", lowered[0]))
        }
        (NamespaceId::Agents, "rest_timer") => {
            expect_arity(args, 1, "agents.rest_timer")?;
            Ok(format!("state.agent_rest_timer({}).unwrap_or(0.0)", lowered[0]))
        }
        // `agents.engaged_with(id)` → `state.agent_engaged_with(id)`. The
        // accessor returns `Option<AgentId>`; the `mask Cast` predicate
        // compares against `None`, which the `== None` / `!= None`
        // rewrite in `lower_expr_kind` routes through `is_none()` /
        // `is_some()`. Task 157.
        (NamespaceId::Agents, "engaged_with") => {
            expect_arity(args, 1, "agents.engaged_with")?;
            Ok(format!("state.agent_engaged_with({})", lowered[0]))
        }
        // `abilities.known(agent, ability)` — the mask-side sibling of
        // the physics-side `abilities.is_known(ability)`. The first arg
        // (agent) is ignored: the registry is sim-wide and the mask
        // gate does not yet key on per-agent spellbooks. Task 157.
        (NamespaceId::Abilities, "known") => {
            expect_arity(args, 2, "abilities.known")?;
            Ok(format!("state.ability_registry.get({}).is_some()", lowered[1]))
        }
        // `abilities.cooldown_ready(agent, ability)` — gates on BOTH
        // the per-agent global cooldown (GCD) and the per-(agent, slot)
        // local cooldown. Introduced at task 157 with a single-cursor
        // gate; migrated 2026-04-22 by the ability-cooldowns subsystem
        // (Task 6) to route through `SimState::can_cast_ability`, which
        // reads the dual-cursor pair. The `ability` arg carries a live
        // `AbilityId` from the `mask Cast` runtime signature; we call
        // `.slot()` on it to pick out the local-cooldown slot, then
        // bound-cast to `u8` (helper's signature is `u8`; the registry
        // caps ability slots at `MAX_ABILITIES = 8`, so the truncation
        // is lossless).
        (NamespaceId::Abilities, "cooldown_ready") => {
            expect_arity(args, 2, "abilities.cooldown_ready")?;
            Ok(format!(
                "state.can_cast_ability({}, ({}).slot() as u8, state.tick as u32)",
                lowered[0], lowered[1]
            ))
        }
        // `abilities.hostile_only(ability)` reads `AbilityProgram.gate
        // .hostile_only`. Unknown ids fall back to `false` — same
        // permissive behaviour as `abilities.is_known` in physics
        // emission (an unregistered id can't constrain anything).
        (NamespaceId::Abilities, "hostile_only") => {
            expect_arity(args, 1, "abilities.hostile_only")?;
            Ok(format!(
                "state.ability_registry.get({}).map(|p| p.gate.hostile_only).unwrap_or(false)",
                lowered[0]
            ))
        }
        // `abilities.range(ability)` reads `AbilityProgram.area` (MVP
        // ships `Area::SingleTarget { range }` only). Unknown ids
        // yield `0.0` — the surrounding distance clause will reject.
        (NamespaceId::Abilities, "range") => {
            expect_arity(args, 1, "abilities.range")?;
            Ok(format!(
                "state.ability_registry.get({}).map(|p| match p.area {{ crate::ability::Area::SingleTarget {{ range }} => range }}).unwrap_or(0.0)",
                lowered[0]
            ))
        }
        _ => Err(EmitError::Unsupported(format!(
            "stdlib call `{}.{method}` not supported in milestone 4 mask emission",
            ns.name()
        ))),
    }
}

fn lower_builtin_call(
    b: Builtin,
    args: &[IrCallArg],
    hoisted: &[String],
    ctx: EmitContext<'_>,
) -> Result<String, EmitError> {
    let lowered = lower_positional_args(args, b.name(), hoisted, ctx)?;
    match b {
        Builtin::Distance => {
            expect_arity(args, 2, "distance")?;
            let recv = maybe_paren(&lowered[0]);
            Ok(format!("{recv}.distance({})", lowered[1]))
        }
        Builtin::PlanarDistance => {
            expect_arity(args, 2, "planar_distance")?;
            let lhs = maybe_paren(&lowered[0]);
            let rhs = maybe_paren(&lowered[1]);
            Ok(format!("{lhs}.truncate().distance({rhs}.truncate())"))
        }
        Builtin::Min => {
            expect_arity(args, 2, "min")?;
            Ok(format!("({}).min({})", lowered[0], lowered[1]))
        }
        Builtin::Max => {
            expect_arity(args, 2, "max")?;
            Ok(format!("({}).max({})", lowered[0], lowered[1]))
        }
        Builtin::Abs => {
            expect_arity(args, 1, "abs")?;
            Ok(format!("({}).abs()", lowered[0]))
        }
        _ => Err(EmitError::Unsupported(format!(
            "builtin `{}` not supported in milestone 4 mask emission",
            b.name()
        ))),
    }
}

/// Wrap `s` in parens unless it's already a simple identifier / accessor
/// chain that rustfmt wouldn't add parens to. Reduces `((x)).method()`
/// to `x.method()` for the hoisted-binding common case.
fn maybe_paren(s: &str) -> String {
    let is_simple = s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.');
    if is_simple { s.to_string() } else { format!("({s})") }
}

/// Lower a call whose callee couldn't be resolved by the compiler. These
/// are game-view fns the DSL mentions that haven't been lifted into a
/// `view` declaration yet. The convention: `UnresolvedCall("foo", args)`
/// emits `crate::rules::foo(state, args...)`. The `rules` module was
/// retired in task 140 when the last shim (`is_hostile`) became a DSL
/// view, so any new `UnresolvedCall` will fail `rustc`'s link. Prefer
/// declaring a `view` in `assets/sim/views.sim` over reviving the shim.
fn lower_unresolved_call(
    name: &str,
    args: &[IrCallArg],
    hoisted: &[String],
    ctx: EmitContext<'_>,
) -> Result<String, EmitError> {
    let lowered = lower_positional_args(args, name, hoisted, ctx)?;
    let mut argv = vec!["state".to_string()];
    argv.extend(lowered);
    Ok(format!("crate::rules::{name}({})", argv.join(", ")))
}

fn lower_positional_args(
    args: &[IrCallArg],
    call_name: &str,
    hoisted: &[String],
    ctx: EmitContext<'_>,
) -> Result<Vec<String>, EmitError> {
    args.iter()
        .map(|a| {
            if a.name.is_some() {
                return Err(EmitError::Unsupported(format!(
                    "named argument on call `{call_name}` not supported in milestone 4"
                )));
            }
            lower_expr_with_hoist(&a.value, hoisted, ctx)
        })
        .collect()
}

fn expect_arity(args: &[IrCallArg], expected: usize, name: &str) -> Result<(), EmitError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(EmitError::Unsupported(format!(
            "`{name}` expects {expected} arg(s), got {}",
            args.len()
        )))
    }
}

fn binop_str(op: BinOp) -> &'static str {
    match op {
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::Eq => "==",
        BinOp::NotEq => "!=",
        BinOp::Lt => "<",
        BinOp::LtEq => "<=",
        BinOp::Gt => ">",
        BinOp::GtEq => ">=",
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
    }
}

fn unop_str(op: UnOp) -> &'static str {
    match op {
        UnOp::Not => "!",
        UnOp::Neg => "-",
    }
}

/// Map an `IrType` used as a mask-head positional param onto its emitted
/// Rust type string. Only the handful of id / small-scalar types the
/// mask surface accepts are supported; everything else errors out at
/// emit time so an unsupported head type doesn't silently lower to a
/// confusing signature. Task 157.
fn render_head_type(ty: &IrType, mask_name: &str) -> Result<String, EmitError> {
    let rendered = match ty {
        IrType::AgentId => "AgentId",
        // Fully-qualified engine paths — the mask file emits only
        // `use crate::ids::AgentId;`, so the rarer id types path
        // through their owning module to avoid bloating the import
        // preamble for every mask file.
        IrType::AbilityId => "crate::ability::AbilityId",
        IrType::ItemId => "crate::ids::ItemId",
        IrType::GroupId => "crate::ids::GroupId",
        IrType::QuestId => "crate::ids::QuestId",
        IrType::EventId => "crate::ids::EventId",
        IrType::AuctionId => "crate::ids::AuctionId",
        _ => {
            return Err(EmitError::Unsupported(format!(
                "mask `{mask_name}` head param type {ty:?} not supported"
            )));
        }
    };
    Ok(rendered.to_string())
}

/// Render a float so it parses back identically. `f64::to_string` drops the
/// trailing `.0` for whole numbers (`3.0` → `"3"`), which would type-infer
/// to an integer in Rust. Always emit at least one fractional digit so
/// the literal stays `f32`/`f64`-typed wherever it's used.
fn render_float(v: f64) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') || s == "inf" || s == "-inf" || s == "NaN"
    {
        s
    } else {
        format!("{s}.0")
    }
}

// ---------------------------------------------------------------------------
// Naming utilities — kept identical to emit_physics.rs. If either drifts,
// both must move to a shared util module.
// ---------------------------------------------------------------------------

fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    let mut prev_upper = false;
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !prev_upper {
                out.push('_');
            }
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            prev_upper = true;
        } else {
            out.push(ch);
            prev_upper = false;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::ir::{
        IrActionHead, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, LocalRef, MaskIR,
        NamespaceId,
    };

    fn span() -> Span {
        Span::dummy()
    }

    fn local(name: &str, id: u16) -> IrExprNode {
        IrExprNode { kind: IrExpr::Local(LocalRef(id), name.to_string()), span: span() }
    }

    fn ns_call(ns: NamespaceId, method: &str, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::NamespaceCall {
                ns,
                method: method.to_string(),
                args: args
                    .into_iter()
                    .map(|a| IrCallArg { name: None, value: a, span: span() })
                    .collect(),
            },
            span: span(),
        }
    }

    fn unresolved(name: &str, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::UnresolvedCall(
                name.to_string(),
                args.into_iter()
                    .map(|a| IrCallArg { name: None, value: a, span: span() })
                    .collect(),
            ),
            span: span(),
        }
    }

    fn builtin(b: Builtin, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::BuiltinCall(
                b,
                args.into_iter()
                    .map(|a| IrCallArg { name: None, value: a, span: span() })
                    .collect(),
            ),
            span: span(),
        }
    }

    fn binop(op: BinOp, lhs: IrExprNode, rhs: IrExprNode) -> IrExprNode {
        IrExprNode { kind: IrExpr::Binary(op, Box::new(lhs), Box::new(rhs)), span: span() }
    }

    fn lit_float(v: f64) -> IrExprNode {
        IrExprNode { kind: IrExpr::LitFloat(v), span: span() }
    }

    /// Build the attack-mask IR that `masks.sim` lowers to. Predicate:
    /// `agents.alive(target) && is_hostile(self, target) &&
    ///  distance(agents.pos(self), agents.pos(target)) < 2.0`.
    fn attack_mask_ir() -> MaskIR {
        let self_local = local("self", 0);
        let target_local = local("target", 1);

        let alive = ns_call(NamespaceId::Agents, "alive", vec![target_local.clone()]);
        let hostile = unresolved(
            "is_hostile",
            vec![self_local.clone(), target_local.clone()],
        );
        let self_pos = ns_call(NamespaceId::Agents, "pos", vec![self_local.clone()]);
        let target_pos = ns_call(NamespaceId::Agents, "pos", vec![target_local.clone()]);
        let dist = builtin(Builtin::Distance, vec![self_pos, target_pos]);
        let lt = binop(BinOp::Lt, dist, lit_float(2.0));
        let and1 = binop(BinOp::And, alive, hostile);
        let predicate = binop(BinOp::And, and1, lt);

        MaskIR {
            head: IrActionHead {
                name: "Attack".into(),
                shape: IrActionHeadShape::Positional(vec![(
                    "target".into(),
                    LocalRef(1),
                    IrType::AgentId,
                )]),
                span: span(),
            },
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(),
        }
    }

    #[test]
    fn attack_mask_emits_expected_shape() {
        let mask = attack_mask_ir();
        let out = emit_mask(&mask, Some("assets/sim/masks.sim")).unwrap();

        // Header + imports.
        assert!(out.contains("// GENERATED by dsl_compiler from assets/sim/masks.sim."));
        assert!(out.contains("use crate::ids::AgentId;"));
        assert!(out.contains("use crate::state::SimState;"));

        // Fn signature with `self_id` + head-argument `target`.
        assert!(
            out.contains(
                "pub fn mask_attack(state: &SimState, self_id: AgentId, target: AgentId) -> bool {"
            ),
            "unexpected signature in:\n{out}"
        );

        // Hoisted pos bindings — one per unique local referenced via
        // agents.pos(...). `self` → `self_pos` (from `self_id`); `target`
        // → `target_pos` (from `target`).
        assert!(
            out.contains("let self_pos = state.agent_pos(self_id).unwrap_or(glam::Vec3::ZERO);"),
            "missing hoisted self_pos in:\n{out}"
        );
        assert!(
            out.contains("let target_pos = state.agent_pos(target).unwrap_or(glam::Vec3::ZERO);"),
            "missing hoisted target_pos in:\n{out}"
        );

        // Alive clause.
        assert!(
            out.contains("if !(state.agent_alive(target)) {"),
            "missing alive guard in:\n{out}"
        );

        // Unresolved `is_hostile` call routed through `crate::rules::*`.
        assert!(
            out.contains("if !(crate::rules::is_hostile(state, self_id, target)) {"),
            "missing is_hostile guard in:\n{out}"
        );

        // Distance clause using the hoisted bindings and short-circuit
        // multi-line `if` body.
        assert!(
            out.contains("if !(self_pos.distance(target_pos) < 2.0) {"),
            "missing distance guard in:\n{out}"
        );
        assert!(
            out.contains("        return false;"),
            "missing return in guard body:\n{out}"
        );
        // Trailing fallthrough.
        assert!(out.contains("    true\n}"), "missing true fallthrough in:\n{out}");
    }

    #[test]
    fn lower_expr_local_rewrites_self() {
        let out = lower_expr(&local("self", 0)).unwrap();
        assert_eq!(out, "self_id");
    }

    #[test]
    fn aggregate_mod_reexports_mask_fn() {
        let mask = attack_mask_ir();
        let out = emit_mask_mod(std::slice::from_ref(&mask));
        assert!(out.contains("pub mod attack;"));
        assert!(out.contains("pub use attack::mask_attack;"));
        assert!(out.contains("pub fn register()"));
    }

    #[test]
    fn empty_aggregate_mod_is_a_no_op() {
        let out = emit_mask_mod(&[]);
        assert!(out.contains("pub fn register() {}"));
        assert!(!out.contains("pub mod"));
    }

    #[test]
    fn view_call_to_lazy_view_emits_crate_generated_path() {
        use crate::ir::{
            DecayHint, IrParam, IrType, ViewBodyIR, ViewIR, ViewKind, ViewRef,
        };
        // Synthesise a mask that calls `view::is_hostile(self, target)` via
        // a ViewCall node. With a @lazy view in ctx, the lowering should
        // route through `crate::generated::views::is_hostile(...)`.
        let lazy_view = ViewIR {
            name: "is_hostile".into(),
            params: vec![
                IrParam {
                    name: "a".into(),
                    local: LocalRef(0),
                    ty: IrType::Named("Agent".into()),
                    span: span(),
                },
                IrParam {
                    name: "b".into(),
                    local: LocalRef(1),
                    ty: IrType::Named("Agent".into()),
                    span: span(),
                },
            ],
            return_ty: IrType::Bool,
            body: ViewBodyIR::Expr(IrExprNode {
                kind: IrExpr::LitBool(true),
                span: span(),
            }),
            annotations: vec![],
            kind: ViewKind::Lazy,
            decay: None,
            span: span(),
        };
        let views = vec![lazy_view];
        let ctx = EmitContext { views: &views };

        let self_local = local("self", 0);
        let target_local = local("target", 1);
        let view_call = IrExprNode {
            kind: IrExpr::ViewCall(
                ViewRef(0),
                vec![
                    IrCallArg { name: None, value: self_local.clone(), span: span() },
                    IrCallArg { name: None, value: target_local.clone(), span: span() },
                ],
            ),
            span: span(),
        };
        let alive = ns_call(NamespaceId::Agents, "alive", vec![target_local.clone()]);
        let predicate = binop(BinOp::And, alive, view_call);
        let mask = MaskIR {
            head: IrActionHead {
                name: "Attack".into(),
                shape: IrActionHeadShape::Positional(vec![(
                    "target".into(),
                    LocalRef(1),
                    IrType::AgentId,
                )]),
                span: span(),
            },
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(),
        };
        let out = emit_mask_with_ctx(&mask, None, ctx).unwrap();
        assert!(
            out.contains("crate::generated::views::is_hostile(state, self_id, target)"),
            "missing lazy view call in:\n{out}"
        );
        let _ = DecayHint {
            rate: 0.5,
            per: crate::ir::DecayUnit::Tick,
            span: span(),
        };
    }

    #[test]
    fn view_call_to_materialized_with_decay_emits_state_views_get_with_tick() {
        use crate::ir::{
            DecayHint, DecayUnit, FoldHandlerIR, IrEventPattern, IrParam, IrType, StorageHint,
            ViewBodyIR, ViewIR, ViewKind, ViewRef,
        };
        let mat_view = ViewIR {
            name: "threat_level".into(),
            params: vec![
                IrParam {
                    name: "a".into(),
                    local: LocalRef(0),
                    ty: IrType::Named("Agent".into()),
                    span: span(),
                },
                IrParam {
                    name: "b".into(),
                    local: LocalRef(1),
                    ty: IrType::Named("Agent".into()),
                    span: span(),
                },
            ],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: IrExprNode { kind: IrExpr::LitFloat(0.0), span: span() },
                handlers: vec![FoldHandlerIR {
                    pattern: IrEventPattern {
                        name: "AgentAttacked".into(),
                        event: None,
                        bindings: vec![],
                        span: span(),
                    },
                    body: vec![],
                    span: span(),
                }],
                clamp: None,
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PairMap),
            decay: Some(DecayHint {
                rate: 0.98,
                per: DecayUnit::Tick,
                span: span(),
            }),
            span: span(),
        };
        let views = vec![mat_view];
        let ctx = EmitContext { views: &views };

        // `view::threat_level(self, target) > 0.0` as the mask predicate.
        let self_local = local("self", 0);
        let target_local = local("target", 1);
        let view_call = IrExprNode {
            kind: IrExpr::ViewCall(
                ViewRef(0),
                vec![
                    IrCallArg { name: None, value: self_local.clone(), span: span() },
                    IrCallArg { name: None, value: target_local.clone(), span: span() },
                ],
            ),
            span: span(),
        };
        let zero = lit_float(0.0);
        let predicate = binop(BinOp::Gt, view_call, zero);
        let mask = MaskIR {
            head: IrActionHead {
                name: "Attack".into(),
                shape: IrActionHeadShape::Positional(vec![(
                    "target".into(),
                    LocalRef(1),
                    IrType::AgentId,
                )]),
                span: span(),
            },
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(),
        };
        let out = emit_mask_with_ctx(&mask, None, ctx).unwrap();
        assert!(
            out.contains("state.views.threat_level.get(self_id, target, state.tick)"),
            "missing materialized view call with tick in:\n{out}"
        );
    }

    #[test]
    fn named_action_head_is_rejected() {
        let mut mask = attack_mask_ir();
        mask.head.shape = IrActionHeadShape::Named(vec![]);
        let err = emit_mask(&mask, None).unwrap_err();
        assert!(err.contains("named-field action head"), "unexpected err: {err}");
    }
}
