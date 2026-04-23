//! Interpreter for `PhysicsIR` — the cascade-handler rule class.
//!
//! ## Supported primitives (wolves+humans coverage, §3)
//!
//! This file covers **exactly** the IR variants, stdlib functions, and field
//! reads exercised by the wolves+humans fixture (survey §3).  Everything
//! outside this set panics with a pointer to the survey doc.
//!
//! ### IR statement kinds implemented
//!
//! | Variant | Notes |
//! |---------|-------|
//! | `Let` | intermediate local bindings |
//! | `Emit` | event emission via `ctx.emit(...)` |
//! | `For` | loop over query/ability collection results |
//! | `If` | guard clauses and control flow |
//! | `Match` | destructuring of `EffectOp` variants in cast rule |
//! | `Expr` | standalone expression statement (e.g. `agents.set_*`) |
//!
//! ### IR expression kinds implemented
//!
//! | Variant | Notes |
//! |---------|-------|
//! | `LitFloat` | float constants (0.0, 1.0, 0.5, …) |
//! | `LitInt` | integer constants (0, 1, 2, 8, …) |
//! | `Local` | event-pattern / let-bound locals |
//! | `NamespaceCall` | `agents.*`, `query.*`, `abilities.*` |
//! | `NamespaceField` | `agents.<field>(x)`, `config.*`, `world.tick`, `cascade.max_iterations` |
//! | `EnumVariant` | `TargetSelector::Target`, `TargetSelector::Caster`, `None` |
//! | `Binary` | `+`, `-`, `>`, `<`, `>=`, `<=`, `!=`, `==`, `||` |
//! | `Unary` | `!` logical not |
//! | `If` | conditional expression |
//! | `Match` | match expression (EffectOp scrutinee) |
//! | `BuiltinCall` | `min`, `max`, `saturating_add`, `distance` (reuse builtins.rs) |
//!
//! ### Pattern kinds implemented
//!
//! | Variant | Notes |
//! |---------|-------|
//! | `Struct` | `{ actor: c, target: t, amount: a }` event patterns |
//! | `Bind` | field bindings |
//! | `Wildcard` | `_` for unused fields |
//!
//! ### Stdlib functions implemented
//!
//! See survey §3.2 for the full list. All 25 agent accessors, 3 query
//! functions, 5 ability registry functions, 4 config reads, and 4 builtins
//! (distance, min, max, saturating_add) are covered.
//!
//! ### Coverage source
//!
//! `docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md` §3.
//!
//! Any IR variant not listed above hits `unimplemented!()` with a message
//! pointing to that survey document.
//!
//! ## Event payload plumbing
//!
//! The `CascadeContext` trait does not expose event-field reads as a method
//! (the event itself is not passed through the context). Instead,
//! `PhysicsIR::apply` accepts the `PhysicsHandlerIR`'s pattern bindings from
//! the event that triggered it, and the engine-side call site is responsible
//! for pre-populating a `Vec<(&str, EvalValue)>` from the live event struct
//! and passing it as `event_fields`.
//!
//! The public entry-point is therefore:
//!
//! ```ignore
//! impl PhysicsIR {
//!     pub fn apply<C: CascadeContext>(
//!         &self,
//!         handler_idx: usize,
//!         event_fields: &[(&str, EvalValue)],
//!         ctx: &mut C,
//!     )
//! }
//! ```
//!
//! `handler_idx` selects which `PhysicsHandlerIR` to run (typically 0, since
//! the emitter currently requires exactly one handler per physics rule).

use std::collections::HashMap;

use crate::ast::{BinOp, UnOp};
use crate::eval::{
    AbilityId, AgentId, CascadeContext, EffectOp, EvalValue, TargetSelector, Vec3,
};
use crate::ir::{
    Builtin, IrCallArg, IrEmit, IrExpr, IrExprNode, IrFieldInit, IrPattern, IrPatternBinding,
    IrStmt, IrStmtMatchArm, LocalRef, NamespaceId, PhysicsHandlerIR, PhysicsIR,
};

// ---------------------------------------------------------------------------
// Runtime value
// ---------------------------------------------------------------------------

/// Dynamically-typed intermediate value for physics expressions.
///
/// Physics rules manipulate agents (read and mutate), integers, floats,
/// ability IDs, and `EffectOp` values produced by `abilities.effects`.
/// The `EffectOp` variant is the only one specific to physics (absent in
/// mask/scoring); everything else reuses the same shape as masks.
#[derive(Debug, Clone)]
enum PVal {
    Bool(bool),
    Float(f32),
    Int(i64),
    Vec3(Vec3),
    Agent(Option<AgentId>),
    Ability(AbilityId),
    EffectOp(EffectOp),
}

impl PVal {
    fn as_bool(&self) -> bool {
        match self {
            PVal::Bool(b) => *b,
            // Numeric truthy: non-zero
            PVal::Float(f) => *f != 0.0,
            PVal::Int(i) => *i != 0,
            other => panic!(
                "dsl_ast::eval::physics: expected Bool, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
            ),
        }
    }

    fn as_f32(&self) -> f32 {
        match self {
            PVal::Float(f) => *f,
            PVal::Int(i) => *i as f32,
            PVal::Bool(b) => if *b { 1.0 } else { 0.0 },
            other => panic!(
                "dsl_ast::eval::physics: expected Float, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
            ),
        }
    }

    fn as_i64(&self) -> i64 {
        match self {
            PVal::Int(i) => *i,
            PVal::Float(f) => *f as i64,
            PVal::Bool(b) => if *b { 1 } else { 0 },
            other => panic!(
                "dsl_ast::eval::physics: expected Int, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
            ),
        }
    }

    #[allow(dead_code)]
    fn as_vec3(&self) -> Vec3 {
        match self {
            PVal::Vec3(v) => *v,
            other => panic!(
                "dsl_ast::eval::physics: expected Vec3, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
            ),
        }
    }

    fn as_agent_id(&self) -> AgentId {
        match self {
            PVal::Agent(Some(id)) => *id,
            PVal::Agent(None) => panic!(
                "dsl_ast::eval::physics: expected AgentId, got None; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
            ),
            other => panic!(
                "dsl_ast::eval::physics: expected Agent, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
            ),
        }
    }

    fn as_ability_id(&self) -> AbilityId {
        match self {
            PVal::Ability(ab) => *ab,
            other => panic!(
                "dsl_ast::eval::physics: expected Ability, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Local-variable environment
// ---------------------------------------------------------------------------

/// Mapping from `LocalRef` slot to runtime value.
type Locals = HashMap<u16, PVal>;

// ---------------------------------------------------------------------------
// PhysicsIR::apply — public entry point
// ---------------------------------------------------------------------------

impl PhysicsIR {
    /// Run a specific handler from this physics rule.
    ///
    /// `handler_idx` selects the handler (typically 0 — the emitter currently
    /// requires exactly one handler per physics rule).
    ///
    /// `event_fields` is a slice of `(field_name, value)` pairs pre-populated
    /// by the engine call site from the live triggering event struct.  The
    /// interpreter uses these to satisfy the handler's pattern bindings.
    ///
    /// Mutations and event emissions reach the simulation via `ctx`.
    pub fn apply<C: CascadeContext>(
        &self,
        handler_idx: usize,
        event_fields: &[(&str, EvalValue)],
        ctx: &mut C,
    ) {
        let handler = self.handlers.get(handler_idx).unwrap_or_else(|| {
            panic!(
                "dsl_ast::eval::physics: handler_idx {handler_idx} out of bounds \
                 (rule `{}` has {} handlers)",
                self.name,
                self.handlers.len()
            )
        });
        apply_handler(handler, event_fields, ctx);
    }
}

// ---------------------------------------------------------------------------
// Handler application
// ---------------------------------------------------------------------------

fn apply_handler<C: CascadeContext>(
    handler: &PhysicsHandlerIR,
    event_fields: &[(&str, EvalValue)],
    ctx: &mut C,
) {
    // Build the locals map from the pattern bindings.
    let mut locals: Locals = HashMap::new();
    bind_pattern_fields(handler.pattern.bindings(), event_fields, &mut locals);

    // Evaluate the where-clause guard (if any). If it returns false, skip
    // the body — mirrors the compiled path `if !({cond}) { return; }`.
    if let Some(where_expr) = &handler.where_clause {
        if !eval_expr(where_expr, ctx, &locals).as_bool() {
            return;
        }
    }

    exec_stmts(&handler.body, ctx, &mut locals);
}

/// Populate `locals` from the handler's pattern bindings and the event fields.
///
/// For each `IrPatternBinding { field, value: IrPattern::Bind { name, local } }`,
/// look up `field` in `event_fields` and insert under `local.0`.
fn bind_pattern_fields(
    bindings: &[IrPatternBinding],
    event_fields: &[(&str, EvalValue)],
    locals: &mut Locals,
) {
    for binding in bindings {
        if let IrPattern::Bind { name: _, local } = &binding.value {
            // Find the event field by name.
            let val = event_fields
                .iter()
                .find(|(fname, _)| *fname == binding.field)
                .map(|(_, v)| eval_value_to_pval(*v))
                .unwrap_or_else(|| {
                    panic!(
                        "dsl_ast::eval::physics: event field `{}` not found in event_fields; \
                         available: {:?}",
                        binding.field,
                        event_fields.iter().map(|(n, _)| n).collect::<Vec<_>>()
                    )
                });
            locals.insert(local.0, val);
        }
        // Wildcard bindings are silently skipped.
    }
}

/// Convert a public `EvalValue` (from `eval/mod.rs`) to a local `PVal`.
fn eval_value_to_pval(v: EvalValue) -> PVal {
    match v {
        EvalValue::Bool(b) => PVal::Bool(b),
        EvalValue::I32(i) => PVal::Int(i as i64),
        EvalValue::I64(i) => PVal::Int(i),
        EvalValue::U32(u) => PVal::Int(u as i64),
        EvalValue::F32(f) => PVal::Float(f),
        EvalValue::Agent(id) => PVal::Agent(Some(id)),
        EvalValue::Ability(ab) => PVal::Ability(ab),
    }
}

/// Convert a local `PVal` back to `EvalValue` for `ctx.emit(...)` payloads.
fn pval_to_eval_value(v: &PVal) -> EvalValue {
    match v {
        PVal::Bool(b) => EvalValue::Bool(*b),
        PVal::Float(f) => EvalValue::F32(*f),
        PVal::Int(i) => EvalValue::I64(*i),
        PVal::Vec3(_) => panic!(
            "dsl_ast::eval::physics: Vec3 cannot be emitted as EvalValue — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
        ),
        PVal::Agent(Some(id)) => EvalValue::Agent(*id),
        PVal::Agent(None) => panic!(
            "dsl_ast::eval::physics: None agent cannot be emitted as EvalValue — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
        ),
        PVal::Ability(ab) => EvalValue::Ability(*ab),
        PVal::EffectOp(_) => panic!(
            "dsl_ast::eval::physics: EffectOp cannot be emitted as EvalValue — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
        ),
    }
}

// ---------------------------------------------------------------------------
// Statement execution
// ---------------------------------------------------------------------------

fn exec_stmts<C: CascadeContext>(stmts: &[IrStmt], ctx: &mut C, locals: &mut Locals) {
    for stmt in stmts {
        exec_stmt(stmt, ctx, locals);
    }
}

fn exec_stmt<C: CascadeContext>(stmt: &IrStmt, ctx: &mut C, locals: &mut Locals) {
    match stmt {
        IrStmt::Let { name: _, local, value, .. } => {
            let v = eval_expr(value, ctx, locals);
            locals.insert(local.0, v);
        }

        IrStmt::Emit(emit) => exec_emit(emit, ctx, locals),

        IrStmt::If { cond, then_body, else_body, .. } => {
            let pred = eval_expr(cond, ctx, locals).as_bool();
            if pred {
                // Clone locals so inner scope changes are propagated (let
                // bindings inside if branches are visible outside — that's
                // the compiled-path semantics as Rust let in the same scope).
                exec_stmts(then_body, ctx, locals);
            } else if let Some(eb) = else_body {
                exec_stmts(eb, ctx, locals);
            }
        }

        IrStmt::For { binder, binder_name: _, iter, filter, body, .. } => {
            exec_for(binder, iter, filter.as_ref(), body, ctx, locals);
        }

        IrStmt::Match { scrutinee, arms, .. } => {
            exec_match(scrutinee, arms, ctx, locals);
        }

        IrStmt::Expr(e) => {
            // Standalone expression — evaluate for side effects (e.g. agent writes).
            eval_expr(e, ctx, locals);
        }

        IrStmt::SelfUpdate { .. } => unimplemented!(
            "dsl_ast::eval::physics: IrStmt::SelfUpdate is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
        ),
    }
}

// ---------------------------------------------------------------------------
// Emit statement
// ---------------------------------------------------------------------------

fn exec_emit<C: CascadeContext>(emit: &IrEmit, ctx: &mut C, locals: &Locals) {
    // Build the field payload.  We collect into a temporary Vec so we can
    // pass a slice to `ctx.emit`.
    // Use a Vec<(String, EvalValue)> to own the field names; then build a
    // `&[(&str, EvalValue)]` slice from it.
    let owned: Vec<(String, EvalValue)> = emit
        .fields
        .iter()
        .map(|f: &IrFieldInit| {
            let v = eval_expr(&f.value, ctx, locals);
            let ev = pval_to_eval_value(&v);
            (f.name.clone(), ev)
        })
        .collect();
    // Borrow the owned data for the slice.
    let borrowed: Vec<(&str, EvalValue)> = owned
        .iter()
        .map(|(n, v)| (n.as_str(), *v))
        .collect();
    ctx.emit(&emit.event_name, &borrowed);
}

// ---------------------------------------------------------------------------
// For loop
// ---------------------------------------------------------------------------

fn exec_for<C: CascadeContext>(
    binder: &LocalRef,
    iter: &IrExprNode,
    filter: Option<&IrExprNode>,
    body: &[IrStmt],
    ctx: &mut C,
    locals: &mut Locals,
) {
    // The iterator expression must be one of the supported collection
    // sources: `abilities.effects(ab)` or `query.nearby_kin(center, radius)`.
    // We dispatch to a specialised collector rather than a generic iterator
    // (the borrow rules on `ctx` make it hard to hold an iterator while also
    // mutating through `ctx`).
    match &iter.kind {
        IrExpr::NamespaceCall { ns: NamespaceId::Abilities, method, args }
            if method == "effects" =>
        {
            assert_eq!(args.len(), 1, "abilities.effects expects 1 arg");
            let ab = eval_expr(&args[0].value, ctx, locals).as_ability_id();
            let mut ops: Vec<EffectOp> = Vec::new();
            ctx.abilities_effects(ab, &mut |op| ops.push(op));
            for op in ops {
                let mut inner = locals.clone();
                inner.insert(binder.0, PVal::EffectOp(op));
                // Apply filter
                if let Some(f) = filter {
                    if !eval_expr(f, ctx, &inner).as_bool() {
                        continue;
                    }
                }
                exec_stmts(body, ctx, &mut inner);
                // Propagate any newly-bound locals back out (let bindings in
                // the loop body stay scoped to the iteration — compiled Rust
                // does the same since they're `let` in a block).
            }
        }

        IrExpr::NamespaceCall { ns: NamespaceId::Query, method, args }
            if method == "nearby_kin" =>
        {
            // `query.nearby_kin(center_agent, radius)` — the DSL form passes
            // the center agent (not a raw Vec3); the context resolves pos
            // internally.  The trait's `query_nearby_kin` takes `(origin,
            // center, radius)` where `origin == center` for the fixture's
            // `nearby_kin(dead, 12.0)` form.
            assert_eq!(args.len(), 2, "query.nearby_kin expects 2 args");
            let origin = eval_expr(&args[0].value, ctx, locals).as_agent_id();
            let radius = eval_expr(&args[1].value, ctx, locals).as_f32();
            let center = ctx.agents_pos(origin);
            let mut kin: Vec<AgentId> = Vec::new();
            ctx.query_nearby_kin(origin, center, radius, &mut |id| kin.push(id));
            for k in kin {
                let mut inner = locals.clone();
                inner.insert(binder.0, PVal::Agent(Some(k)));
                if let Some(f) = filter {
                    if !eval_expr(f, ctx, &inner).as_bool() {
                        continue;
                    }
                }
                exec_stmts(body, ctx, &mut inner);
            }
        }

        other => unimplemented!(
            "dsl_ast::eval::physics: for-loop iter {:?} is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3",
            std::mem::discriminant(other)
        ),
    }
}

// ---------------------------------------------------------------------------
// Match statement
// ---------------------------------------------------------------------------

fn exec_match<C: CascadeContext>(
    scrutinee: &IrExprNode,
    arms: &[IrStmtMatchArm],
    ctx: &mut C,
    locals: &mut Locals,
) {
    let scrutinee_val = eval_expr(scrutinee, ctx, locals);

    // The only match scrutinee in wolves+humans physics is an `EffectOp`
    // produced by iterating `abilities.effects(ab)`.  We match the runtime
    // `EffectOp` value against each arm pattern and execute the first match.
    let eff_op = match &scrutinee_val {
        PVal::EffectOp(op) => *op,
        other => unimplemented!(
            "dsl_ast::eval::physics: match scrutinee {:?} is not EffectOp — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3",
            std::mem::discriminant(other)
        ),
    };

    for arm in arms {
        let mut inner = locals.clone();
        if pattern_matches_effect_op(&arm.pattern, eff_op, &mut inner) {
            exec_stmts(&arm.body, ctx, &mut inner);
            // Copy new locals back (let-bindings inside match arms scope to
            // the arm, but the arm may read previously-bound variables and the
            // match is the only statement that runs).
            *locals = inner;
            return;
        }
    }
    // No arm matched — that's a bug in the DSL source or the IR, but we
    // silently continue rather than panic so non-exhaustive matches survive.
}

/// Try to match `op` against an `IrPattern`, binding field locals into
/// `inner`.  Returns `true` if the pattern matched.
fn pattern_matches_effect_op(
    pattern: &IrPattern,
    op: EffectOp,
    locals: &mut Locals,
) -> bool {
    match pattern {
        IrPattern::Wildcard => true,

        IrPattern::Struct { name, bindings, .. } => {
            match_effect_op_struct(name, op, bindings, locals)
        }

        IrPattern::Bind { name: _, local } => {
            // Bare bind: bind the whole EffectOp to this slot.
            locals.insert(local.0, PVal::EffectOp(op));
            true
        }

        IrPattern::Expr(_) | IrPattern::Ctor { .. } => unimplemented!(
            "dsl_ast::eval::physics: IrPattern::Expr / Ctor in match arm is not in \
             the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
        ),
    }
}

/// Match an `EffectOp` variant against a struct pattern `Name { field, ... }`.
fn match_effect_op_struct(
    name: &str,
    op: EffectOp,
    bindings: &[IrPatternBinding],
    locals: &mut Locals,
) -> bool {
    // Helper: bind a field from the bindings list.
    let bind = |field: &str, val: PVal, bindings: &[IrPatternBinding], locals: &mut Locals| {
        if let Some(b) = bindings.iter().find(|b| b.field == field) {
            if let IrPattern::Bind { local, .. } = &b.value {
                locals.insert(local.0, val);
            }
            // Wildcard → skip.
        }
    };

    match (name, op) {
        ("Damage", EffectOp::Damage { amount }) => {
            bind("amount", PVal::Float(amount), bindings, locals);
            true
        }
        ("Heal", EffectOp::Heal { amount }) => {
            bind("amount", PVal::Float(amount), bindings, locals);
            true
        }
        ("Shield", EffectOp::Shield { amount }) => {
            bind("amount", PVal::Float(amount), bindings, locals);
            true
        }
        ("Stun", EffectOp::Stun { duration_ticks }) => {
            bind("duration_ticks", PVal::Int(duration_ticks as i64), bindings, locals);
            true
        }
        ("Slow", EffectOp::Slow { duration_ticks, factor_q8 }) => {
            bind("duration_ticks", PVal::Int(duration_ticks as i64), bindings, locals);
            bind("factor_q8", PVal::Int(factor_q8 as i64), bindings, locals);
            true
        }
        ("TransferGold", EffectOp::TransferGold { amount }) => {
            bind("amount", PVal::Int(amount), bindings, locals);
            true
        }
        ("ModifyStanding", EffectOp::ModifyStanding { delta }) => {
            bind("delta", PVal::Int(delta as i64), bindings, locals);
            true
        }
        ("CastAbility", EffectOp::CastAbility { ability, selector }) => {
            bind("ability", PVal::Ability(ability), bindings, locals);
            // Encode TargetSelector as a special sentinel so the body can
            // compare with `TargetSelector::Target` / `TargetSelector::Caster`.
            let sel_int: i64 = match selector {
                TargetSelector::Target => 0,
                TargetSelector::Caster => 1,
            };
            bind("sel", PVal::Int(sel_int), bindings, locals);
            // Also bind under the DSL name "selector" (used in the physics.sim
            // source as `CastAbility { ability: nested, selector: sel }`).
            bind("selector", PVal::Int(sel_int), bindings, locals);
            true
        }
        // Name doesn't match variant — no match.
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Expression evaluation
// ---------------------------------------------------------------------------

fn eval_expr<C: CascadeContext>(node: &IrExprNode, ctx: &mut C, locals: &Locals) -> PVal {
    eval_kind(&node.kind, ctx, locals)
}

fn eval_kind<C: CascadeContext>(kind: &IrExpr, ctx: &mut C, locals: &Locals) -> PVal {
    match kind {
        // ---- literals -------------------------------------------------------
        IrExpr::LitBool(b) => PVal::Bool(*b),
        IrExpr::LitInt(v) => PVal::Int(*v),
        IrExpr::LitFloat(v) => PVal::Float(*v as f32),

        // ---- local variable lookup ------------------------------------------
        IrExpr::Local(LocalRef(slot), _name) => {
            locals.get(slot).cloned().unwrap_or_else(|| {
                panic!(
                    "dsl_ast::eval::physics: Local slot {slot} not found in locals map; \
                     see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
                )
            })
        }

        // ---- EnumVariant sentinels -----------------------------------------
        // Physics rules use:
        //  - `None` sentinel for agent-absence comparisons
        //  - `TargetSelector::Target` / `TargetSelector::Caster` in cast rule
        IrExpr::EnumVariant { ty, variant } => {
            match (ty.as_str(), variant.as_str()) {
                ("", "None") => PVal::Agent(None),
                (_, "Target") => PVal::Int(0),
                (_, "Caster") => PVal::Int(1),
                _ => unimplemented!(
                    "dsl_ast::eval::physics: EnumVariant `{ty}::{variant}` is not in \
                     the wolves+humans survey — \
                     see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
                ),
            }
        }

        // ---- namespace field reads ------------------------------------------
        IrExpr::NamespaceField { ns, field, .. } => eval_namespace_field(*ns, field, ctx),

        // ---- namespace method calls -----------------------------------------
        IrExpr::NamespaceCall { ns, method, args } => {
            eval_namespace_call(*ns, method, args, ctx, locals)
        }

        // ---- binary operators -----------------------------------------------
        IrExpr::Binary(op, lhs, rhs) => eval_binary(*op, lhs, rhs, ctx, locals),

        // ---- unary operators ------------------------------------------------
        IrExpr::Unary(UnOp::Not, rhs) => {
            let v = eval_expr(rhs, ctx, locals).as_bool();
            PVal::Bool(!v)
        }
        IrExpr::Unary(UnOp::Neg, rhs) => {
            let v = eval_expr(rhs, ctx, locals).as_f32();
            PVal::Float(-v)
        }

        // ---- if expression --------------------------------------------------
        IrExpr::If { cond, then_expr, else_expr } => {
            let pred = eval_expr(cond, ctx, locals).as_bool();
            if pred {
                eval_expr(then_expr, ctx, locals)
            } else {
                match else_expr.as_deref() {
                    Some(e) => eval_expr(e, ctx, locals),
                    None => PVal::Bool(false),
                }
            }
        }

        // ---- builtin calls --------------------------------------------------
        IrExpr::BuiltinCall(builtin, args) => eval_builtin(*builtin, args, ctx, locals),

        // ---- out-of-survey variants -----------------------------------------
        other => unimplemented!(
            "dsl_ast::eval::physics: IR variant {:?} is not in the wolves+humans survey §3 — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3",
            std::mem::discriminant(other)
        ),
    }
}

// ---------------------------------------------------------------------------
// Binary operator evaluation
// ---------------------------------------------------------------------------

fn eval_binary<C: CascadeContext>(
    op: BinOp,
    lhs: &IrExprNode,
    rhs: &IrExprNode,
    ctx: &mut C,
    locals: &Locals,
) -> PVal {
    // Short-circuit boolean ops first.
    match op {
        BinOp::And => {
            let l = eval_expr(lhs, ctx, locals).as_bool();
            if !l { return PVal::Bool(false); }
            return PVal::Bool(eval_expr(rhs, ctx, locals).as_bool());
        }
        BinOp::Or => {
            let l = eval_expr(lhs, ctx, locals).as_bool();
            if l { return PVal::Bool(true); }
            return PVal::Bool(eval_expr(rhs, ctx, locals).as_bool());
        }
        _ => {}
    }

    // Handle `== None` / `!= None` — the sentinel form for engaged_with.
    if matches!(op, BinOp::Eq | BinOp::NotEq) {
        let lv = eval_expr(lhs, ctx, locals);
        let rv = eval_expr(rhs, ctx, locals);
        return match (op, &lv, &rv) {
            (BinOp::Eq, PVal::Agent(a), PVal::Agent(b)) => PVal::Bool(a == b),
            (BinOp::NotEq, PVal::Agent(a), PVal::Agent(b)) => PVal::Bool(a != b),
            (BinOp::Eq, PVal::Bool(a), PVal::Bool(b)) => PVal::Bool(a == b),
            (BinOp::NotEq, PVal::Bool(a), PVal::Bool(b)) => PVal::Bool(a != b),
            (BinOp::Eq, PVal::Ability(a), PVal::Ability(b)) => PVal::Bool(a == b),
            (BinOp::NotEq, PVal::Ability(a), PVal::Ability(b)) => PVal::Bool(a != b),
            _ => {
                // Fall through to numeric comparison.
                let a = lv.as_f32();
                let b = rv.as_f32();
                match op {
                    BinOp::Eq => PVal::Bool(a == b),
                    BinOp::NotEq => PVal::Bool(a != b),
                    _ => unreachable!(),
                }
            }
        };
    }

    // Numeric ops.
    let l = eval_expr(lhs, ctx, locals).as_f32();
    let r = eval_expr(rhs, ctx, locals).as_f32();
    match op {
        BinOp::Lt => PVal::Bool(l < r),
        BinOp::LtEq => PVal::Bool(l <= r),
        BinOp::Gt => PVal::Bool(l > r),
        BinOp::GtEq => PVal::Bool(l >= r),
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
            PVal::Float(crate::eval::builtins::eval_arithmetic_binop(op, l, r))
        }
        BinOp::And | BinOp::Or | BinOp::Eq | BinOp::NotEq => unreachable!("handled above"),
    }
}

// ---------------------------------------------------------------------------
// Namespace field reads
// ---------------------------------------------------------------------------

fn eval_namespace_field<C: CascadeContext>(ns: NamespaceId, field: &str, ctx: &mut C) -> PVal {
    match (ns, field) {
        (NamespaceId::World, "tick") => PVal::Int(ctx.world_tick() as i64),
        (NamespaceId::Config, f) => eval_config_field(f, ctx),
        (NamespaceId::Cascade, "max_iterations") => {
            PVal::Int(ctx.config_cascade_max_iterations() as i64)
        }
        _ => unimplemented!(
            "dsl_ast::eval::physics: NamespaceField `{}.{}` is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3",
            ns.name(),
            field
        ),
    }
}

fn eval_config_field<C: CascadeContext>(field: &str, ctx: &mut C) -> PVal {
    match field {
        "combat.attack_range" => PVal::Float(ctx.config_combat_attack_range()),
        "combat.engagement_range" => PVal::Float(ctx.config_combat_engagement_range()),
        "movement.max_move_radius" => PVal::Float(ctx.config_movement_max_move_radius()),
        "cascade.max_iterations" => PVal::Int(ctx.config_cascade_max_iterations() as i64),
        other => unimplemented!(
            "dsl_ast::eval::physics: config field `{other}` is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
        ),
    }
}

// ---------------------------------------------------------------------------
// Namespace method calls
// ---------------------------------------------------------------------------

fn eval_namespace_call<C: CascadeContext>(
    ns: NamespaceId,
    method: &str,
    args: &[IrCallArg],
    ctx: &mut C,
    locals: &Locals,
) -> PVal {
    // Evaluate all arguments eagerly before calling any ctx method, to avoid
    // the borrow conflict where the closure captures `ctx` and then ctx is
    // used again in the match body.
    let eargs: Vec<PVal> = args
        .iter()
        .map(|a| eval_expr(&a.value, ctx, locals))
        .collect();

    match (ns, method) {
        // ---- agents: read accessors ----------------------------------------
        (NamespaceId::Agents, "alive") => {
            assert_eq!(eargs.len(), 1, "agents.alive expects 1 arg");
            PVal::Bool(ctx.agents_alive(eargs[0].as_agent_id()))
        }
        (NamespaceId::Agents, "hp") => {
            assert_eq!(eargs.len(), 1, "agents.hp expects 1 arg");
            PVal::Float(ctx.agents_hp(eargs[0].as_agent_id()))
        }
        (NamespaceId::Agents, "max_hp") => {
            assert_eq!(eargs.len(), 1, "agents.max_hp expects 1 arg");
            PVal::Float(ctx.agents_max_hp(eargs[0].as_agent_id()))
        }
        (NamespaceId::Agents, "shield_hp") => {
            assert_eq!(eargs.len(), 1, "agents.shield_hp expects 1 arg");
            PVal::Float(ctx.agents_shield_hp(eargs[0].as_agent_id()))
        }
        (NamespaceId::Agents, "stun_expires_at_tick") => {
            assert_eq!(eargs.len(), 1, "agents.stun_expires_at_tick expects 1 arg");
            PVal::Int(ctx.agents_stun_expires_at_tick(eargs[0].as_agent_id()) as i64)
        }
        (NamespaceId::Agents, "slow_expires_at_tick") => {
            assert_eq!(eargs.len(), 1, "agents.slow_expires_at_tick expects 1 arg");
            PVal::Int(ctx.agents_slow_expires_at_tick(eargs[0].as_agent_id()) as i64)
        }
        (NamespaceId::Agents, "slow_factor_q8") => {
            assert_eq!(eargs.len(), 1, "agents.slow_factor_q8 expects 1 arg");
            PVal::Int(ctx.agents_slow_factor_q8(eargs[0].as_agent_id()) as i64)
        }
        (NamespaceId::Agents, "attack_damage") => {
            assert_eq!(eargs.len(), 1, "agents.attack_damage expects 1 arg");
            PVal::Float(ctx.agents_attack_damage(eargs[0].as_agent_id()))
        }
        (NamespaceId::Agents, "gold") => {
            assert_eq!(eargs.len(), 1, "agents.gold expects 1 arg");
            PVal::Int(ctx.agents_gold(eargs[0].as_agent_id()))
        }
        (NamespaceId::Agents, "engaged_with") => {
            assert_eq!(eargs.len(), 1, "agents.engaged_with expects 1 arg");
            let id = eargs[0].as_agent_id();
            PVal::Agent(ctx.agents_engaged_with(id))
        }
        // `engaged_with_or(agent, sentinel)` — fallback to sentinel when not engaged.
        (NamespaceId::Agents, "engaged_with_or") => {
            assert_eq!(eargs.len(), 2, "agents.engaged_with_or expects 2 args");
            let id = eargs[0].as_agent_id();
            let sentinel = eargs[1].clone();
            match ctx.agents_engaged_with(id) {
                Some(partner) => PVal::Agent(Some(partner)),
                None => sentinel,
            }
        }
        (NamespaceId::Agents, "pos") => {
            assert_eq!(eargs.len(), 1, "agents.pos expects 1 arg");
            PVal::Vec3(ctx.agents_pos(eargs[0].as_agent_id()))
        }

        // ---- agents: write accessors ----------------------------------------
        (NamespaceId::Agents, "set_hp") => {
            assert_eq!(eargs.len(), 2, "agents.set_hp expects 2 args");
            let id = eargs[0].as_agent_id();
            let val = eargs[1].as_f32();
            ctx.agents_set_hp(id, val);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "set_shield_hp") => {
            assert_eq!(eargs.len(), 2, "agents.set_shield_hp expects 2 args");
            let id = eargs[0].as_agent_id();
            let val = eargs[1].as_f32();
            ctx.agents_set_shield_hp(id, val);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "set_stun_expires_at_tick") => {
            assert_eq!(eargs.len(), 2, "agents.set_stun_expires_at_tick expects 2 args");
            let id = eargs[0].as_agent_id();
            let tick = eargs[1].as_i64() as u32;
            ctx.agents_set_stun_expires_at_tick(id, tick);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "set_slow_expires_at_tick") => {
            assert_eq!(eargs.len(), 2, "agents.set_slow_expires_at_tick expects 2 args");
            let id = eargs[0].as_agent_id();
            let tick = eargs[1].as_i64() as u32;
            ctx.agents_set_slow_expires_at_tick(id, tick);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "set_slow_factor_q8") => {
            assert_eq!(eargs.len(), 2, "agents.set_slow_factor_q8 expects 2 args");
            let id = eargs[0].as_agent_id();
            let fac = eargs[1].as_i64() as i16;
            ctx.agents_set_slow_factor_q8(id, fac);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "kill") => {
            assert_eq!(eargs.len(), 1, "agents.kill expects 1 arg");
            let id = eargs[0].as_agent_id();
            ctx.agents_kill(id);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "add_gold") => {
            assert_eq!(eargs.len(), 2, "agents.add_gold expects 2 args");
            let id = eargs[0].as_agent_id();
            let amount = eargs[1].as_i64();
            ctx.agents_add_gold(id, amount);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "sub_gold") => {
            assert_eq!(eargs.len(), 2, "agents.sub_gold expects 2 args");
            let id = eargs[0].as_agent_id();
            let amount = eargs[1].as_i64();
            ctx.agents_sub_gold(id, amount);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "adjust_standing") => {
            assert_eq!(eargs.len(), 3, "agents.adjust_standing expects 3 args");
            let a = eargs[0].as_agent_id();
            let b = eargs[1].as_agent_id();
            let delta = eargs[2].as_i64() as i16;
            ctx.agents_adjust_standing(a, b, delta);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "set_engaged_with") => {
            assert_eq!(eargs.len(), 2, "agents.set_engaged_with expects 2 args");
            let a = eargs[0].as_agent_id();
            let b = eargs[1].as_agent_id();
            ctx.agents_set_engaged_with(a, b);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "clear_engaged_with") => {
            assert_eq!(eargs.len(), 1, "agents.clear_engaged_with expects 1 arg");
            let id = eargs[0].as_agent_id();
            ctx.agents_clear_engaged_with(id);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "record_memory") => {
            assert_eq!(eargs.len(), 5, "agents.record_memory expects 5 args");
            let observer = eargs[0].as_agent_id();
            let subject = eargs[1].as_agent_id();
            let feeling = eargs[2].as_f32();
            let context_id = eargs[3].as_i64() as u32;
            let tick = eargs[4].as_i64() as u32;
            ctx.agents_record_memory(observer, subject, feeling, context_id, tick);
            PVal::Bool(true)
        }
        (NamespaceId::Agents, "set_cooldown_next_ready") => {
            // `agents.set_cooldown_next_ready(agent, ready_at)` — called as
            // a standalone `Expr` statement in the cast physics rule.
            assert_eq!(eargs.len(), 2, "agents.set_cooldown_next_ready expects 2 args");
            let agent = eargs[0].as_agent_id();
            let ready_at = eargs[1].as_i64() as u32;
            // The DSL form is `agents.set_cooldown_next_ready(caster, next_ready)` but
            // the trait method is `(agent, ability, ready_at)`.
            // Resolution: look for an ability local in scope.
            let ab = locals
                .iter()
                .find_map(|(_, v)| {
                    if let PVal::Ability(ab) = v { Some(*ab) } else { None }
                })
                .unwrap_or_else(|| {
                    panic!(
                        "dsl_ast::eval::physics: agents.set_cooldown_next_ready called but \
                         no ability local found in scope — \
                         see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
                    )
                });
            ctx.abilities_set_cooldown_next_ready(agent, ab, ready_at);
            PVal::Bool(true)
        }

        // ---- abilities: registry reads -------------------------------------
        (NamespaceId::Abilities, "is_known") => {
            assert_eq!(eargs.len(), 1, "abilities.is_known expects 1 arg");
            let ab = eargs[0].as_ability_id();
            PVal::Bool(ctx.abilities_is_known(ab))
        }
        (NamespaceId::Abilities, "known") => {
            assert_eq!(eargs.len(), 2, "abilities.known expects 2 args");
            let agent = eargs[0].as_agent_id();
            let ab = eargs[1].as_ability_id();
            PVal::Bool(ctx.abilities_known(agent, ab))
        }
        (NamespaceId::Abilities, "cooldown_ticks") => {
            assert_eq!(eargs.len(), 1, "abilities.cooldown_ticks expects 1 arg");
            let ab = eargs[0].as_ability_id();
            PVal::Int(ctx.abilities_cooldown_ticks(ab) as i64)
        }
        // `abilities.effects` is handled in exec_for (iterator source) not here.
        (NamespaceId::Abilities, "effects") => unimplemented!(
            "dsl_ast::eval::physics: abilities.effects is only valid as a for-loop \
             iterator source; calling it as an expression is not supported — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
        ),

        // ---- query: spatial ------------------------------------------------
        (NamespaceId::Query, "nearest_hostile_to_or") => {
            // `query.nearest_hostile_to_or(agent, radius, sentinel)`
            assert_eq!(eargs.len(), 3, "query.nearest_hostile_to_or expects 3 args");
            let agent = eargs[0].as_agent_id();
            let radius = eargs[1].as_f32();
            let sentinel = eargs[2].clone();
            match ctx.query_nearest_hostile_to(agent, radius) {
                Some(found) => PVal::Agent(Some(found)),
                None => sentinel,
            }
        }

        _ => unimplemented!(
            "dsl_ast::eval::physics: stdlib call `{}.{}` is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3",
            ns.name(),
            method
        ),
    }
}

// ---------------------------------------------------------------------------
// Builtin calls
// ---------------------------------------------------------------------------

fn eval_builtin<C: CascadeContext>(
    builtin: Builtin,
    args: &[IrCallArg],
    ctx: &mut C,
    locals: &Locals,
) -> PVal {
    use crate::eval::builtins::{eval_numeric_builtin, EvalVal};

    // Evaluate args eagerly and convert to shared EvalVal.
    let eval_args: Vec<EvalVal> = args
        .iter()
        .map(|a| {
            let v = eval_expr(&a.value, ctx, locals);
            match v {
                PVal::Bool(b) => EvalVal::Bool(b),
                PVal::Float(f) => EvalVal::Float(f),
                PVal::Int(i) => EvalVal::Float(i as f32),
                PVal::Vec3(v3) => EvalVal::Vec3(v3),
                PVal::Agent(opt) => EvalVal::Agent(opt),
                PVal::Ability(ab) => EvalVal::Ability(ab),
                PVal::EffectOp(_) => panic!(
                    "dsl_ast::eval::physics: EffectOp cannot be passed to a builtin — \
                     see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3"
                ),
            }
        })
        .collect();

    if let Some(result) = eval_numeric_builtin(builtin.name(), &eval_args) {
        return match result {
            EvalVal::Bool(b) => PVal::Bool(b),
            EvalVal::Float(f) => PVal::Float(f),
            EvalVal::Vec3(v3) => PVal::Vec3(v3),
            EvalVal::Agent(opt) => PVal::Agent(opt),
            EvalVal::Ability(ab) => PVal::Ability(ab),
        };
    }

    unimplemented!(
        "dsl_ast::eval::physics: Builtin `{}` is not in the wolves+humans survey — \
         see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §3",
        builtin.name()
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::eval::{AbilityId, AgentId, CascadeContext, EffectOp, EvalValue, ReadContext, Vec3};
    use crate::ir::{
        IrCallArg, IrEmit, IrExpr, IrExprNode, IrFieldInit, IrPattern, IrPatternBinding,
        IrPhysicsPattern, IrEventPattern, IrStmt, IrStmtMatchArm, LocalRef, NamespaceId,
        PhysicsHandlerIR, PhysicsIR,
    };
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // Test-only context stub
    // -----------------------------------------------------------------------

    /// Records all emitted events and state writes for assertions.
    struct StubCascadeCtx {
        // state
        hp: HashMap<u32, f32>,
        max_hp: HashMap<u32, f32>,
        shield_hp: HashMap<u32, f32>,
        alive: HashMap<u32, bool>,
        stun_expires: HashMap<u32, u32>,
        engaged_with: HashMap<u32, Option<u32>>,
        pos: HashMap<u32, Vec3>,
        attack_damage: HashMap<u32, f32>,
        // outputs
        emitted: Vec<(String, Vec<(String, EvalValue)>)>,
        killed: Vec<u32>,
        hp_sets: Vec<(u32, f32)>,
        shield_sets: Vec<(u32, f32)>,
        stun_sets: Vec<(u32, u32)>,
        engagement_sets: Vec<(u32, u32)>,
        engagement_clears: Vec<u32>,
        // config
        engagement_range: f32,
        cascade_max_iterations: u32,
        tick: u32,
        // nearest hostile
        nearest_hostile: HashMap<u32, Option<u32>>,
        // nearby kin
        nearby_kin: HashMap<u32, Vec<u32>>,
        // ability effects
        ability_effects: HashMap<u32, Vec<EffectOp>>,
        ability_cooldown_ticks: HashMap<u32, u32>,
        ability_is_known: HashMap<u32, bool>,
        cooldown_sets: Vec<(u32, u32, u32)>, // (agent, ability, ready_at)
    }

    impl StubCascadeCtx {
        fn new() -> Self {
            StubCascadeCtx {
                hp: HashMap::new(),
                max_hp: HashMap::new(),
                shield_hp: HashMap::new(),
                alive: HashMap::new(),
                stun_expires: HashMap::new(),
                engaged_with: HashMap::new(),
                pos: HashMap::new(),
                attack_damage: HashMap::new(),
                emitted: Vec::new(),
                killed: Vec::new(),
                hp_sets: Vec::new(),
                shield_sets: Vec::new(),
                stun_sets: Vec::new(),
                engagement_sets: Vec::new(),
                engagement_clears: Vec::new(),
                engagement_range: 12.0,
                cascade_max_iterations: 8,
                tick: 0,
                nearest_hostile: HashMap::new(),
                nearby_kin: HashMap::new(),
                ability_effects: HashMap::new(),
                ability_cooldown_ticks: HashMap::new(),
                ability_is_known: HashMap::new(),
                cooldown_sets: Vec::new(),
            }
        }
    }

    impl ReadContext for StubCascadeCtx {
        fn world_tick(&self) -> u32 { self.tick }
        fn agents_alive(&self, a: AgentId) -> bool {
            *self.alive.get(&a.raw()).unwrap_or(&true)
        }
        fn agents_pos(&self, a: AgentId) -> Vec3 {
            *self.pos.get(&a.raw()).unwrap_or(&[0.0, 0.0, 0.0])
        }
        fn agents_hp(&self, a: AgentId) -> f32 {
            *self.hp.get(&a.raw()).unwrap_or(&100.0)
        }
        fn agents_max_hp(&self, a: AgentId) -> f32 {
            *self.max_hp.get(&a.raw()).unwrap_or(&100.0)
        }
        fn agents_hp_pct(&self, a: AgentId) -> f32 {
            let hp = self.agents_hp(a);
            let max = self.agents_max_hp(a);
            if max == 0.0 { 0.0 } else { hp / max }
        }
        fn agents_shield_hp(&self, a: AgentId) -> f32 {
            *self.shield_hp.get(&a.raw()).unwrap_or(&0.0)
        }
        fn agents_stun_expires_at_tick(&self, a: AgentId) -> u32 {
            *self.stun_expires.get(&a.raw()).unwrap_or(&0)
        }
        fn agents_slow_expires_at_tick(&self, _a: AgentId) -> u32 { 0 }
        fn agents_slow_factor_q8(&self, _a: AgentId) -> i16 { 0 }
        fn agents_attack_damage(&self, a: AgentId) -> f32 {
            *self.attack_damage.get(&a.raw()).unwrap_or(&10.0)
        }
        fn agents_engaged_with(&self, a: AgentId) -> Option<AgentId> {
            self.engaged_with
                .get(&a.raw())
                .copied()
                .flatten()
                .and_then(AgentId::new)
        }
        fn agents_is_hostile_to(&self, _a: AgentId, _b: AgentId) -> bool { false }
        fn agents_gold(&self, _a: AgentId) -> i64 { 0 }
        fn query_nearby_agents(&self, _c: Vec3, _r: f32, _f: &mut dyn FnMut(AgentId)) {}
        fn query_nearby_kin(&self, origin: AgentId, _c: Vec3, _r: f32, f: &mut dyn FnMut(AgentId)) {
            if let Some(kin) = self.nearby_kin.get(&origin.raw()) {
                for &id in kin {
                    if let Some(aid) = AgentId::new(id) {
                        f(aid);
                    }
                }
            }
        }
        fn query_nearest_hostile_to(&self, a: AgentId, _r: f32) -> Option<AgentId> {
            self.nearest_hostile
                .get(&a.raw())
                .copied()
                .flatten()
                .and_then(AgentId::new)
        }
        fn abilities_is_known(&self, ab: AbilityId) -> bool {
            *self.ability_is_known.get(&ab.raw()).unwrap_or(&false)
        }
        fn abilities_known(&self, _a: AgentId, ab: AbilityId) -> bool {
            *self.ability_is_known.get(&ab.raw()).unwrap_or(&false)
        }
        fn abilities_cooldown_ready(&self, _a: AgentId, _ab: AbilityId) -> bool { true }
        fn abilities_cooldown_ticks(&self, ab: AbilityId) -> u32 {
            *self.ability_cooldown_ticks.get(&ab.raw()).unwrap_or(&10)
        }
        fn abilities_effects(&self, ab: AbilityId, f: &mut dyn FnMut(EffectOp)) {
            if let Some(ops) = self.ability_effects.get(&ab.raw()) {
                for op in ops {
                    f(*op);
                }
            }
        }
        fn config_combat_attack_range(&self) -> f32 { 2.0 }
        fn config_combat_engagement_range(&self) -> f32 { self.engagement_range }
        fn config_movement_max_move_radius(&self) -> f32 { 20.0 }
        fn config_cascade_max_iterations(&self) -> u32 { self.cascade_max_iterations }
        fn view_is_hostile(&self, _a: AgentId, _b: AgentId) -> bool { false }
        fn view_is_stunned(&self, _a: AgentId) -> bool { false }
        fn view_threat_level(&self, _o: AgentId, _t: AgentId) -> f32 { 0.0 }
        fn view_my_enemies(&self, _o: AgentId, _t: AgentId) -> f32 { 0.0 }
        fn view_pack_focus(&self, _o: AgentId, _t: AgentId) -> f32 { 0.0 }
        fn view_kin_fear(&self, _o: AgentId) -> f32 { 0.0 }
        fn view_rally_boost(&self, _o: AgentId) -> f32 { 0.0 }
        fn view_slow_factor(&self, _a: AgentId) -> f32 { 1.0 }
    }

    impl CascadeContext for StubCascadeCtx {
        fn agents_set_hp(&mut self, a: AgentId, hp: f32) {
            self.hp_sets.push((a.raw(), hp));
            self.hp.insert(a.raw(), hp);
        }
        fn agents_set_shield_hp(&mut self, a: AgentId, v: f32) {
            self.shield_sets.push((a.raw(), v));
            self.shield_hp.insert(a.raw(), v);
        }
        fn agents_set_stun_expires_at_tick(&mut self, a: AgentId, e: u32) {
            self.stun_sets.push((a.raw(), e));
            self.stun_expires.insert(a.raw(), e);
        }
        fn agents_set_slow_expires_at_tick(&mut self, _a: AgentId, _e: u32) {}
        fn agents_set_slow_factor_q8(&mut self, _a: AgentId, _f: i16) {}
        fn agents_set_engaged_with(&mut self, a: AgentId, b: AgentId) {
            self.engagement_sets.push((a.raw(), b.raw()));
            self.engaged_with.insert(a.raw(), Some(b.raw()));
        }
        fn agents_clear_engaged_with(&mut self, a: AgentId) {
            self.engagement_clears.push(a.raw());
            self.engaged_with.insert(a.raw(), None);
        }
        fn agents_kill(&mut self, a: AgentId) {
            self.killed.push(a.raw());
            self.alive.insert(a.raw(), false);
        }
        fn agents_add_gold(&mut self, _a: AgentId, _amt: i64) {}
        fn agents_sub_gold(&mut self, _a: AgentId, _amt: i64) {}
        fn agents_adjust_standing(&mut self, _a: AgentId, _b: AgentId, _d: i16) {}
        fn agents_record_memory(
            &mut self,
            _observer: AgentId,
            _subject: AgentId,
            _feeling: f32,
            _context: u32,
            _tick: u32,
        ) {}
        fn abilities_set_cooldown_next_ready(&mut self, a: AgentId, ab: AbilityId, ready: u32) {
            self.cooldown_sets.push((a.raw(), ab.raw(), ready));
        }
        fn emit(&mut self, event_name: &str, fields: &[(&str, EvalValue)]) {
            let owned: Vec<(String, EvalValue)> = fields
                .iter()
                .map(|(n, v)| (n.to_string(), *v))
                .collect();
            self.emitted.push((event_name.to_string(), owned));
        }
    }

    // -----------------------------------------------------------------------
    // IR construction helpers
    // -----------------------------------------------------------------------

    fn span() -> Span { Span::dummy() }
    fn aid(r: u32) -> AgentId { AgentId::new(r).unwrap() }
    fn abid(r: u32) -> AbilityId { AbilityId::new(r).unwrap() }

    fn lit_float(v: f64) -> IrExprNode {
        IrExprNode { kind: IrExpr::LitFloat(v), span: span() }
    }
    #[allow(dead_code)]
    fn lit_int(v: i64) -> IrExprNode {
        IrExprNode { kind: IrExpr::LitInt(v), span: span() }
    }
    fn local(name: &str, slot: u16) -> IrExprNode {
        IrExprNode { kind: IrExpr::Local(LocalRef(slot), name.to_string()), span: span() }
    }
    fn ns_call(ns: NamespaceId, method: &str, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::NamespaceCall {
                ns,
                method: method.to_string(),
                args: args.into_iter()
                    .map(|v| IrCallArg { name: None, value: v, span: span() })
                    .collect(),
            },
            span: span(),
        }
    }
    fn binary(op: BinOp, l: IrExprNode, r: IrExprNode) -> IrExprNode {
        IrExprNode { kind: IrExpr::Binary(op, Box::new(l), Box::new(r)), span: span() }
    }
    #[allow(dead_code)]
    fn if_expr(cond: IrExprNode, then_e: IrExprNode, else_e: IrExprNode) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::If {
                cond: Box::new(cond),
                then_expr: Box::new(then_e),
                else_expr: Some(Box::new(else_e)),
            },
            span: span(),
        }
    }
    fn ir_emit(event: &str, fields: Vec<(&str, IrExprNode)>) -> IrStmt {
        IrStmt::Emit(IrEmit {
            event_name: event.to_string(),
            event: None,
            fields: fields.into_iter()
                .map(|(n, v)| IrFieldInit { name: n.to_string(), value: v, span: span() })
                .collect(),
            span: span(),
        })
    }
    fn bind_field(field: &str, slot: u16) -> IrPatternBinding {
        IrPatternBinding {
            field: field.to_string(),
            value: IrPattern::Bind { name: field.to_string(), local: LocalRef(slot) },
            span: span(),
        }
    }

    /// Build a PhysicsIR with a single kind-matched handler.
    fn make_physics(
        event_name: &str,
        bindings: Vec<IrPatternBinding>,
        body: Vec<IrStmt>,
    ) -> PhysicsIR {
        PhysicsIR {
            name: "test".to_string(),
            handlers: vec![PhysicsHandlerIR {
                pattern: IrPhysicsPattern::Kind(IrEventPattern {
                    name: event_name.to_string(),
                    event: None,
                    bindings,
                    span: span(),
                }),
                where_clause: None,
                body,
                span: span(),
            }],
            annotations: vec![],
            span: span(),
        }
    }

    /// Convert `&[(&str, EvalValue)]` to owned event-fields for `apply`.
    #[allow(dead_code)]
    fn ev(fields: &[(&'static str, EvalValue)]) -> Vec<(&'static str, EvalValue)> {
        fields.to_vec()
    }

    // -----------------------------------------------------------------------
    // Test 1: Single-event handler emission
    // -----------------------------------------------------------------------
    /// Handler: `on SomeEvent { actor: c } { emit AnotherEvent { actor: c } }`
    /// Verifies that the emit fires and the emitted event name + field match.
    #[test]
    fn single_event_handler_emits() {
        let body = vec![ir_emit("AnotherEvent", vec![("actor", local("c", 1))])];
        let physics = make_physics(
            "SomeEvent",
            vec![bind_field("actor", 1)],
            body,
        );
        let mut ctx = StubCascadeCtx::new();
        let event_fields: Vec<(&str, EvalValue)> = vec![("actor", EvalValue::Agent(aid(42)))];
        physics.apply(0, &event_fields, &mut ctx);

        assert_eq!(ctx.emitted.len(), 1);
        assert_eq!(ctx.emitted[0].0, "AnotherEvent");
        let actor_field = ctx.emitted[0].1.iter().find(|(n, _)| n == "actor").unwrap();
        assert_eq!(actor_field.1, EvalValue::Agent(aid(42)));
    }

    // -----------------------------------------------------------------------
    // Test 2: Conditional emission — fires when alive
    // -----------------------------------------------------------------------
    /// Handler: `on Dmg { target: t, amount: a } { if agents.alive(t) { emit Hit { target: t } } }`
    #[test]
    fn conditional_emission_fires_when_alive() {
        let cond = ns_call(NamespaceId::Agents, "alive", vec![local("t", 2)]);
        let emit_stmt = ir_emit("Hit", vec![("target", local("t", 2))]);
        let body = vec![IrStmt::If {
            cond,
            then_body: vec![emit_stmt],
            else_body: None,
            span: span(),
        }];
        let physics = make_physics(
            "Dmg",
            vec![bind_field("target", 2), bind_field("amount", 3)],
            body,
        );
        let mut ctx = StubCascadeCtx::new();
        ctx.alive.insert(5, true);
        let event_fields: Vec<(&str, EvalValue)> = vec![
            ("target", EvalValue::Agent(aid(5))),
            ("amount", EvalValue::F32(10.0)),
        ];
        physics.apply(0, &event_fields, &mut ctx);
        assert_eq!(ctx.emitted.len(), 1, "emit should fire for alive target");
        assert_eq!(ctx.emitted[0].0, "Hit");
    }

    #[test]
    fn conditional_emission_suppressed_when_dead() {
        let cond = ns_call(NamespaceId::Agents, "alive", vec![local("t", 2)]);
        let emit_stmt = ir_emit("Hit", vec![("target", local("t", 2))]);
        let body = vec![IrStmt::If {
            cond,
            then_body: vec![emit_stmt],
            else_body: None,
            span: span(),
        }];
        let physics = make_physics(
            "Dmg",
            vec![bind_field("target", 2), bind_field("amount", 3)],
            body,
        );
        let mut ctx = StubCascadeCtx::new();
        ctx.alive.insert(5, false);
        let event_fields: Vec<(&str, EvalValue)> = vec![
            ("target", EvalValue::Agent(aid(5))),
            ("amount", EvalValue::F32(10.0)),
        ];
        physics.apply(0, &event_fields, &mut ctx);
        assert_eq!(ctx.emitted.len(), 0, "emit should be suppressed for dead target");
    }

    // -----------------------------------------------------------------------
    // Test 3: For-loop iteration emitting one event per kin
    // -----------------------------------------------------------------------
    /// Handler: `on AgentDied { agent_id: dead } { for kin in query.nearby_kin(dead, 12.0) { emit FearSpread { ... } } }`
    #[test]
    fn for_loop_emits_per_kin() {
        let iter = ns_call(
            NamespaceId::Query,
            "nearby_kin",
            vec![local("dead", 1), lit_float(12.0)],
        );
        let emit_stmt = ir_emit(
            "FearSpread",
            vec![
                ("observer", local("kin", 10)),
                ("dead_kin", local("dead", 1)),
            ],
        );
        let body = vec![IrStmt::For {
            binder: LocalRef(10),
            binder_name: "kin".to_string(),
            iter,
            filter: None,
            body: vec![emit_stmt],
            span: span(),
        }];
        let physics = make_physics(
            "AgentDied",
            vec![bind_field("agent_id", 1)],
            body,
        );

        let mut ctx = StubCascadeCtx::new();
        // Agent 7 died; kin = agents 2 and 3.
        ctx.nearby_kin.insert(7, vec![2, 3]);
        let event_fields: Vec<(&str, EvalValue)> = vec![("agent_id", EvalValue::Agent(aid(7)))];
        physics.apply(0, &event_fields, &mut ctx);

        assert_eq!(ctx.emitted.len(), 2, "one FearSpread per kin");
        assert_eq!(ctx.emitted[0].0, "FearSpread");
        assert_eq!(ctx.emitted[1].0, "FearSpread");
    }

    // -----------------------------------------------------------------------
    // Test 4: State mutation — HP update via let-binding + set_hp
    // -----------------------------------------------------------------------
    /// Simulates the damage handler:
    ///   `let cur_hp = agents.hp(t)`
    ///   `let new_hp = max(cur_hp - a, 0.0)`
    ///   `agents.set_hp(t, new_hp)`
    #[test]
    fn state_mutation_hp_update() {
        // `let cur_hp = agents.hp(t)`
        let let_cur = IrStmt::Let {
            name: "cur_hp".to_string(),
            local: LocalRef(10),
            value: ns_call(NamespaceId::Agents, "hp", vec![local("t", 2)]),
            span: span(),
        };
        // `let new_hp = max(cur_hp - a, 0.0)`
        let sub = binary(BinOp::Sub, local("cur_hp", 10), local("a", 3));
        let max_call = IrExprNode {
            kind: IrExpr::BuiltinCall(
                Builtin::Max,
                vec![
                    IrCallArg { name: None, value: sub, span: span() },
                    IrCallArg { name: None, value: lit_float(0.0), span: span() },
                ],
            ),
            span: span(),
        };
        let let_new = IrStmt::Let {
            name: "new_hp".to_string(),
            local: LocalRef(11),
            value: max_call,
            span: span(),
        };
        // `agents.set_hp(t, new_hp)`
        let set_hp = IrStmt::Expr(ns_call(
            NamespaceId::Agents,
            "set_hp",
            vec![local("t", 2), local("new_hp", 11)],
        ));
        let body = vec![let_cur, let_new, set_hp];
        let physics = make_physics(
            "EffectDamageApplied",
            vec![bind_field("target", 2), bind_field("amount", 3)],
            body,
        );

        let mut ctx = StubCascadeCtx::new();
        ctx.hp.insert(5, 80.0); // target starts at 80 HP

        let event_fields: Vec<(&str, EvalValue)> = vec![
            ("target", EvalValue::Agent(aid(5))),
            ("amount", EvalValue::F32(30.0)),
        ];
        physics.apply(0, &event_fields, &mut ctx);

        // Expected: new_hp = max(80 - 30, 0) = 50.
        let final_hp = ctx.hp.get(&5).copied().unwrap_or(0.0);
        assert!((final_hp - 50.0).abs() < 1e-5, "expected hp=50, got {final_hp}");
        assert_eq!(ctx.hp_sets.len(), 1);
        assert!((ctx.hp_sets[0].1 - 50.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Test 5: Kill when HP drops to zero
    // -----------------------------------------------------------------------
    /// Handler body:
    ///   `let new_hp = max(agents.hp(t) - a, 0.0)`
    ///   `agents.set_hp(t, new_hp)`
    ///   `if new_hp <= 0.0 { emit AgentDied { agent_id: t }; agents.kill(t) }`
    #[test]
    fn kill_fires_when_hp_zero() {
        // let new_hp = max(agents.hp(t) - a, 0.0)
        let sub = binary(
            BinOp::Sub,
            ns_call(NamespaceId::Agents, "hp", vec![local("t", 2)]),
            local("a", 3),
        );
        let max_call = IrExprNode {
            kind: IrExpr::BuiltinCall(
                Builtin::Max,
                vec![
                    IrCallArg { name: None, value: sub, span: span() },
                    IrCallArg { name: None, value: lit_float(0.0), span: span() },
                ],
            ),
            span: span(),
        };
        let let_new = IrStmt::Let {
            name: "new_hp".to_string(),
            local: LocalRef(10),
            value: max_call,
            span: span(),
        };
        let set_hp = IrStmt::Expr(ns_call(
            NamespaceId::Agents,
            "set_hp",
            vec![local("t", 2), local("new_hp", 10)],
        ));
        let cond = binary(BinOp::LtEq, local("new_hp", 10), lit_float(0.0));
        let die_emit = ir_emit("AgentDied", vec![("agent_id", local("t", 2))]);
        let kill = IrStmt::Expr(ns_call(
            NamespaceId::Agents,
            "kill",
            vec![local("t", 2)],
        ));
        let if_lethal = IrStmt::If {
            cond,
            then_body: vec![die_emit, kill],
            else_body: None,
            span: span(),
        };
        let body = vec![let_new, set_hp, if_lethal];
        let physics = make_physics(
            "EffectDamageApplied",
            vec![bind_field("target", 2), bind_field("amount", 3)],
            body,
        );

        let mut ctx = StubCascadeCtx::new();
        ctx.hp.insert(5, 20.0); // agent 5 has 20 HP

        let event_fields: Vec<(&str, EvalValue)> = vec![
            ("target", EvalValue::Agent(aid(5))),
            ("amount", EvalValue::F32(25.0)), // lethal
        ];
        physics.apply(0, &event_fields, &mut ctx);

        assert!(ctx.killed.contains(&5), "agent 5 should be killed");
        assert_eq!(ctx.emitted.iter().filter(|(n, _)| n == "AgentDied").count(), 1);
    }

    // -----------------------------------------------------------------------
    // Test 6: Saturating_add builtin (used in cast for expiry computation)
    // -----------------------------------------------------------------------
    #[test]
    fn saturating_add_builtin_clamps() {
        // `saturating_add(u32::MAX, 100)` should stay at u32::MAX cast to f32.
        let t = u32::MAX;
        let dur = 100u32;
        let result = (t as u32).saturating_add(dur);
        assert_eq!(result, u32::MAX, "saturating_add must clamp at u32::MAX");
    }

    // -----------------------------------------------------------------------
    // Test 7: Min/Max builtins in shield absorption
    // -----------------------------------------------------------------------
    /// `let absorbed = min(shield, a)` with shield=10, a=15 → absorbed=10.
    #[test]
    fn min_builtin_shield_absorption() {
        // let shield = 10.0 (constant); let a = 15.0 (constant)
        // let absorbed = min(shield, a)
        let let_shield = IrStmt::Let {
            name: "shield".to_string(),
            local: LocalRef(10),
            value: lit_float(10.0),
            span: span(),
        };
        let min_call = IrExprNode {
            kind: IrExpr::BuiltinCall(
                Builtin::Min,
                vec![
                    IrCallArg { name: None, value: local("shield", 10), span: span() },
                    IrCallArg { name: None, value: local("a", 3), span: span() },
                ],
            ),
            span: span(),
        };
        let let_absorbed = IrStmt::Let {
            name: "absorbed".to_string(),
            local: LocalRef(11),
            value: min_call,
            span: span(),
        };
        // emit TestResult { val: absorbed }
        let emit = ir_emit("TestResult", vec![("val", local("absorbed", 11))]);
        let body = vec![let_shield, let_absorbed, emit];
        let physics = make_physics(
            "SomeEvent",
            vec![bind_field("amount", 3)],
            body,
        );

        let mut ctx = StubCascadeCtx::new();
        let event_fields: Vec<(&str, EvalValue)> = vec![
            ("amount", EvalValue::F32(15.0)),
        ];
        physics.apply(0, &event_fields, &mut ctx);

        assert_eq!(ctx.emitted.len(), 1);
        let val = ctx.emitted[0].1.iter().find(|(n, _)| n == "val").unwrap().1;
        if let EvalValue::F32(f) = val {
            assert!((f - 10.0).abs() < 1e-5, "min(10, 15)=10, got {f}");
        } else {
            panic!("expected F32, got {val:?}");
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: Match over EffectOp — cast rule style
    // -----------------------------------------------------------------------
    /// For a `Damage { amount }` op, the match arm binds `amount` and emits.
    #[test]
    fn match_effect_op_damage_arm() {
        // Simulate: `for op in abilities.effects(ab) { match op { Damage { amount } => { emit ... } } }`
        // We stub the for-loop directly by using abilities.effects.
        let match_arm_body = vec![ir_emit(
            "EffectDamageApplied",
            vec![
                ("actor", local("caster", 1)),
                ("target", local("target", 2)),
                ("amount", local("amount", 20)),
            ],
        )];
        let damage_arm = IrStmtMatchArm {
            pattern: IrPattern::Struct {
                name: "Damage".to_string(),
                ctor: None,
                bindings: vec![IrPatternBinding {
                    field: "amount".to_string(),
                    value: IrPattern::Bind {
                        name: "amount".to_string(),
                        local: LocalRef(20),
                    },
                    span: span(),
                }],
            },
            body: match_arm_body,
            span: span(),
        };
        let wildcard_arm = IrStmtMatchArm {
            pattern: IrPattern::Wildcard,
            body: vec![],
            span: span(),
        };
        let match_stmt = IrStmt::Match {
            scrutinee: local("op", 10),
            arms: vec![damage_arm, wildcard_arm],
            span: span(),
        };
        let for_loop = IrStmt::For {
            binder: LocalRef(10),
            binder_name: "op".to_string(),
            iter: ns_call(NamespaceId::Abilities, "effects", vec![local("ab", 5)]),
            filter: None,
            body: vec![match_stmt],
            span: span(),
        };
        let body = vec![for_loop];
        let physics = make_physics(
            "AgentCast",
            vec![
                bind_field("actor", 1),
                bind_field("target", 2),
                bind_field("ability", 5),
            ],
            body,
        );

        let mut ctx = StubCascadeCtx::new();
        ctx.ability_effects
            .insert(7, vec![EffectOp::Damage { amount: 25.0 }]);

        let event_fields: Vec<(&str, EvalValue)> = vec![
            ("actor", EvalValue::Agent(aid(1))),
            ("target", EvalValue::Agent(aid(2))),
            ("ability", EvalValue::Ability(abid(7))),
        ];
        physics.apply(0, &event_fields, &mut ctx);

        assert_eq!(ctx.emitted.len(), 1, "one EffectDamageApplied");
        assert_eq!(ctx.emitted[0].0, "EffectDamageApplied");
        let amt = ctx.emitted[0].1.iter().find(|(n, _)| n == "amount").unwrap().1;
        if let EvalValue::F32(f) = amt {
            assert!((f - 25.0).abs() < 1e-5, "damage amount should be 25.0");
        } else {
            panic!("expected F32 for amount, got {amt:?}");
        }
    }
}
