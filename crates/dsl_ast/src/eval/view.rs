//! Interpreter for `ViewIR` — the materialized-view fold rule class.
//!
//! ## Supported primitives (wolves+humans coverage, §4)
//!
//! This file covers **exactly** the IR variants, stdlib functions, and field
//! reads exercised by the wolves+humans fixture (survey §4). Everything
//! outside this set panics with a pointer to the survey doc.
//!
//! ### View body kinds implemented
//!
//! | Variant | Notes |
//! |---------|-------|
//! | `ViewBodyIR::Expr` | lazy views (`is_hostile`, `is_stunned`, `slow_factor`) |
//! | `ViewBodyIR::Fold` | materialized views with event handlers and optional decay |
//!
//! ### IR expression kinds implemented (in fold bodies)
//!
//! | Variant | Notes |
//! |---------|-------|
//! | `LitFloat` | float constants (0.0, 1000.0, 10.0) |
//! | `LitInt` | integer constants (0) |
//! | `Local` | view parameters (`a`, `b`) and fold-pattern bindings |
//! | `NamespaceField` | `world.tick`, `agents.stun_expires_at_tick(a)`, etc. |
//! | `NamespaceCall` | `agents.*` accessor calls |
//! | `Binary` | `<`, `>`, `==` comparisons |
//! | `If` | ternary conditional (e.g. `slow_factor`) |
//!
//! ### IR statement kinds implemented (in fold bodies)
//!
//! | Variant | Notes |
//! |---------|-------|
//! | `SelfUpdate` | `self += 1.0` / `self += 1` fold accumulation |
//!
//! ### Stdlib functions implemented (via `ViewContext: ReadContext`)
//!
//! | Function | View(s) Using | Purpose |
//! |----------|---------|---------|
//! | `agents.is_hostile_to(a, b)` | is_hostile | Hostility matrix lookup |
//! | `agents.stun_expires_at_tick(a)` | is_stunned | Stun expiry for predicate check |
//! | `agents.slow_expires_at_tick(a)` | slow_factor | Slow expiry for active check |
//! | `agents.slow_factor_q8(a)` | slow_factor | Slow factor read (q8 fixed-point) |
//! | `world.tick` | is_stunned, slow_factor | Current tick for expiry comparison |
//!
//! ### Coverage source
//!
//! `docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md` §4.
//!
//! Any IR variant not listed above hits `unimplemented!()` with a message
//! pointing to that survey document.
//!
//! ## Event payload plumbing
//!
//! The `ViewContext` trait does not expose event-field reads as a method.
//! `ViewIR::fold` accepts `event_kind: &str` and a pre-populated
//! `event_fields: &[(&str, EvalValue)]` slice. The engine-side call site
//! (Task 11) populates these from the live event struct and calls `fold` for
//! each (view, event) pair. If `event_kind` does not match the fold handler's
//! declared pattern name, the call is a no-op.
//!
//! View parameters (`a`, `b`) are supplied by the caller as the `observer`
//! slice: for a per-agent view `[a]`, for a pair-keyed view `[a, b]`.
//!
//! ## Public API
//!
//! ```ignore
//! impl ViewIR {
//!     pub fn fold<C: ViewContext>(
//!         &self,
//!         event_kind: &str,
//!         event_fields: &[(&str, EvalValue)],
//!         observer: &[AgentId],
//!         ctx: &mut C,
//!     )
//! }
//! ```

use std::collections::HashMap;

use crate::ast::{BinOp, UnOp};
use crate::eval::{AgentId, EvalValue, ViewContext};
use crate::ir::{
    FoldHandlerIR, IrCallArg, IrExpr, IrExprNode, IrPattern, IrPatternBinding, IrStmt,
    NamespaceId, ViewBodyIR, ViewIR,
};

// ---------------------------------------------------------------------------
// Runtime value
// ---------------------------------------------------------------------------

/// Dynamically-typed intermediate value for view expressions.
///
/// Views are read-only observers that fold into scalar accumulators.
/// The set of value shapes is smaller than physics — no `EffectOp`,
/// no `Vec3` needed for the wolves+humans fixture.
#[derive(Debug, Clone)]
enum VVal {
    Bool(bool),
    Float(f32),
    Int(i64),
    Agent(Option<AgentId>),
}

impl VVal {
    fn as_bool(&self) -> bool {
        match self {
            VVal::Bool(b) => *b,
            VVal::Float(f) => *f != 0.0,
            VVal::Int(i) => *i != 0,
            VVal::Agent(Some(_)) => true,
            VVal::Agent(None) => false,
        }
    }

    fn as_f32(&self) -> f32 {
        match self {
            VVal::Float(f) => *f,
            VVal::Int(i) => *i as f32,
            other => panic!(
                "dsl_ast::eval::view: expected Float, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
            ),
        }
    }

    fn as_i64(&self) -> i64 {
        match self {
            VVal::Int(i) => *i,
            VVal::Float(f) => *f as i64,
            other => panic!(
                "dsl_ast::eval::view: expected Int, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Locals map
// ---------------------------------------------------------------------------

type Locals = HashMap<u16, VVal>;

// ---------------------------------------------------------------------------
// ViewIR::fold
// ---------------------------------------------------------------------------

impl ViewIR {
    /// Fold an incoming event into this view's materialized storage.
    ///
    /// `event_kind` — the DSL event name (e.g. `"AgentAttacked"`).
    /// `event_fields` — flat key-value pairs pre-populated from the live event.
    /// `observer` — the view key slots: `[a]` for per-agent views, `[a, b]` for
    ///              pair-keyed views. Must match the view's declared param count.
    ///
    /// No-op when:
    /// - The view body is `Expr` (lazy view — has no fold handlers).
    /// - No fold handler pattern matches `event_kind`.
    /// - The view body is `Fold` but has zero matching handlers for `event_kind`.
    pub fn fold<C: ViewContext>(
        &self,
        event_kind: &str,
        event_fields: &[(&str, EvalValue)],
        observer: &[AgentId],
        ctx: &mut C,
    ) {
        let fold_body = match &self.body {
            ViewBodyIR::Expr(_) => return, // lazy view — no fold handlers
            ViewBodyIR::Fold { handlers, .. } => handlers,
        };

        // Build locals from the view params (a, b, …) bound to observer slots.
        let mut locals: Locals = HashMap::new();
        for (idx, param) in self.params.iter().enumerate() {
            if let Some(&agent) = observer.get(idx) {
                locals.insert(param.local.0, VVal::Agent(Some(agent)));
            }
        }

        // Find the matching handler(s) for this event_kind.
        let view_name = self.name.as_str();
        for handler in fold_body {
            if handler.pattern.name != event_kind {
                continue;
            }
            apply_fold_handler(handler, event_fields, view_name, observer, &mut locals, ctx);
        }
    }
}

// ---------------------------------------------------------------------------
// Fold handler application
// ---------------------------------------------------------------------------

fn apply_fold_handler<C: ViewContext>(
    handler: &FoldHandlerIR,
    event_fields: &[(&str, EvalValue)],
    view_name: &str,
    observer: &[AgentId],
    locals: &mut Locals,
    ctx: &mut C,
) {
    // Bind event pattern fields into locals.
    bind_pattern_fields(&handler.pattern.bindings, event_fields, locals);

    // Execute the handler body statements.
    exec_stmts(&handler.body, view_name, observer, locals, ctx);
}

/// Bind event pattern field bindings into locals.
fn bind_pattern_fields(
    bindings: &[IrPatternBinding],
    event_fields: &[(&str, EvalValue)],
    locals: &mut Locals,
) {
    for binding in bindings {
        match &binding.value {
            IrPattern::Bind { name: _, local } => {
                let val = event_fields
                    .iter()
                    .find(|(fname, _)| *fname == binding.field)
                    .map(|(_, v)| eval_value_to_vval(*v))
                    .unwrap_or_else(|| {
                        panic!(
                            "dsl_ast::eval::view: event field `{}` not found in event_fields; \
                             available: {:?}",
                            binding.field,
                            event_fields.iter().map(|(n, _)| n).collect::<Vec<_>>()
                        )
                    });
                locals.insert(local.0, val);
            }
            IrPattern::Wildcard => {}
            other => unimplemented!(
                "dsl_ast::eval::view: unsupported pattern binding variant {other:?} — \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
            ),
        }
    }
}

/// Convert a public `EvalValue` to a local `VVal`.
fn eval_value_to_vval(v: EvalValue) -> VVal {
    match v {
        EvalValue::Bool(b) => VVal::Bool(b),
        EvalValue::I32(i) => VVal::Int(i as i64),
        EvalValue::I64(i) => VVal::Int(i),
        EvalValue::U32(u) => VVal::Int(u as i64),
        EvalValue::F32(f) => VVal::Float(f),
        EvalValue::Agent(id) => VVal::Agent(Some(id)),
        EvalValue::Ability(_) => unimplemented!(
            "dsl_ast::eval::view: AbilityId value in view event field — \
             not exercised by wolves+humans; \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
        ),
    }
}

// ---------------------------------------------------------------------------
// Statement execution
// ---------------------------------------------------------------------------

fn exec_stmts<C: ViewContext>(
    stmts: &[IrStmt],
    view_name: &str,
    observer: &[AgentId],
    locals: &mut Locals,
    ctx: &mut C,
) {
    for stmt in stmts {
        exec_stmt(stmt, view_name, observer, locals, ctx);
    }
}

fn exec_stmt<C: ViewContext>(
    stmt: &IrStmt,
    view_name: &str,
    observer: &[AgentId],
    locals: &mut Locals,
    ctx: &mut C,
) {
    match stmt {
        IrStmt::SelfUpdate { op, value, .. } => {
            let delta = eval_expr(value, locals, ctx);
            match op.as_str() {
                "+=" => {
                    // Determine if this is an f32 or i64 fold by the delta type.
                    match delta {
                        VVal::Float(f) => ctx.view_self_add(view_name, observer, f),
                        VVal::Int(i) => ctx.view_self_add_int(view_name, observer, i),
                        VVal::Bool(b) => {
                            ctx.view_self_add(view_name, observer, if b { 1.0 } else { 0.0 })
                        }
                        other => panic!(
                            "dsl_ast::eval::view: SelfUpdate += expected numeric delta, \
                             got {other:?}; \
                             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
                        ),
                    }
                }
                other => unimplemented!(
                    "dsl_ast::eval::view: SelfUpdate op `{other}` not in survey §4; \
                     see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
                ),
            }
        }

        IrStmt::If { cond, then_body, else_body, .. } => {
            let c = eval_expr(cond, locals, ctx).as_bool();
            if c {
                exec_stmts(then_body, view_name, observer, locals, ctx);
            } else if let Some(eb) = else_body {
                exec_stmts(eb, view_name, observer, locals, ctx);
            }
        }

        IrStmt::Let { local, value, .. } => {
            let v = eval_expr(value, locals, ctx);
            locals.insert(local.0, v);
        }

        IrStmt::Expr(expr) => {
            eval_expr(expr, locals, ctx);
        }

        other => unimplemented!(
            "dsl_ast::eval::view: IrStmt variant {other:?} not in survey §4; \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
        ),
    }
}

// ---------------------------------------------------------------------------
// Expression evaluation
// ---------------------------------------------------------------------------

fn eval_expr<C: ViewContext>(node: &IrExprNode, locals: &Locals, ctx: &mut C) -> VVal {
    eval_kind(&node.kind, locals, ctx)
}

fn eval_kind<C: ViewContext>(kind: &IrExpr, locals: &Locals, ctx: &mut C) -> VVal {
    match kind {
        // ---- Literals ----
        IrExpr::LitBool(b) => VVal::Bool(*b),
        IrExpr::LitFloat(f) => VVal::Float(*f as f32),
        IrExpr::LitInt(i) => VVal::Int(*i),

        // ---- Locals ----
        IrExpr::Local(local_ref, name) => locals
            .get(&local_ref.0)
            .cloned()
            .unwrap_or_else(|| panic!("dsl_ast::eval::view: unbound local `{name}`")),

        // ---- Namespace field reads: world.tick, agents.*  ----
        IrExpr::NamespaceField { ns, field, .. } => eval_namespace_field(*ns, field, locals, ctx),

        // ---- Namespace call: agents.<accessor>(arg) ----
        IrExpr::NamespaceCall { ns, method, args } => {
            eval_namespace_call(*ns, method, args, locals, ctx)
        }

        // ---- Binary operators ----
        IrExpr::Binary(op, lhs, rhs) => eval_binary(*op, lhs, rhs, locals, ctx),

        // ---- Unary operator (! only) ----
        IrExpr::Unary(op, operand) => {
            let v = eval_expr(operand, locals, ctx);
            match op {
                UnOp::Not => VVal::Bool(!v.as_bool()),
                other => unimplemented!(
                    "dsl_ast::eval::view: UnOp::{other:?} not in survey §4; \
                     see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
                ),
            }
        }

        // ---- Conditional expression ----
        IrExpr::If { cond, then_expr, else_expr } => {
            if eval_expr(cond, locals, ctx).as_bool() {
                eval_expr(then_expr, locals, ctx)
            } else if let Some(else_e) = else_expr {
                eval_expr(else_e, locals, ctx)
            } else {
                VVal::Bool(false)
            }
        }

        other => unimplemented!(
            "dsl_ast::eval::view: IrExpr variant {:?} not in survey §4; \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4",
            std::mem::discriminant(other)
        ),
    }
}

// ---------------------------------------------------------------------------
// Namespace field reads
// ---------------------------------------------------------------------------

fn eval_namespace_field<C: ViewContext>(
    ns: NamespaceId,
    field: &str,
    _locals: &Locals,
    ctx: &mut C,
) -> VVal {
    match ns {
        NamespaceId::World => match field {
            "tick" => VVal::Int(ctx.world_tick() as i64),
            other => unimplemented!(
                "dsl_ast::eval::view: world.{other} not in survey §4; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
            ),
        },
        other => unimplemented!(
            "dsl_ast::eval::view: NamespaceField ns={other:?} not in survey §4; \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
        ),
    }
}

// ---------------------------------------------------------------------------
// Namespace call evaluation
// ---------------------------------------------------------------------------

fn eval_namespace_call<C: ViewContext>(
    ns: NamespaceId,
    method: &str,
    args: &[IrCallArg],
    locals: &Locals,
    ctx: &mut C,
) -> VVal {
    match ns {
        NamespaceId::Agents => eval_agents_call(method, args, locals, ctx),
        other => unimplemented!(
            "dsl_ast::eval::view: NamespaceCall ns={other:?} not in survey §4; \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
        ),
    }
}

fn eval_agents_call<C: ViewContext>(
    method: &str,
    args: &[IrCallArg],
    locals: &Locals,
    ctx: &mut C,
) -> VVal {
    /// Helper: evaluate an arg, expected to be an Agent.
    fn agent_arg<C: ViewContext>(
        args: &[IrCallArg],
        idx: usize,
        locals: &Locals,
        ctx: &mut C,
    ) -> AgentId {
        match eval_expr(&args[idx].value, locals, ctx) {
            VVal::Agent(Some(id)) => id,
            VVal::Agent(None) => panic!(
                "dsl_ast::eval::view: agents call arg[{idx}] is None agent"
            ),
            other => panic!(
                "dsl_ast::eval::view: agents call arg[{idx}] expected Agent, got {other:?}"
            ),
        }
    }

    match method {
        "is_hostile_to" => {
            let a = agent_arg(args, 0, locals, ctx);
            let b = agent_arg(args, 1, locals, ctx);
            VVal::Bool(ctx.agents_is_hostile_to(a, b))
        }
        "stun_expires_at_tick" => {
            let a = agent_arg(args, 0, locals, ctx);
            VVal::Int(ctx.agents_stun_expires_at_tick(a) as i64)
        }
        "slow_expires_at_tick" => {
            let a = agent_arg(args, 0, locals, ctx);
            VVal::Int(ctx.agents_slow_expires_at_tick(a) as i64)
        }
        "slow_factor_q8" => {
            let a = agent_arg(args, 0, locals, ctx);
            VVal::Int(ctx.agents_slow_factor_q8(a) as i64)
        }
        other => unimplemented!(
            "dsl_ast::eval::view: agents.{other} not in survey §4; \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
        ),
    }
}

// ---------------------------------------------------------------------------
// Binary operator evaluation
// ---------------------------------------------------------------------------

fn eval_binary<C: ViewContext>(
    op: BinOp,
    lhs: &IrExprNode,
    rhs: &IrExprNode,
    locals: &Locals,
    ctx: &mut C,
) -> VVal {
    match op {
        // Short-circuit boolean ops
        BinOp::And => {
            let l = eval_expr(lhs, locals, ctx).as_bool();
            if !l { return VVal::Bool(false); }
            VVal::Bool(eval_expr(rhs, locals, ctx).as_bool())
        }
        BinOp::Or => {
            let l = eval_expr(lhs, locals, ctx).as_bool();
            if l { return VVal::Bool(true); }
            VVal::Bool(eval_expr(rhs, locals, ctx).as_bool())
        }

        // Comparison ops — evaluate both sides, compare numerically or as agents
        BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq | BinOp::Eq | BinOp::NotEq => {
            let l = eval_expr(lhs, locals, ctx);
            let r = eval_expr(rhs, locals, ctx);
            let result = match (&l, &r) {
                (VVal::Float(_), _) | (_, VVal::Float(_)) => {
                    let lf = l.as_f32();
                    let rf = r.as_f32();
                    match op {
                        BinOp::Lt => lf < rf,
                        BinOp::LtEq => lf <= rf,
                        BinOp::Gt => lf > rf,
                        BinOp::GtEq => lf >= rf,
                        BinOp::Eq => lf == rf,
                        BinOp::NotEq => lf != rf,
                        _ => unreachable!(),
                    }
                }
                (VVal::Int(_), VVal::Int(_)) => {
                    let li = l.as_i64();
                    let ri = r.as_i64();
                    match op {
                        BinOp::Lt => li < ri,
                        BinOp::LtEq => li <= ri,
                        BinOp::Gt => li > ri,
                        BinOp::GtEq => li >= ri,
                        BinOp::Eq => li == ri,
                        BinOp::NotEq => li != ri,
                        _ => unreachable!(),
                    }
                }
                (VVal::Bool(lb), VVal::Bool(rb)) => match op {
                    BinOp::Eq => lb == rb,
                    BinOp::NotEq => lb != rb,
                    _ => panic!(
                        "dsl_ast::eval::view: bool comparison with {:?}", op
                    ),
                },
                (VVal::Agent(la), VVal::Agent(ra)) => match op {
                    BinOp::Eq => la == ra,
                    BinOp::NotEq => la != ra,
                    _ => panic!(
                        "dsl_ast::eval::view: agent comparison with {:?}", op
                    ),
                },
                _ => panic!(
                    "dsl_ast::eval::view: mismatched types in comparison {l:?} {op:?} {r:?}"
                ),
            };
            VVal::Bool(result)
        }

        // Arithmetic ops
        BinOp::Add => {
            let l = eval_expr(lhs, locals, ctx);
            let r = eval_expr(rhs, locals, ctx);
            match (&l, &r) {
                (VVal::Float(_), _) | (_, VVal::Float(_)) => {
                    VVal::Float(l.as_f32() + r.as_f32())
                }
                _ => VVal::Int(l.as_i64() + r.as_i64()),
            }
        }
        BinOp::Sub => {
            let l = eval_expr(lhs, locals, ctx);
            let r = eval_expr(rhs, locals, ctx);
            match (&l, &r) {
                (VVal::Float(_), _) | (_, VVal::Float(_)) => {
                    VVal::Float(l.as_f32() - r.as_f32())
                }
                _ => VVal::Int(l.as_i64() - r.as_i64()),
            }
        }

        other => unimplemented!(
            "dsl_ast::eval::view: BinOp::{other:?} not in survey §4; \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §4"
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, Span};
    use crate::eval::{AgentId, EvalValue, ReadContext, Vec3};
    use crate::ir::{
        FoldHandlerIR, IrEventPattern, IrExprNode, IrParam, IrPatternBinding,
        IrStmt, IrType, LocalRef, NamespaceId, ViewBodyIR, ViewIR, ViewKind, StorageHint,
    };

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn dummy_span() -> Span {
        Span { start: 0, end: 0 }
    }

    fn lit_float(f: f64) -> IrExprNode {
        IrExprNode { kind: IrExpr::LitFloat(f), span: dummy_span() }
    }

    fn lit_int(i: i64) -> IrExprNode {
        IrExprNode { kind: IrExpr::LitInt(i), span: dummy_span() }
    }

    fn aid(raw: u32) -> AgentId {
        AgentId::new(raw).unwrap()
    }

    // -----------------------------------------------------------------------
    // Mock ReadContext + ViewContext
    // -----------------------------------------------------------------------

    /// Minimal test context for view fold tests. Records calls to
    /// `view_self_add` and `view_self_add_int`.
    struct MockViewCtx {
        tick: u32,
        stun_expires: std::collections::HashMap<u32, u32>,
        slow_expires: std::collections::HashMap<u32, u32>,
        slow_factor: std::collections::HashMap<u32, i16>,
        /// Recorded `(view_name, key, delta)` for `view_self_add`.
        pub adds_f32: Vec<(String, Vec<AgentId>, f32)>,
        /// Recorded `(view_name, key, delta)` for `view_self_add_int`.
        pub adds_i64: Vec<(String, Vec<AgentId>, i64)>,
    }

    impl MockViewCtx {
        fn new(tick: u32) -> Self {
            Self {
                tick,
                stun_expires: Default::default(),
                slow_expires: Default::default(),
                slow_factor: Default::default(),
                adds_f32: Vec::new(),
                adds_i64: Vec::new(),
            }
        }
    }

    impl ReadContext for MockViewCtx {
        fn world_tick(&self) -> u32 { self.tick }
        fn agents_alive(&self, _: AgentId) -> bool { true }
        fn agents_pos(&self, _: AgentId) -> Vec3 { [0.0, 0.0, 0.0] }
        fn agents_hp(&self, _: AgentId) -> f32 { 100.0 }
        fn agents_max_hp(&self, _: AgentId) -> f32 { 100.0 }
        fn agents_hp_pct(&self, _: AgentId) -> f32 { 1.0 }
        fn agents_shield_hp(&self, _: AgentId) -> f32 { 0.0 }
        fn agents_stun_expires_at_tick(&self, agent: AgentId) -> u32 {
            *self.stun_expires.get(&agent.raw()).unwrap_or(&0)
        }
        fn agents_slow_expires_at_tick(&self, agent: AgentId) -> u32 {
            *self.slow_expires.get(&agent.raw()).unwrap_or(&0)
        }
        fn agents_slow_factor_q8(&self, agent: AgentId) -> i16 {
            *self.slow_factor.get(&agent.raw()).unwrap_or(&0)
        }
        fn agents_attack_damage(&self, _: AgentId) -> f32 { 10.0 }
        fn agents_engaged_with(&self, _: AgentId) -> Option<AgentId> { None }
        fn agents_is_hostile_to(&self, a: AgentId, b: AgentId) -> bool { a.raw() != b.raw() }
        fn agents_gold(&self, _: AgentId) -> i64 { 0 }
        fn query_nearby_agents(&self, _: Vec3, _: f32, _: &mut dyn FnMut(AgentId)) {}
        fn query_nearby_kin(&self, _: AgentId, _: Vec3, _: f32, _: &mut dyn FnMut(AgentId)) {}
        fn query_nearest_hostile_to(&self, _: AgentId, _: f32) -> Option<AgentId> { None }
        fn abilities_is_known(&self, _: crate::eval::AbilityId) -> bool { false }
        fn abilities_known(&self, _: AgentId, _: crate::eval::AbilityId) -> bool { false }
        fn abilities_cooldown_ready(&self, _: AgentId, _: crate::eval::AbilityId) -> bool { true }
        fn abilities_cooldown_ticks(&self, _: crate::eval::AbilityId) -> u32 { 0 }
        fn abilities_effects(&self, _: crate::eval::AbilityId, _: &mut dyn FnMut(crate::eval::EffectOp)) {}
        fn config_combat_attack_range(&self) -> f32 { 2.0 }
        fn config_combat_engagement_range(&self) -> f32 { 12.0 }
        fn config_movement_max_move_radius(&self) -> f32 { 20.0 }
        fn config_cascade_max_iterations(&self) -> u32 { 8 }
        fn view_is_hostile(&self, a: AgentId, b: AgentId) -> bool { a.raw() != b.raw() }
        fn view_is_stunned(&self, agent: AgentId) -> bool {
            self.agents_stun_expires_at_tick(agent) > self.tick
        }
        fn view_threat_level(&self, _: AgentId, _: AgentId) -> f32 { 0.0 }
        fn view_my_enemies(&self, _: AgentId, _: AgentId) -> f32 { 0.0 }
        fn view_pack_focus(&self, _: AgentId, _: AgentId) -> f32 { 0.0 }
        fn view_kin_fear(&self, _: AgentId) -> f32 { 0.0 }
        fn view_rally_boost(&self, _: AgentId) -> f32 { 0.0 }
        fn view_slow_factor(&self, _: AgentId) -> f32 { 1.0 }
    }

    impl ViewContext for MockViewCtx {
        fn view_self_add(&mut self, view_name: &str, key: &[AgentId], delta: f32) {
            self.adds_f32.push((view_name.to_string(), key.to_vec(), delta));
        }
        fn view_self_add_int(&mut self, view_name: &str, key: &[AgentId], delta: i64) {
            self.adds_i64.push((view_name.to_string(), key.to_vec(), delta));
        }
    }

    // -----------------------------------------------------------------------
    // Helper: build a minimal @materialized ViewIR with one fold handler
    // -----------------------------------------------------------------------

    /// Build a two-param materialized view:
    ///   view <name>(a: Agent, b: Agent) -> f32 {
    ///     initial: 0.0,
    ///     on <event_kind> { actor: b, target: a } { self += <delta_expr> }
    ///   }
    fn build_view_f32(
        name: &str,
        event_kind: &str,
        delta_expr: IrExprNode,
    ) -> ViewIR {
        // Param 0 = `a` at local slot 0; Param 1 = `b` at local slot 1.
        let param_a = IrParam {
            name: "a".to_string(),
            local: LocalRef(0),
            ty: IrType::AgentId,
            span: dummy_span(),
        };
        let param_b = IrParam {
            name: "b".to_string(),
            local: LocalRef(1),
            ty: IrType::AgentId,
            span: dummy_span(),
        };

        let actor_bind = IrPatternBinding {
            field: "actor".to_string(),
            value: IrPattern::Bind { name: "b".to_string(), local: LocalRef(1) },
            span: dummy_span(),
        };
        let target_bind = IrPatternBinding {
            field: "target".to_string(),
            value: IrPattern::Bind { name: "a".to_string(), local: LocalRef(0) },
            span: dummy_span(),
        };

        let handler = FoldHandlerIR {
            pattern: IrEventPattern {
                name: event_kind.to_string(),
                event: None,
                bindings: vec![actor_bind, target_bind],
                span: dummy_span(),
            },
            body: vec![IrStmt::SelfUpdate {
                op: "+=".to_string(),
                value: delta_expr,
                span: dummy_span(),
            }],
            span: dummy_span(),
        };

        ViewIR {
            name: name.to_string(),
            params: vec![param_a, param_b],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_float(0.0),
                handlers: vec![handler],
                clamp: None,
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PerEntityTopK { k: 8, keyed_on: 0 }),
            decay: None,
            span: dummy_span(),
        }
    }

    /// Build a one-param materialized view with an i64 (`self += 1`) body.
    fn build_view_i64(name: &str, event_kind: &str) -> ViewIR {
        let param_a = IrParam {
            name: "a".to_string(),
            local: LocalRef(0),
            ty: IrType::AgentId,
            span: dummy_span(),
        };

        let actor_bind = IrPatternBinding {
            field: "actor".to_string(),
            value: IrPattern::Bind { name: "a".to_string(), local: LocalRef(0) },
            span: dummy_span(),
        };

        let handler = FoldHandlerIR {
            pattern: IrEventPattern {
                name: event_kind.to_string(),
                event: None,
                bindings: vec![actor_bind],
                span: dummy_span(),
            },
            body: vec![IrStmt::SelfUpdate {
                op: "+=".to_string(),
                value: lit_int(1),
                span: dummy_span(),
            }],
            span: dummy_span(),
        };

        ViewIR {
            name: name.to_string(),
            params: vec![param_a],
            return_ty: IrType::AgentId,
            body: ViewBodyIR::Fold {
                initial: lit_int(0),
                handlers: vec![handler],
                clamp: None,
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PerEntityTopK { k: 1, keyed_on: 0 }),
            decay: None,
            span: dummy_span(),
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: event-kind filter — no-op on non-matching event
    // -----------------------------------------------------------------------

    #[test]
    fn test_event_kind_filter_noop() {
        let view = build_view_f32("threat_level", "AgentAttacked", lit_float(1.0));
        let mut ctx = MockViewCtx::new(10);
        let a = aid(1);
        let b = aid(2);

        // Wrong event kind — should not call view_self_add.
        view.fold(
            "SomeOtherEvent",
            &[("actor", EvalValue::Agent(b)), ("target", EvalValue::Agent(a))],
            &[a, b],
            &mut ctx,
        );

        assert!(ctx.adds_f32.is_empty(), "no-op expected for non-matching event kind");
        assert!(ctx.adds_i64.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 2: lazy view — no-op (Expr body has no fold handlers)
    // -----------------------------------------------------------------------

    #[test]
    fn test_lazy_view_noop() {
        use crate::ir::ViewKind;

        let param_a = IrParam {
            name: "a".to_string(),
            local: LocalRef(0),
            ty: IrType::AgentId,
            span: dummy_span(),
        };
        let lazy_view = ViewIR {
            name: "is_hostile".to_string(),
            params: vec![param_a.clone(), IrParam {
                name: "b".to_string(),
                local: LocalRef(1),
                ty: IrType::AgentId,
                span: dummy_span(),
            }],
            return_ty: IrType::Bool,
            body: ViewBodyIR::Expr(lit_float(1.0)),
            annotations: vec![],
            kind: ViewKind::Lazy,
            decay: None,
            span: dummy_span(),
        };

        let mut ctx = MockViewCtx::new(0);
        let a = aid(1);
        let b = aid(2);

        lazy_view.fold("AgentAttacked", &[], &[a, b], &mut ctx);

        assert!(ctx.adds_f32.is_empty(), "lazy view should produce no fold mutations");
        assert!(ctx.adds_i64.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 3: f32 fold (threat_level style: self += 1.0 on AgentAttacked)
    // -----------------------------------------------------------------------

    #[test]
    fn test_f32_fold_on_matching_event() {
        let view = build_view_f32("threat_level", "AgentAttacked", lit_float(1.0));
        let mut ctx = MockViewCtx::new(10);
        let a = aid(1);
        let b = aid(2);

        view.fold(
            "AgentAttacked",
            &[("actor", EvalValue::Agent(b)), ("target", EvalValue::Agent(a))],
            &[a, b],
            &mut ctx,
        );

        assert_eq!(ctx.adds_f32.len(), 1);
        let (ref vn, ref key, delta) = ctx.adds_f32[0];
        assert_eq!(vn, "threat_level");
        assert_eq!(key, &[a, b]);
        assert!((delta - 1.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Test 4: i64 fold (engaged_with style: self += 1 on EngagementCommitted)
    // -----------------------------------------------------------------------

    #[test]
    fn test_i64_fold_on_matching_event() {
        let view = build_view_i64("engaged_with", "EngagementCommitted");
        let mut ctx = MockViewCtx::new(5);
        let a = aid(3);

        view.fold(
            "EngagementCommitted",
            &[("actor", EvalValue::Agent(a))],
            &[a],
            &mut ctx,
        );

        assert_eq!(ctx.adds_i64.len(), 1);
        let (ref vn, ref key, delta) = ctx.adds_i64[0];
        assert_eq!(vn, "engaged_with");
        assert_eq!(key, &[a]);
        assert_eq!(delta, 1);
    }

    // -----------------------------------------------------------------------
    // Test 5: composed body expression — delta is a conditional expression
    //   `if world.tick < 5 { 2.0 } else { 1.0 }`
    // -----------------------------------------------------------------------

    #[test]
    fn test_composed_body_conditional_delta() {
        // Build delta: `if world.tick < 5 { 2.0 } else { 1.0 }`
        let world_tick_expr = IrExprNode {
            kind: IrExpr::NamespaceField {
                ns: NamespaceId::World,
                field: "tick".to_string(),
                ty: crate::ir::IrType::U32,
            },
            span: dummy_span(),
        };
        let cond = IrExprNode {
            kind: IrExpr::Binary(
                BinOp::Lt,
                Box::new(world_tick_expr),
                Box::new(lit_int(5)),
            ),
            span: dummy_span(),
        };
        let delta_expr = IrExprNode {
            kind: IrExpr::If {
                cond: Box::new(cond),
                then_expr: Box::new(lit_float(2.0)),
                else_expr: Some(Box::new(lit_float(1.0))),
            },
            span: dummy_span(),
        };

        // tick=3 < 5 → delta = 2.0
        let view = build_view_f32("threat_level", "AgentAttacked", delta_expr.clone());
        let mut ctx = MockViewCtx::new(3);
        let a = aid(1);
        let b = aid(2);

        view.fold(
            "AgentAttacked",
            &[("actor", EvalValue::Agent(b)), ("target", EvalValue::Agent(a))],
            &[a, b],
            &mut ctx,
        );
        assert!((ctx.adds_f32[0].2 - 2.0).abs() < f32::EPSILON, "expected delta=2.0 when tick<5");

        // tick=10 >= 5 → delta = 1.0
        let view2 = build_view_f32("threat_level", "AgentAttacked", delta_expr);
        let mut ctx2 = MockViewCtx::new(10);
        view2.fold(
            "AgentAttacked",
            &[("actor", EvalValue::Agent(b)), ("target", EvalValue::Agent(a))],
            &[a, b],
            &mut ctx2,
        );
        assert!((ctx2.adds_f32[0].2 - 1.0).abs() < f32::EPSILON, "expected delta=1.0 when tick>=5");
    }

    // -----------------------------------------------------------------------
    // Test 6: Multiple handlers — second event kind also dispatches
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_handlers_both_match() {
        // View listens on two event kinds (like threat_level: AgentAttacked + EffectDamageApplied)
        let param_a = IrParam {
            name: "a".to_string(),
            local: LocalRef(0),
            ty: IrType::AgentId,
            span: dummy_span(),
        };
        let param_b = IrParam {
            name: "b".to_string(),
            local: LocalRef(1),
            ty: IrType::AgentId,
            span: dummy_span(),
        };

        let make_handler = |event_name: &str| FoldHandlerIR {
            pattern: IrEventPattern {
                name: event_name.to_string(),
                event: None,
                bindings: vec![
                    IrPatternBinding {
                        field: "actor".to_string(),
                        value: IrPattern::Bind { name: "b".to_string(), local: LocalRef(1) },
                        span: dummy_span(),
                    },
                    IrPatternBinding {
                        field: "target".to_string(),
                        value: IrPattern::Bind { name: "a".to_string(), local: LocalRef(0) },
                        span: dummy_span(),
                    },
                ],
                span: dummy_span(),
            },
            body: vec![IrStmt::SelfUpdate {
                op: "+=".to_string(),
                value: lit_float(1.0),
                span: dummy_span(),
            }],
            span: dummy_span(),
        };

        let view = ViewIR {
            name: "threat_level".to_string(),
            params: vec![param_a, param_b],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_float(0.0),
                handlers: vec![
                    make_handler("AgentAttacked"),
                    make_handler("EffectDamageApplied"),
                ],
                clamp: None,
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PerEntityTopK { k: 8, keyed_on: 0 }),
            decay: None,
            span: dummy_span(),
        };

        let mut ctx = MockViewCtx::new(0);
        let a = aid(1);
        let b = aid(2);

        // First event kind fires handler 0
        view.fold(
            "AgentAttacked",
            &[("actor", EvalValue::Agent(b)), ("target", EvalValue::Agent(a))],
            &[a, b],
            &mut ctx,
        );
        assert_eq!(ctx.adds_f32.len(), 1, "handler for AgentAttacked should fire");

        // Second event kind fires handler 1
        view.fold(
            "EffectDamageApplied",
            &[("actor", EvalValue::Agent(b)), ("target", EvalValue::Agent(a))],
            &[a, b],
            &mut ctx,
        );
        assert_eq!(ctx.adds_f32.len(), 2, "handler for EffectDamageApplied should fire");
    }

    // -----------------------------------------------------------------------
    // Test 7: Wildcard pattern binding is silently skipped
    // -----------------------------------------------------------------------

    #[test]
    fn test_wildcard_pattern_binding_skipped() {
        // Handler with a wildcard on one field
        let param_a = IrParam {
            name: "a".to_string(),
            local: LocalRef(0),
            ty: IrType::AgentId,
            span: dummy_span(),
        };

        let wildcard_bind = IrPatternBinding {
            field: "target".to_string(),
            value: IrPattern::Wildcard,
            span: dummy_span(),
        };
        let actor_bind = IrPatternBinding {
            field: "actor".to_string(),
            value: IrPattern::Bind { name: "a".to_string(), local: LocalRef(0) },
            span: dummy_span(),
        };

        let handler = FoldHandlerIR {
            pattern: IrEventPattern {
                name: "AgentAttacked".to_string(),
                event: None,
                bindings: vec![actor_bind, wildcard_bind],
                span: dummy_span(),
            },
            body: vec![IrStmt::SelfUpdate {
                op: "+=".to_string(),
                value: lit_float(1.0),
                span: dummy_span(),
            }],
            span: dummy_span(),
        };

        let view = ViewIR {
            name: "my_enemies".to_string(),
            params: vec![param_a],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_float(0.0),
                handlers: vec![handler],
                clamp: None,
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PerEntityTopK { k: 8, keyed_on: 0 }),
            decay: None,
            span: dummy_span(),
        };

        let mut ctx = MockViewCtx::new(0);
        let a = aid(1);
        let b = aid(2);

        view.fold(
            "AgentAttacked",
            &[("actor", EvalValue::Agent(a)), ("target", EvalValue::Agent(b))],
            &[a],
            &mut ctx,
        );

        assert_eq!(ctx.adds_f32.len(), 1, "fold should succeed with wildcard field binding");
    }
}
