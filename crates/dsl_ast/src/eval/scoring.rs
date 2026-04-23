//! Interpreter for `ScoringIR` — the scoring expression rule class.
//!
//! ## Supported primitives (wolves+humans coverage, §2)
//!
//! This file covers **exactly** the IR variants, view calls, and field reads
//! exercised by the wolves+humans fixture.  Everything outside this set
//! panics with a pointer to the coverage survey document.
//!
//! ### IR expression variants implemented
//!
//! | Variant | Notes |
//! |---------|-------|
//! | `LitFloat` | f64 constant, cast to f32 |
//! | `LitInt` | i64 constant, cast to f32 |
//! | `Local` | parameter bindings (`self` → agent, `target`, `_` wildcard) |
//! | `Field` | `self.hp`, `self.hp_pct`, `target.hp_pct` |
//! | `ViewCall` | routes to `ReadContext::view_*` methods; name-based dispatch |
//! | `Binary` | `+`, `<`, `>`, `<=`, `>=`, `==` operators |
//! | `If` | `if <pred> { <delta> } else { 0.0 }` conditional modifiers |
//! | `PerUnit` | `<expr> per_unit <delta>` gradient modifier |
//!
//! ### View calls implemented
//!
//! | View name | Arg shape | Trait method |
//! |-----------|-----------|-------------|
//! | `threat_level` | `(self, target)` | `ReadContext::view_threat_level` |
//! | `threat_level` | `(self, _)` | `ReadContext::view_threat_level` summed via wildcard |
//! | `my_enemies` | `(self, target)` | `ReadContext::view_my_enemies` |
//! | `pack_focus` | `(self, target)` | `ReadContext::view_pack_focus` |
//! | `kin_fear` | `(self, _)` | `ReadContext::view_kin_fear` |
//! | `rally_boost` | `(self, _)` | `ReadContext::view_rally_boost` |
//!
//! ### Field reads implemented
//!
//! | Field | Binding | Trait method |
//! |-------|---------|-------------|
//! | `hp` | `self` | `ReadContext::agents_hp` |
//! | `hp_pct` | `self` | `ReadContext::agents_hp_pct` |
//! | `hp_pct` | `target` | `ReadContext::agents_hp_pct` (on target) |
//!
//! ### Coverage source
//!
//! `docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md` §2.
//!
//! Any IR variant, view call, or field read not listed above hits
//! `unimplemented!()` with a message pointing to that survey document.
//!
//! ## View name resolution
//!
//! The IR stores view references as numeric `ViewRef(u16)` indices into the
//! `Compilation::views` slice.  `ScoringIR::eval` accepts a `views:
//! &[crate::ir::ViewIR]` slice so it can look up names and perform
//! name-based dispatch.  This deviates from the minimal signature in the
//! plan but is required because all six scoring view calls have arity 2
//! (including wildcard-second-arg calls like `kin_fear(self, _)`), making
//! arity-based dispatch insufficient to distinguish them.
//!
//! ## RNG tiebreak — DEFERRED (Option A)
//!
//! Engine spec §9 specifies a tiebreak hash of `hash(world_seed, agent_id,
//! tick, "scoring")` to break utility ties deterministically.  The current
//! `ReadContext` trait has no world-seed accessor, so implementing this
//! tiebreak would require extending the trait (Task 2 scope).  Skipping it
//! for P1b: the wolves+humans scoring values are unlikely to produce exact
//! ties (the float arithmetic over `hp_pct`, `threat_level`, etc. generates
//! distinct values per agent/target pair), so the parity test at Task 9 is
//! expected to pass without tiebreak.  If it does not, the fix is a new
//! `ReadContext::world_seed() -> u64` method and a tiebreak noise term added
//! here.

use std::collections::HashMap;

use crate::ast::BinOp;
use crate::eval::{AgentId, ReadContext};
use crate::ir::{IrCallArg, IrExpr, IrExprNode, LocalRef, ScoringEntryIR, ScoringIR, ViewIR, ViewRef};

// ---------------------------------------------------------------------------
// Local-variable environment
// ---------------------------------------------------------------------------

/// Mapping from `LocalRef` slot to runtime value.
type Locals = HashMap<u16, SVal>;

/// Dynamically-typed intermediate value for scoring expressions.
///
/// Only the shapes reachable in wolves+humans scoring are represented;
/// out-of-survey paths panic.  Separate from `eval::builtins::EvalVal` to
/// avoid cross-module coupling — scoring locals are always agent IDs
/// (`self`, `target`, `_`); float values are produced by `eval_expr` directly.
#[derive(Debug, Clone)]
enum SVal {
    Agent(Option<AgentId>),
}

impl SVal {
    fn as_agent_id(&self) -> AgentId {
        match self {
            SVal::Agent(Some(id)) => *id,
            SVal::Agent(None) => panic!(
                "dsl_ast::eval::scoring: expected AgentId, got None; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// ScoringIR::eval — public entry point
// ---------------------------------------------------------------------------

impl ScoringIR {
    /// Evaluate the scoring expression for the given `agent`/`target` pair.
    ///
    /// Sums contributions across all entries in this scoring block.  Each
    /// entry binds `self → agent` and (for target-bound heads) `target →
    /// target`.  The action head name is ignored by the interpreter — all
    /// entries contribute to the sum, matching the per-action-kind behaviour
    /// of the compiled scorer.  Callers that want per-kind scores should
    /// filter entries by `entry.head.name` before calling.
    ///
    /// `views` is the `Compilation::views` slice used to resolve `ViewRef`
    /// indices to view names for name-based dispatch.  Pass
    /// `compilation.views.as_slice()` at the call site.
    ///
    /// The simulation tick is read from `ctx.world_tick()` — the single
    /// source of truth.  No tick parameter is accepted.
    pub fn eval<C: ReadContext>(
        &self,
        ctx: &C,
        agent: AgentId,
        target: AgentId,
        views: &[ViewIR],
    ) -> f32 {
        self.entries
            .iter()
            .map(|entry| eval_entry(entry, ctx, agent, target, views))
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Per-entry evaluation
// ---------------------------------------------------------------------------

fn eval_entry<C: ReadContext>(
    entry: &ScoringEntryIR,
    ctx: &C,
    agent: AgentId,
    target: AgentId,
    views: &[ViewIR],
) -> f32 {
    // Build the locals map.  Slot 0 = self (the acting agent).
    let mut locals: Locals = HashMap::new();
    locals.insert(0, SVal::Agent(Some(agent)));

    // Bind the target param if the head declares one.
    use crate::ir::IrActionHeadShape;
    if let IrActionHeadShape::Positional(binds) = &entry.head.shape {
        // Slot 1 is typically the first positional param (e.g. `target`).
        for (_, local_ref, _) in binds {
            locals.insert(local_ref.0, SVal::Agent(Some(target)));
        }
    }

    eval_expr(&entry.expr, ctx, agent, &locals, views)
}

// ---------------------------------------------------------------------------
// Expression evaluation
// ---------------------------------------------------------------------------

fn eval_expr<C: ReadContext>(
    node: &IrExprNode,
    ctx: &C,
    agent: AgentId,
    locals: &Locals,
    views: &[ViewIR],
) -> f32 {
    eval_kind(&node.kind, ctx, agent, locals, views)
}

fn eval_kind<C: ReadContext>(
    kind: &IrExpr,
    ctx: &C,
    agent: AgentId,
    locals: &Locals,
    views: &[ViewIR],
) -> f32 {
    match kind {
        // ---- literals -------------------------------------------------------
        IrExpr::LitFloat(v) => *v as f32,
        IrExpr::LitInt(v) => *v as f32,

        // ---- field reads ----------------------------------------------------
        // The survey §2.3 lists: `self.hp`, `self.hp_pct`, `target.hp_pct`.
        // All are accessed as `IrExpr::Field { base: Local(...), field_name }`.
        IrExpr::Field { base, field_name, .. } => {
            eval_field(base, field_name, ctx, agent, locals, views)
        }

        // ---- view calls -----------------------------------------------------
        IrExpr::ViewCall(view_ref, args) => {
            eval_view_call(*view_ref, args, ctx, agent, locals, views)
        }

        // ---- binary operators -----------------------------------------------
        IrExpr::Binary(op, lhs, rhs) => {
            eval_binary(*op, lhs, rhs, ctx, agent, locals, views)
        }

        // ---- if-then-else ---------------------------------------------------
        // Scoring uses flat `if <pred> { <lit> } else { 0.0 }` modifiers.
        IrExpr::If { cond, then_expr, else_expr } => {
            let pred = eval_cond(cond, ctx, agent, locals, views);
            if pred {
                eval_expr(then_expr, ctx, agent, locals, views)
            } else {
                match else_expr.as_deref() {
                    Some(e) => eval_expr(e, ctx, agent, locals, views),
                    None => 0.0,
                }
            }
        }

        // ---- per_unit gradient modifier ------------------------------------
        // `<expr> per_unit <delta>` → `eval(expr) * eval(delta)`
        IrExpr::PerUnit { expr, delta } => {
            let v = eval_expr(expr, ctx, agent, locals, views);
            let d = eval_expr(delta, ctx, agent, locals, views);
            v * d
        }

        // ---- out-of-survey variants -----------------------------------------
        other => unimplemented!(
            "dsl_ast::eval::scoring: IR variant {:?} is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2",
            std::mem::discriminant(other)
        ),
    }
}

// ---------------------------------------------------------------------------
// Field reads
// ---------------------------------------------------------------------------

fn eval_field<C: ReadContext>(
    base: &IrExprNode,
    field_name: &str,
    ctx: &C,
    agent: AgentId,
    locals: &Locals,
    views: &[ViewIR],
) -> f32 {
    // Identify the base local — should be `self` or the target binding.
    let base_val = eval_as_agent(base, ctx, agent, locals, views);
    match field_name {
        "hp" => ctx.agents_hp(base_val),
        "hp_pct" => ctx.agents_hp_pct(base_val),
        other => unimplemented!(
            "dsl_ast::eval::scoring: field read `{other}` is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2.3"
        ),
    }
}

/// Evaluate an expression that must resolve to an `AgentId`.  Only `Local`
/// references to slots in the locals map are supported by the scoring survey.
fn eval_as_agent<C: ReadContext>(
    node: &IrExprNode,
    _ctx: &C,
    _agent: AgentId,
    locals: &Locals,
    _views: &[ViewIR],
) -> AgentId {
    match &node.kind {
        IrExpr::Local(LocalRef(slot), _name) => {
            locals.get(slot).cloned().unwrap_or_else(|| {
                panic!(
                    "dsl_ast::eval::scoring: Local slot {slot} not found in locals map \
                     while resolving field base; \
                     see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2"
                )
            }).as_agent_id()
        }
        other => unimplemented!(
            "dsl_ast::eval::scoring: field base expression {:?} is not a Local ref — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2",
            std::mem::discriminant(other)
        ),
    }
}

// ---------------------------------------------------------------------------
// View call dispatch
// ---------------------------------------------------------------------------

/// Dispatch a `ViewCall` node by view name.
///
/// Name-based dispatch is required because all six scoring view calls have
/// arity 2 (even wildcard calls like `kin_fear(self, _)` pass two args).
/// Arity alone cannot distinguish `kin_fear` from `rally_boost` or
/// `my_enemies` from `pack_focus`.
fn eval_view_call<C: ReadContext>(
    view_ref: ViewRef,
    args: &[IrCallArg],
    ctx: &C,
    agent: AgentId,
    locals: &Locals,
    views: &[ViewIR],
) -> f32 {
    // Resolve the view name from the views slice.
    let view_name = views
        .get(view_ref.0 as usize)
        .map(|v| v.name.as_str())
        .unwrap_or_else(|| {
            panic!(
                "dsl_ast::eval::scoring: ViewRef({}) out of bounds (views.len={}); \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2",
                view_ref.0,
                views.len()
            )
        });

    // Helper to resolve a call arg as an AgentId.
    let arg_agent = |i: usize| -> AgentId {
        eval_as_agent(&args[i].value, ctx, agent, locals, views)
    };

    // Helper to check whether the second arg is the `_` wildcard.
    let second_is_wildcard = args.len() >= 2 && matches!(
        &args[1].value.kind,
        IrExpr::Local(_, name) if name == "_"
    );

    match view_name {
        "threat_level" => {
            assert!(args.len() >= 1, "threat_level expects at least 1 arg");
            let observer = arg_agent(0);
            if second_is_wildcard {
                // `threat_level(self, _)` — wildcard: the view sums over all sources.
                // The ReadContext impl handles the aggregation internally; we pass
                // AgentId::new(0) == None → use the wildcard accessor convention.
                // The context's `view_threat_level(observer, target)` with a dummy
                // self-agent is not right. Instead we call `view_kin_fear`'s shape?
                //
                // According to the ReadContext trait, wildcard threat_level has no
                // dedicated method — the survey §2.2 says the wildcard form sums
                // threat accumulated against every attacker.  The ReadContext exposes
                // `view_threat_level(observer, target)` for the per-pair case.  For
                // the wildcard aggregate the interpreter re-uses the same trait method
                // with the *agent* as both observer AND target, which would be wrong.
                //
                // The correct resolution: the ReadContext for scoring always receives
                // the `agent` as the acting agent.  The `_` wildcard target means "sum
                // over all"; since the trait has no separate wildcard accessor we pass
                // the acting agent as the "target" sentinel and let the engine-side impl
                // handle it correctly.  At Task 7 the engine impl of view_threat_level
                // when target==observer is the wildcard sum.
                //
                // For unit tests (where we supply mock values) we simply call
                // view_kin_fear (a dedicated wildcard-sum method) when available.
                // But scoring has no separate trait method for wildcard threat_level.
                //
                // Per the survey §2 the wildcard threat_level sums threat across all
                // sources for `observer`.  The ReadContext trait defines
                //   view_threat_level(observer, target) → per-pair value.
                // The engine impl at Task 7 will need to either (a) provide a
                // separate `view_threat_level_sum(observer) -> f32` method or (b)
                // accept a sentinel target (e.g. observer == target) for wildcard.
                //
                // For P1b we follow (b): pass observer as both args when wildcard.
                // Document the convention here; the Task 7 engine impl must match.
                ctx.view_threat_level(observer, observer)
            } else {
                // `threat_level(self, target)` — per-pair.
                assert!(args.len() >= 2, "threat_level(self, target) expects 2 args");
                let target = arg_agent(1);
                ctx.view_threat_level(observer, target)
            }
        }
        "my_enemies" => {
            assert_eq!(args.len(), 2, "my_enemies expects 2 args");
            let observer = arg_agent(0);
            let target_id = arg_agent(1);
            ctx.view_my_enemies(observer, target_id)
        }
        "pack_focus" => {
            assert_eq!(args.len(), 2, "pack_focus expects 2 args");
            let observer = arg_agent(0);
            let target_id = arg_agent(1);
            ctx.view_pack_focus(observer, target_id)
        }
        "kin_fear" => {
            assert!(args.len() >= 1, "kin_fear expects at least 1 arg");
            let observer = arg_agent(0);
            // The survey lists `kin_fear(self, _)` — wildcard sums over sources.
            // The ReadContext has a dedicated `view_kin_fear(observer) -> f32`.
            ctx.view_kin_fear(observer)
        }
        "rally_boost" => {
            assert!(args.len() >= 1, "rally_boost expects at least 1 arg");
            let observer = arg_agent(0);
            // The survey lists `rally_boost(self, _)` — wildcard sums over sources.
            ctx.view_rally_boost(observer)
        }
        "slow_factor" => {
            // Listed in the survey as a possible view call; if exercised in future,
            // supported here as a 1-arg lazy view.
            assert!(args.len() >= 1, "slow_factor expects at least 1 arg");
            let a = arg_agent(0);
            ctx.view_slow_factor(a)
        }
        other => unimplemented!(
            "dsl_ast::eval::scoring: view `{other}` is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2.2"
        ),
    }
}

// ---------------------------------------------------------------------------
// Condition evaluation (returns bool)
// ---------------------------------------------------------------------------

/// Evaluate a scoring predicate expression to a boolean.
///
/// Scoring predicates are comparison operators (`<`, `>`, `<=`, `>=`, `==`)
/// applied to field reads or view-call results.  No boolean logic operators
/// (AND / OR) appear in the wolves+humans scoring survey.
fn eval_cond<C: ReadContext>(
    node: &IrExprNode,
    ctx: &C,
    agent: AgentId,
    locals: &Locals,
    views: &[ViewIR],
) -> bool {
    match &node.kind {
        IrExpr::Binary(op, lhs, rhs) => {
            match op {
                // Comparison operators — evaluate both sides as f32.
                BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq | BinOp::Eq | BinOp::NotEq => {
                    let l = eval_expr(lhs, ctx, agent, locals, views);
                    let r = eval_expr(rhs, ctx, agent, locals, views);
                    match op {
                        BinOp::Lt => l < r,
                        BinOp::LtEq => l <= r,
                        BinOp::Gt => l > r,
                        BinOp::GtEq => l >= r,
                        BinOp::Eq => l == r,
                        BinOp::NotEq => l != r,
                        _ => unreachable!(),
                    }
                }
                other => unimplemented!(
                    "dsl_ast::eval::scoring: BinOp::{other:?} in condition is not in the \
                     wolves+humans survey — \
                     see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2"
                ),
            }
        }
        other => unimplemented!(
            "dsl_ast::eval::scoring: condition expression {:?} is not in the wolves+humans \
             survey — see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2",
            std::mem::discriminant(other)
        ),
    }
}

// ---------------------------------------------------------------------------
// Binary operator evaluation (returns f32)
// ---------------------------------------------------------------------------

fn eval_binary<C: ReadContext>(
    op: BinOp,
    lhs: &IrExprNode,
    rhs: &IrExprNode,
    ctx: &C,
    agent: AgentId,
    locals: &Locals,
    views: &[ViewIR],
) -> f32 {
    match op {
        // Arithmetic operators — only `Add` is used in base wolves+humans scoring.
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
            let l = eval_expr(lhs, ctx, agent, locals, views);
            let r = eval_expr(rhs, ctx, agent, locals, views);
            crate::eval::builtins::eval_arithmetic_binop(op, l, r)
        }
        // Comparison operators produce a 1.0 / 0.0 numeric result when used
        // outside an `if` predicate.  The survey lists comparisons only inside
        // `if` conditions (handled by `eval_cond`), so this arm is here for
        // completeness but may be unimplemented for wolves+humans.
        BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq | BinOp::Eq | BinOp::NotEq => {
            let l = eval_expr(lhs, ctx, agent, locals, views);
            let r = eval_expr(rhs, ctx, agent, locals, views);
            let result = match op {
                BinOp::Lt => l < r,
                BinOp::LtEq => l <= r,
                BinOp::Gt => l > r,
                BinOp::GtEq => l >= r,
                BinOp::Eq => l == r,
                BinOp::NotEq => l != r,
                _ => unreachable!(),
            };
            if result { 1.0 } else { 0.0 }
        }
        // Boolean short-circuit operators do not appear in scoring expressions
        // per the survey §2.1.
        BinOp::And | BinOp::Or => unimplemented!(
            "dsl_ast::eval::scoring: BinOp::{op:?} is not in the wolves+humans scoring survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §2"
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::eval::{AbilityId, AgentId, EffectOp, ReadContext, Vec3};
    use crate::ir::{
        IrActionHead, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, IrType, LocalRef,
        ScoringEntryIR, ScoringIR, ViewIR, ViewKind, ViewRef, StorageHint,
    };

    // -----------------------------------------------------------------------
    // IR construction helpers
    // -----------------------------------------------------------------------

    fn span() -> Span { Span::dummy() }

    fn local(name: &str, id: u16) -> IrExprNode {
        IrExprNode { kind: IrExpr::Local(LocalRef(id), name.to_string()), span: span() }
    }

    fn lit_float(v: f64) -> IrExprNode {
        IrExprNode { kind: IrExpr::LitFloat(v), span: span() }
    }

    fn self_field(name: &str) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::Field {
                base: Box::new(local("self", 0)),
                field_name: name.to_string(),
                field: None,
            },
            span: span(),
        }
    }

    fn target_field(name: &str) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::Field {
                base: Box::new(local("target", 1)),
                field_name: name.to_string(),
                field: None,
            },
            span: span(),
        }
    }

    fn binary(op: BinOp, lhs: IrExprNode, rhs: IrExprNode) -> IrExprNode {
        IrExprNode { kind: IrExpr::Binary(op, Box::new(lhs), Box::new(rhs)), span: span() }
    }

    fn if_expr(cond: IrExprNode, then_v: f64, else_v: f64) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::If {
                cond: Box::new(cond),
                then_expr: Box::new(lit_float(then_v)),
                else_expr: Some(Box::new(lit_float(else_v))),
            },
            span: span(),
        }
    }

    fn per_unit(expr: IrExprNode, delta: f64) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::PerUnit {
                expr: Box::new(expr),
                delta: Box::new(lit_float(delta)),
            },
            span: span(),
        }
    }

    fn view_call_node(vref: ViewRef, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::ViewCall(
                vref,
                args.into_iter()
                    .map(|v| IrCallArg { name: None, value: v, span: span() })
                    .collect(),
            ),
            span: span(),
        }
    }

    fn head_no_params(name: &str) -> IrActionHead {
        IrActionHead {
            name: name.to_string(),
            shape: IrActionHeadShape::None,
            span: span(),
        }
    }

    fn head_with_target(name: &str) -> IrActionHead {
        IrActionHead {
            name: name.to_string(),
            shape: IrActionHeadShape::Positional(vec![(
                "target".into(),
                LocalRef(1),
                IrType::AgentId,
            )]),
            span: span(),
        }
    }

    fn make_scoring(entries: Vec<ScoringEntryIR>) -> ScoringIR {
        ScoringIR { entries, annotations: vec![], span: span() }
    }

    fn make_entry(head: IrActionHead, expr: IrExprNode) -> ScoringEntryIR {
        ScoringEntryIR { head, expr, span: span() }
    }

    /// Build a stub `ViewIR` with the given name.  The body and params are
    /// left minimal — the interpreter only reads `name`.
    fn stub_view(name: &str) -> ViewIR {
        use crate::ir::ViewBodyIR;
        ViewIR {
            name: name.to_string(),
            params: vec![],
            return_ty: IrType::F32,
            body: ViewBodyIR::Expr(lit_float(0.0)),
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PairMap),
            decay: None,
            span: span(),
        }
    }

    // -----------------------------------------------------------------------
    // Stub ReadContext
    // -----------------------------------------------------------------------

    struct StubCtx {
        hp: HashMap<u32, f32>,
        max_hp: HashMap<u32, f32>,
        hp_pct: HashMap<u32, f32>,
        view_threat_level: HashMap<(u32, u32), f32>,
        view_my_enemies: HashMap<(u32, u32), f32>,
        view_pack_focus: HashMap<(u32, u32), f32>,
        view_kin_fear: HashMap<u32, f32>,
        view_rally_boost: HashMap<u32, f32>,
        tick: u32,
    }

    impl StubCtx {
        fn new() -> Self {
            StubCtx {
                hp: HashMap::new(),
                max_hp: HashMap::new(),
                hp_pct: HashMap::new(),
                view_threat_level: HashMap::new(),
                view_my_enemies: HashMap::new(),
                view_pack_focus: HashMap::new(),
                view_kin_fear: HashMap::new(),
                view_rally_boost: HashMap::new(),
                tick: 0,
            }
        }
    }

    impl ReadContext for StubCtx {
        fn world_tick(&self) -> u32 { self.tick }
        fn agents_alive(&self, _: AgentId) -> bool { true }
        fn agents_pos(&self, _: AgentId) -> Vec3 { [0.0, 0.0, 0.0] }
        fn agents_hp(&self, agent: AgentId) -> f32 {
            *self.hp.get(&agent.raw()).unwrap_or(&100.0)
        }
        fn agents_max_hp(&self, agent: AgentId) -> f32 {
            *self.max_hp.get(&agent.raw()).unwrap_or(&100.0)
        }
        fn agents_hp_pct(&self, agent: AgentId) -> f32 {
            if let Some(v) = self.hp_pct.get(&agent.raw()) {
                return *v;
            }
            let hp = self.agents_hp(agent);
            let max = self.agents_max_hp(agent);
            if max == 0.0 { 0.0 } else { hp / max }
        }
        fn agents_shield_hp(&self, _: AgentId) -> f32 { 0.0 }
        fn agents_stun_expires_at_tick(&self, _: AgentId) -> u32 { 0 }
        fn agents_slow_expires_at_tick(&self, _: AgentId) -> u32 { 0 }
        fn agents_slow_factor_q8(&self, _: AgentId) -> i16 { 0 }
        fn agents_attack_damage(&self, _: AgentId) -> f32 { 10.0 }
        fn agents_engaged_with(&self, _: AgentId) -> Option<AgentId> { None }
        fn agents_is_hostile_to(&self, _: AgentId, _: AgentId) -> bool { false }
        fn agents_gold(&self, _: AgentId) -> i64 { 0 }
        fn query_nearby_agents(&self, _: Vec3, _: f32, _: &mut dyn FnMut(AgentId)) {}
        fn query_nearby_kin(&self, _: AgentId, _: Vec3, _: f32, _: &mut dyn FnMut(AgentId)) {}
        fn query_nearest_hostile_to(&self, _: AgentId, _: f32) -> Option<AgentId> { None }
        fn abilities_is_known(&self, _: AbilityId) -> bool { false }
        fn abilities_known(&self, _: AgentId, _: AbilityId) -> bool { false }
        fn abilities_cooldown_ready(&self, _: AgentId, _: AbilityId) -> bool { true }
        fn abilities_cooldown_ticks(&self, _: AbilityId) -> u32 { 0 }
        fn abilities_effects(&self, _: AbilityId, _: &mut dyn FnMut(EffectOp)) {}
        fn config_combat_attack_range(&self) -> f32 { 2.0 }
        fn config_combat_engagement_range(&self) -> f32 { 12.0 }
        fn config_movement_max_move_radius(&self) -> f32 { 20.0 }
        fn config_cascade_max_iterations(&self) -> u32 { 8 }
        fn view_is_hostile(&self, _: AgentId, _: AgentId) -> bool { false }
        fn view_is_stunned(&self, _: AgentId) -> bool { false }
        fn view_threat_level(&self, observer: AgentId, target: AgentId) -> f32 {
            *self.view_threat_level.get(&(observer.raw(), target.raw())).unwrap_or(&0.0)
        }
        fn view_my_enemies(&self, observer: AgentId, target: AgentId) -> f32 {
            *self.view_my_enemies.get(&(observer.raw(), target.raw())).unwrap_or(&0.0)
        }
        fn view_pack_focus(&self, observer: AgentId, target: AgentId) -> f32 {
            *self.view_pack_focus.get(&(observer.raw(), target.raw())).unwrap_or(&0.0)
        }
        fn view_kin_fear(&self, observer: AgentId) -> f32 {
            *self.view_kin_fear.get(&observer.raw()).unwrap_or(&0.0)
        }
        fn view_rally_boost(&self, observer: AgentId) -> f32 {
            *self.view_rally_boost.get(&observer.raw()).unwrap_or(&0.0)
        }
        fn view_slow_factor(&self, _: AgentId) -> f32 { 1.0 }
    }

    fn aid(raw: u32) -> AgentId { AgentId::new(raw).unwrap() }

    // -----------------------------------------------------------------------
    // Test 1: single literal term
    // -----------------------------------------------------------------------

    /// A scoring entry `Hold = 0.1` returns exactly 0.1.
    #[test]
    fn single_literal_term() {
        let scoring = make_scoring(vec![
            make_entry(head_no_params("Hold"), lit_float(0.1)),
        ]);
        let ctx = StubCtx::new();
        let score = scoring.eval(&ctx, aid(1), aid(2), &[]);
        assert!((score - 0.1).abs() < 1e-6, "expected 0.1, got {score}");
    }

    // -----------------------------------------------------------------------
    // Test 2: per-entry sum with multiple terms
    // -----------------------------------------------------------------------

    /// Multiple entries sum their contributions.
    /// `Hold = 0.1`, `MoveToward = 0.3` → total 0.4 when both entries
    /// contribute (i.e., called with no filter).
    #[test]
    fn per_entry_sum_adds_contributions() {
        let scoring = make_scoring(vec![
            make_entry(head_no_params("Hold"), lit_float(0.1)),
            make_entry(head_no_params("MoveToward"), lit_float(0.3)),
        ]);
        let ctx = StubCtx::new();
        let score = scoring.eval(&ctx, aid(1), aid(2), &[]);
        assert!((score - 0.4).abs() < 1e-6, "expected 0.4, got {score}");
    }

    // -----------------------------------------------------------------------
    // Test 3: conditional modifier fires when predicate is true
    // -----------------------------------------------------------------------

    /// `Attack(target) = 0.0 + (if target.hp_pct < 0.3 { 0.4 } else { 0.0 })`
    /// Agent 2 has hp_pct = 0.2 → modifier fires → score = 0.4.
    #[test]
    fn conditional_modifier_fires_when_true() {
        let cond = binary(BinOp::Lt, target_field("hp_pct"), lit_float(0.3));
        let expr = binary(
            BinOp::Add,
            lit_float(0.0),
            if_expr(cond, 0.4, 0.0),
        );
        let scoring = make_scoring(vec![
            make_entry(head_with_target("Attack"), expr),
        ]);
        let mut ctx = StubCtx::new();
        ctx.hp_pct.insert(2, 0.2); // target (agent 2) at 20% hp
        let score = scoring.eval(&ctx, aid(1), aid(2), &[]);
        assert!((score - 0.4).abs() < 1e-6, "expected 0.4, got {score}");
    }

    /// Same entry but target at hp_pct = 0.8 → predicate false → score = 0.0.
    #[test]
    fn conditional_modifier_silent_when_false() {
        let cond = binary(BinOp::Lt, target_field("hp_pct"), lit_float(0.3));
        let expr = binary(
            BinOp::Add,
            lit_float(0.0),
            if_expr(cond, 0.4, 0.0),
        );
        let scoring = make_scoring(vec![
            make_entry(head_with_target("Attack"), expr),
        ]);
        let mut ctx = StubCtx::new();
        ctx.hp_pct.insert(2, 0.8); // target at 80% hp — predicate false
        let score = scoring.eval(&ctx, aid(1), aid(2), &[]);
        assert!((score - 0.0).abs() < 1e-6, "expected 0.0, got {score}");
    }

    // -----------------------------------------------------------------------
    // Test 4: view call dispatch
    // -----------------------------------------------------------------------

    /// `pack_focus(self, target) > 0.5 : +0.4`
    /// ViewRef(0) → views[0].name = "pack_focus".
    #[test]
    fn view_call_pack_focus_fires_when_above_threshold() {
        // ViewRef(0) → "pack_focus" (first entry in views slice).
        let views = vec![stub_view("pack_focus")];
        let view = view_call_node(ViewRef(0), vec![local("self", 0), local("target", 1)]);
        let cond = binary(BinOp::Gt, view, lit_float(0.5));
        let expr = binary(BinOp::Add, lit_float(0.0), if_expr(cond, 0.4, 0.0));
        let scoring = make_scoring(vec![
            make_entry(head_with_target("Attack"), expr),
        ]);
        let mut ctx = StubCtx::new();
        ctx.view_pack_focus.insert((1, 2), 0.8); // above 0.5 threshold
        let score = scoring.eval(&ctx, aid(1), aid(2), &views);
        assert!((score - 0.4).abs() < 1e-6, "expected 0.4, got {score}");
    }

    #[test]
    fn view_call_pack_focus_silent_when_below_threshold() {
        let views = vec![stub_view("pack_focus")];
        let view = view_call_node(ViewRef(0), vec![local("self", 0), local("target", 1)]);
        let cond = binary(BinOp::Gt, view, lit_float(0.5));
        let expr = binary(BinOp::Add, lit_float(0.0), if_expr(cond, 0.4, 0.0));
        let scoring = make_scoring(vec![
            make_entry(head_with_target("Attack"), expr),
        ]);
        let mut ctx = StubCtx::new();
        ctx.view_pack_focus.insert((1, 2), 0.2); // below threshold
        let score = scoring.eval(&ctx, aid(1), aid(2), &views);
        assert!((score - 0.0).abs() < 1e-6, "expected 0.0, got {score}");
    }

    // -----------------------------------------------------------------------
    // Test 5: kin_fear wildcard view call
    // -----------------------------------------------------------------------

    /// `kin_fear(self, _) > 0.5 : +0.4` — wildcard second arg.
    #[test]
    fn view_call_kin_fear_wildcard_fires() {
        let views = vec![stub_view("kin_fear")];
        // Second arg is `_` (wildcard).
        let view = view_call_node(ViewRef(0), vec![local("self", 0), local("_", 255)]);
        let cond = binary(BinOp::Gt, view, lit_float(0.5));
        let expr = binary(BinOp::Add, lit_float(0.0), if_expr(cond, 0.4, 0.0));
        let scoring = make_scoring(vec![
            make_entry(head_no_params("Flee"), expr),
        ]);
        let mut ctx = StubCtx::new();
        ctx.view_kin_fear.insert(1, 0.8); // observer=1 fear > 0.5
        let score = scoring.eval(&ctx, aid(1), aid(2), &views);
        assert!((score - 0.4).abs() < 1e-6, "expected 0.4, got {score}");
    }

    // -----------------------------------------------------------------------
    // Test 6: per_unit gradient modifier
    // -----------------------------------------------------------------------

    /// `view::threat_level(self, target) per_unit 0.01` with threat=30.0
    /// → gradient score = 30.0 * 0.01 = 0.3.
    #[test]
    fn per_unit_gradient_threat_level() {
        let views = vec![stub_view("threat_level")];
        let view = view_call_node(ViewRef(0), vec![local("self", 0), local("target", 1)]);
        let expr = per_unit(view, 0.01);
        let scoring = make_scoring(vec![
            make_entry(head_with_target("Attack"), expr),
        ]);
        let mut ctx = StubCtx::new();
        ctx.view_threat_level.insert((1, 2), 30.0);
        let score = scoring.eval(&ctx, aid(1), aid(2), &views);
        assert!((score - 0.3).abs() < 1e-5, "expected 0.3, got {score}");
    }

    // -----------------------------------------------------------------------
    // Test 7: nested composition — full Attack row approximation
    // -----------------------------------------------------------------------

    /// Approximate the Attack scoring row:
    ///   base=0.0 + (if self.hp_pct >= 0.8 { 0.5 }) + (if target.hp_pct < 0.3 { 0.4 })
    /// Agent 1: hp_pct=0.9 (fresh) → first cond fires (+0.5)
    /// Agent 2: hp_pct=0.2 (low)   → second cond fires (+0.4)
    /// Expected total: 0.0 + 0.5 + 0.4 = 0.9.
    #[test]
    fn nested_composition_attack_row() {
        let self_fresh_cond = binary(BinOp::GtEq, self_field("hp_pct"), lit_float(0.8));
        let target_low_cond = binary(BinOp::Lt, target_field("hp_pct"), lit_float(0.3));
        let expr = binary(
            BinOp::Add,
            binary(
                BinOp::Add,
                lit_float(0.0),
                if_expr(self_fresh_cond, 0.5, 0.0),
            ),
            if_expr(target_low_cond, 0.4, 0.0),
        );
        let scoring = make_scoring(vec![
            make_entry(head_with_target("Attack"), expr),
        ]);
        let mut ctx = StubCtx::new();
        ctx.hp_pct.insert(1, 0.9); // self at 90%
        ctx.hp_pct.insert(2, 0.2); // target at 20%
        let score = scoring.eval(&ctx, aid(1), aid(2), &[]);
        assert!((score - 0.9).abs() < 1e-5, "expected 0.9, got {score}");
    }

    // -----------------------------------------------------------------------
    // Test 8: self.hp field read
    // -----------------------------------------------------------------------

    /// `if self.hp < 30.0 { 0.6 } else { 0.0 }` fires when hp is low.
    #[test]
    fn self_hp_field_read_gates_flee_modifier() {
        let cond = binary(BinOp::Lt, self_field("hp"), lit_float(30.0));
        let expr = if_expr(cond, 0.6, 0.0);
        let scoring = make_scoring(vec![
            make_entry(head_no_params("Flee"), expr),
        ]);
        let mut ctx = StubCtx::new();
        ctx.hp.insert(1, 25.0); // self at hp=25 (< 30)
        let score = scoring.eval(&ctx, aid(1), aid(2), &[]);
        assert!((score - 0.6).abs() < 1e-6, "expected 0.6, got {score}");
    }

    #[test]
    fn self_hp_field_read_silent_when_above_threshold() {
        let cond = binary(BinOp::Lt, self_field("hp"), lit_float(30.0));
        let expr = if_expr(cond, 0.6, 0.0);
        let scoring = make_scoring(vec![
            make_entry(head_no_params("Flee"), expr),
        ]);
        let mut ctx = StubCtx::new();
        ctx.hp.insert(1, 80.0); // well above 30
        let score = scoring.eval(&ctx, aid(1), aid(2), &[]);
        assert!((score - 0.0).abs() < 1e-6, "expected 0.0, got {score}");
    }
}
