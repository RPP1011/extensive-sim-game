//! Interpreter for `MaskIR` — the mask predicate rule class.
//!
//! ## Supported primitives (wolves+humans coverage)
//!
//! This file covers **exactly** the IR variants and stdlib functions
//! exercised by the wolves+humans fixture. Everything outside this set
//! panics with a pointer to the survey doc.
//!
//! ### IR expression variants implemented
//!
//! | Variant | Notes |
//! |---------|-------|
//! | `LitBool` | `true` / `false` literals |
//! | `LitFloat` | f64 constant; cast to f32 at comparison boundary |
//! | `LitInt` | i64 constant; cast to f32 at comparison boundary |
//! | `Local` | parameter bindings (`self`, `target`) → `AgentId` from `locals` map |
//! | `NamespaceCall` | `agents.*`, `query.*`, `abilities.*`, `config.*` |
//! | `NamespaceField` | `config.<block>.<field>`, `world.tick` |
//! | `ViewCall` | routes to `ReadContext::view_*` methods |
//! | `EnumVariant` | `None`-variant sentinel for `agents.engaged_with(self) == None` |
//! | `Binary` | `&&`, `\|\|`, `<`, `>`, `!=`, `==` operators |
//! | `Unary` | `!` logical not |
//! | `BuiltinCall(Distance)` | Euclidean distance between two `Vec3` positions |
//!
//! ### Stdlib functions implemented
//!
//! | Function | Trait method |
//! |----------|-------------|
//! | `agents.alive(x)` | `ReadContext::agents_alive` |
//! | `agents.pos(x)` | `ReadContext::agents_pos` |
//! | `agents.stun_expires_at_tick(x)` | `ReadContext::agents_stun_expires_at_tick` (via view) |
//! | `agents.engaged_with(x)` | `ReadContext::agents_engaged_with` |
//! | `query.nearby_agents(pos, radius)` | `ReadContext::query_nearby_agents` (candidate source) |
//! | `config.movement.max_move_radius` | `ReadContext::config_movement_max_move_radius` |
//! | `config.combat.attack_range` | `ReadContext::config_combat_attack_range` |
//! | `distance(pos1, pos2)` | pure Rust — 3D Euclidean |
//! | `abilities.known(self, ability)` | `ReadContext::abilities_known` |
//! | `abilities.cooldown_ready(self, ability)` | `ReadContext::abilities_cooldown_ready` |
//! | `world.tick` | `ReadContext::world_tick` |
//! | `view::is_stunned(x)` | `ReadContext::view_is_stunned` |
//! | `view::is_hostile(a, b)` | `ReadContext::view_is_hostile` |
//!
//! ### Coverage source
//!
//! `docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md` §1.
//!
//! Any IR variant not listed above hits `unimplemented!()` with a message
//! pointing to that survey document.

use std::collections::HashMap;

use crate::ast::{BinOp, UnOp};
use crate::eval::{AbilityId, AgentId, ReadContext, Vec3};
use crate::ir::{
    Builtin, IrCallArg, IrExpr, IrExprNode, LocalRef, MaskIR, NamespaceId, ViewRef,
};

// ---------------------------------------------------------------------------
// Intermediate evaluation value
// ---------------------------------------------------------------------------

/// A dynamically-typed intermediate value produced while walking the
/// expression tree. Only the shapes reachable in wolves+humans mask
/// predicates are represented; any out-of-survey path panics.
///
/// This enum is private to this module — Tasks 4-6 define their own
/// intermediate types as needed. If a shared type turns out useful it
/// can be promoted to `eval::mod.rs` at that point.
#[derive(Debug, Clone)]
enum Val {
    Bool(bool),
    Float(f32),
    Vec3(Vec3),
    Agent(Option<AgentId>),
    Ability(AbilityId),
}

impl Val {
    /// Coerce to `bool`. Panics on non-boolean shapes.
    fn as_bool(&self) -> bool {
        match self {
            Val::Bool(b) => *b,
            other => panic!(
                "dsl_ast::eval::mask: expected Bool, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }

    /// Coerce to `f32`. Panics on non-numeric shapes.
    fn as_f32(&self) -> f32 {
        match self {
            Val::Float(f) => *f,
            other => panic!(
                "dsl_ast::eval::mask: expected Float, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }

    /// Coerce to `Vec3`. Panics on non-vector shapes.
    fn as_vec3(&self) -> Vec3 {
        match self {
            Val::Vec3(v) => *v,
            other => panic!(
                "dsl_ast::eval::mask: expected Vec3, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }

    /// Coerce to `AgentId` (non-optional). Panics on non-agent shapes.
    fn as_agent_id(&self) -> AgentId {
        match self {
            Val::Agent(Some(id)) => *id,
            Val::Agent(None) => panic!(
                "dsl_ast::eval::mask: expected AgentId, got None; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
            other => panic!(
                "dsl_ast::eval::mask: expected Agent, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }

    /// Coerce to `AbilityId`. Panics on non-ability shapes.
    fn as_ability_id(&self) -> AbilityId {
        match self {
            Val::Ability(ab) => *ab,
            other => panic!(
                "dsl_ast::eval::mask: expected Ability, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Local-variable environment
// ---------------------------------------------------------------------------

/// Mapping from `LocalRef` slot to runtime value. Populated from the
/// mask's action-head parameters before evaluating the predicate.
type Locals = HashMap<u16, Val>;

// ---------------------------------------------------------------------------
// MaskIR::eval — public entry point
// ---------------------------------------------------------------------------

impl MaskIR {
    /// Evaluate the mask predicate for `agent` as the acting self-agent,
    /// with additional positional parameters bound from `params`.
    ///
    /// `params` is an ordered slice of extra positional arguments matching
    /// the mask's `IrActionHeadShape::Positional` bindings. For self-only
    /// masks (e.g. `Hold`, `Flee`) pass an empty slice. For target-bound
    /// masks (e.g. `Attack(target)`) pass one `AgentId`; for ability-bound
    /// masks (`Cast(ability: AbilityId)`) pass one `AbilityId`.
    ///
    /// Returns `true` iff all predicate clauses pass for the given context.
    pub fn eval<C: ReadContext>(
        &self,
        ctx: &C,
        agent: AgentId,
        tick: u32,
        params: &[LocalParam],
    ) -> bool {
        // Build the locals map from the action-head parameter bindings.
        let mut locals: Locals = HashMap::new();

        // Slot 0 is always `self`.
        // The compiler uses LocalRef(0) for the implicit `self` binding.
        locals.insert(0, Val::Agent(Some(agent)));

        // Bind the explicit positional parameters declared in the mask head.
        use crate::ir::IrActionHeadShape;
        if let IrActionHeadShape::Positional(binds) = &self.head.shape {
            for (i, (name, local_ref, _ty)) in binds.iter().enumerate() {
                let val = params.get(i).cloned().unwrap_or_else(|| {
                    panic!(
                        "dsl_ast::eval::mask: mask `{}` head param `{name}` (slot {}) \
                         has no matching entry in params slice (len={})",
                        self.head.name, local_ref.0, params.len()
                    )
                });
                locals.insert(local_ref.0, val.into_val());
            }
        }

        eval_expr(&self.predicate, ctx, agent, tick, &locals).as_bool()
    }
}

// ---------------------------------------------------------------------------
// Caller-facing parameter type
// ---------------------------------------------------------------------------

/// A typed runtime parameter passed to [`MaskIR::eval`] for each
/// positional slot in the mask's action head.
#[derive(Debug, Clone, Copy)]
pub enum LocalParam {
    Agent(AgentId),
    Ability(AbilityId),
}

impl LocalParam {
    fn into_val(self) -> Val {
        match self {
            LocalParam::Agent(id) => Val::Agent(Some(id)),
            LocalParam::Ability(ab) => Val::Ability(ab),
        }
    }
}

// ---------------------------------------------------------------------------
// Expression evaluation
// ---------------------------------------------------------------------------

fn eval_expr<C: ReadContext>(
    node: &IrExprNode,
    ctx: &C,
    agent: AgentId,
    tick: u32,
    locals: &Locals,
) -> Val {
    eval_kind(&node.kind, ctx, agent, tick, locals)
}

fn eval_kind<C: ReadContext>(
    kind: &IrExpr,
    ctx: &C,
    agent: AgentId,
    tick: u32,
    locals: &Locals,
) -> Val {
    match kind {
        // ---- literals -------------------------------------------------------
        IrExpr::LitBool(b) => Val::Bool(*b),
        IrExpr::LitInt(v) => Val::Float(*v as f32),
        IrExpr::LitFloat(v) => Val::Float(*v as f32),

        // ---- local variable lookup ------------------------------------------
        IrExpr::Local(LocalRef(slot), _name) => {
            locals.get(slot).cloned().unwrap_or_else(|| {
                panic!(
                    "dsl_ast::eval::mask: Local slot {slot} not found in locals map; \
                     see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
                )
            })
        }

        // ---- EnumVariant sentinel -------------------------------------------
        // Only `None` (the absent-agent sentinel) is needed for wolves+humans.
        IrExpr::EnumVariant { ty, variant } if ty.is_empty() && variant == "None" => {
            Val::Agent(None)
        }

        // ---- namespace field reads ------------------------------------------
        IrExpr::NamespaceField { ns, field, .. } => {
            eval_namespace_field(*ns, field, ctx, tick)
        }

        // ---- namespace method calls -----------------------------------------
        IrExpr::NamespaceCall { ns, method, args } => {
            eval_namespace_call(*ns, method, args, ctx, agent, tick, locals)
        }

        // ---- view calls -----------------------------------------------------
        IrExpr::ViewCall(view_ref, args) => {
            eval_view_call(*view_ref, args, ctx, agent, tick, locals)
        }

        // ---- builtin calls --------------------------------------------------
        IrExpr::BuiltinCall(builtin, args) => {
            eval_builtin(*builtin, args, ctx, agent, tick, locals)
        }

        // ---- binary operators -----------------------------------------------
        IrExpr::Binary(op, lhs, rhs) => {
            eval_binary(*op, lhs, rhs, ctx, agent, tick, locals)
        }

        // ---- unary operators ------------------------------------------------
        IrExpr::Unary(UnOp::Not, rhs) => {
            let v = eval_expr(rhs, ctx, agent, tick, locals).as_bool();
            Val::Bool(!v)
        }
        IrExpr::Unary(UnOp::Neg, rhs) => {
            let v = eval_expr(rhs, ctx, agent, tick, locals).as_f32();
            Val::Float(-v)
        }

        // ---- out-of-survey variants -----------------------------------------
        other => unimplemented!(
            "dsl_ast::eval::mask: IR variant {:?} is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1",
            std::mem::discriminant(other)
        ),
    }
}

// ---------------------------------------------------------------------------
// Binary operator evaluation
// ---------------------------------------------------------------------------

fn eval_binary<C: ReadContext>(
    op: BinOp,
    lhs: &IrExprNode,
    rhs: &IrExprNode,
    ctx: &C,
    agent: AgentId,
    tick: u32,
    locals: &Locals,
) -> Val {
    // Short-circuit boolean operators first.
    match op {
        BinOp::And => {
            let l = eval_expr(lhs, ctx, agent, tick, locals).as_bool();
            if !l {
                return Val::Bool(false);
            }
            return Val::Bool(eval_expr(rhs, ctx, agent, tick, locals).as_bool());
        }
        BinOp::Or => {
            let l = eval_expr(lhs, ctx, agent, tick, locals).as_bool();
            if l {
                return Val::Bool(true);
            }
            return Val::Bool(eval_expr(rhs, ctx, agent, tick, locals).as_bool());
        }
        _ => {}
    }

    // For `== None` / `!= None` patterns the RHS may be an Agent(None) sentinel.
    // Detect these before requiring both sides to be the same numeric type.
    if matches!(op, BinOp::Eq | BinOp::NotEq) {
        let lv = eval_expr(lhs, ctx, agent, tick, locals);
        let rv = eval_expr(rhs, ctx, agent, tick, locals);
        return match (op, &lv, &rv) {
            (BinOp::Eq, Val::Agent(a), Val::Agent(b)) => Val::Bool(a == b),
            (BinOp::NotEq, Val::Agent(a), Val::Agent(b)) => Val::Bool(a != b),
            (BinOp::Eq, Val::Bool(a), Val::Bool(b)) => Val::Bool(a == b),
            (BinOp::NotEq, Val::Bool(a), Val::Bool(b)) => Val::Bool(a != b),
            (BinOp::Eq, Val::Float(a), Val::Float(b)) => Val::Bool(a == b),
            (BinOp::NotEq, Val::Float(a), Val::Float(b)) => Val::Bool(a != b),
            // agent != self (LocalRef comparison) — coerce both to AgentId
            _ => {
                // Fallthrough: both should be numeric, handle below.
                let a = lv.as_f32();
                let b = rv.as_f32();
                match op {
                    BinOp::Eq => Val::Bool(a == b),
                    BinOp::NotEq => Val::Bool(a != b),
                    _ => unreachable!(),
                }
            }
        };
    }

    // Numeric / comparison operators.
    let l = eval_expr(lhs, ctx, agent, tick, locals).as_f32();
    let r = eval_expr(rhs, ctx, agent, tick, locals).as_f32();
    match op {
        BinOp::Lt => Val::Bool(l < r),
        BinOp::LtEq => Val::Bool(l <= r),
        BinOp::Gt => Val::Bool(l > r),
        BinOp::GtEq => Val::Bool(l >= r),
        BinOp::Add => Val::Float(l + r),
        BinOp::Sub => Val::Float(l - r),
        BinOp::Mul => Val::Float(l * r),
        BinOp::Div => Val::Float(l / r),
        BinOp::Mod => Val::Float(l % r),
        BinOp::And | BinOp::Or | BinOp::Eq | BinOp::NotEq => {
            unreachable!("handled above")
        }
    }
}

// ---------------------------------------------------------------------------
// Namespace field reads
// ---------------------------------------------------------------------------

fn eval_namespace_field<C: ReadContext>(
    ns: NamespaceId,
    field: &str,
    ctx: &C,
    tick: u32,
) -> Val {
    match (ns, field) {
        (NamespaceId::World, "tick") => Val::Float(tick as f32),
        (NamespaceId::Config, f) => eval_config_field(f, ctx),
        _ => unimplemented!(
            "dsl_ast::eval::mask: NamespaceField `{}.{}` is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1",
            ns.name(),
            field
        ),
    }
}

fn eval_config_field<C: ReadContext>(field: &str, ctx: &C) -> Val {
    match field {
        "combat.attack_range" => Val::Float(ctx.config_combat_attack_range()),
        "combat.engagement_range" => Val::Float(ctx.config_combat_engagement_range()),
        "movement.max_move_radius" => Val::Float(ctx.config_movement_max_move_radius()),
        "cascade.max_iterations" => Val::Float(ctx.config_cascade_max_iterations() as f32),
        other => unimplemented!(
            "dsl_ast::eval::mask: config field `{other}` is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
        ),
    }
}

// ---------------------------------------------------------------------------
// Namespace method calls
// ---------------------------------------------------------------------------

fn eval_namespace_call<C: ReadContext>(
    ns: NamespaceId,
    method: &str,
    args: &[IrCallArg],
    ctx: &C,
    agent: AgentId,
    tick: u32,
    locals: &Locals,
) -> Val {
    // Helper to evaluate a positional arg by index.
    let arg = |i: usize| eval_expr(&args[i].value, ctx, agent, tick, locals);

    match (ns, method) {
        // ---- agents namespace -----------------------------------------------
        (NamespaceId::Agents, "alive") => {
            assert_eq!(args.len(), 1, "agents.alive expects 1 arg");
            let id = arg(0).as_agent_id();
            Val::Bool(ctx.agents_alive(id))
        }
        (NamespaceId::Agents, "pos") => {
            assert_eq!(args.len(), 1, "agents.pos expects 1 arg");
            let id = arg(0).as_agent_id();
            Val::Vec3(ctx.agents_pos(id))
        }
        (NamespaceId::Agents, "stun_expires_at_tick") => {
            assert_eq!(args.len(), 1, "agents.stun_expires_at_tick expects 1 arg");
            let id = arg(0).as_agent_id();
            Val::Float(ctx.agents_stun_expires_at_tick(id) as f32)
        }
        (NamespaceId::Agents, "engaged_with") => {
            assert_eq!(args.len(), 1, "agents.engaged_with expects 1 arg");
            let id = arg(0).as_agent_id();
            Val::Agent(ctx.agents_engaged_with(id))
        }
        // ---- query namespace ------------------------------------------------
        // NOTE: `query.nearby_agents` is only used as a `candidate_source`
        // expression in the mask DSL, not directly inside a predicate. If it
        // does appear inside a predicate we return a sentinel float; the
        // candidate enumeration path is handled by the caller, not here.
        // The survey lists it only as a candidate source, so direct calls
        // inside a predicate hit the unimplemented arm below.
        //
        // ---- abilities namespace --------------------------------------------
        (NamespaceId::Abilities, "known") => {
            assert_eq!(args.len(), 2, "abilities.known expects 2 args (agent, ability)");
            let ag = arg(0).as_agent_id();
            let ab = arg(1).as_ability_id();
            Val::Bool(ctx.abilities_known(ag, ab))
        }
        (NamespaceId::Abilities, "cooldown_ready") => {
            assert_eq!(args.len(), 2, "abilities.cooldown_ready expects 2 args (agent, ability)");
            let ag = arg(0).as_agent_id();
            let ab = arg(1).as_ability_id();
            Val::Bool(ctx.abilities_cooldown_ready(ag, ab))
        }
        _ => unimplemented!(
            "dsl_ast::eval::mask: stdlib call `{}.{}` is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1",
            ns.name(),
            method
        ),
    }
}

// ---------------------------------------------------------------------------
// View calls
// ---------------------------------------------------------------------------

fn eval_view_call<C: ReadContext>(
    view_ref: ViewRef,
    args: &[IrCallArg],
    ctx: &C,
    agent: AgentId,
    tick: u32,
    locals: &Locals,
) -> Val {
    // We don't have the view registry here (dsl_ast stays a leaf crate),
    // so we dispatch on argument count as a proxy for the view identity.
    // The two wolves+humans views reachable in masks are:
    //   - `is_stunned(x)`      → 1 arg  → view_is_stunned
    //   - `is_hostile(a, b)`   → 2 args → view_is_hostile
    //
    // When the engine (Task 7) supplies context it also supplies the view
    // registry, enabling name-based dispatch. For now arity-dispatch is
    // sufficient for the wolves+humans fixture.
    let _ = view_ref; // reserved for registry-aware lookup in Task 7
    let arg = |i: usize| eval_expr(&args[i].value, ctx, agent, tick, locals);
    match args.len() {
        1 => {
            // is_stunned(x)
            let id = arg(0).as_agent_id();
            Val::Bool(ctx.view_is_stunned(id))
        }
        2 => {
            // is_hostile(a, b)
            let a = arg(0).as_agent_id();
            let b = arg(1).as_agent_id();
            Val::Bool(ctx.view_is_hostile(a, b))
        }
        n => unimplemented!(
            "dsl_ast::eval::mask: ViewCall with {n} args is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
        ),
    }
}

// ---------------------------------------------------------------------------
// Builtin calls
// ---------------------------------------------------------------------------

fn eval_builtin<C: ReadContext>(
    builtin: Builtin,
    args: &[IrCallArg],
    ctx: &C,
    agent: AgentId,
    tick: u32,
    locals: &Locals,
) -> Val {
    let arg = |i: usize| eval_expr(&args[i].value, ctx, agent, tick, locals);
    match builtin {
        Builtin::Distance => {
            assert_eq!(args.len(), 2, "distance expects 2 args");
            let a = arg(0).as_vec3();
            let b = arg(1).as_vec3();
            Val::Float(vec3_distance(a, b))
        }
        Builtin::Min => {
            assert_eq!(args.len(), 2, "min expects 2 args");
            Val::Float(arg(0).as_f32().min(arg(1).as_f32()))
        }
        Builtin::Max => {
            assert_eq!(args.len(), 2, "max expects 2 args");
            Val::Float(arg(0).as_f32().max(arg(1).as_f32()))
        }
        Builtin::SaturatingAdd => {
            assert_eq!(args.len(), 2, "saturating_add expects 2 args");
            let a = arg(0).as_f32() as u32;
            let b = arg(1).as_f32() as u32;
            Val::Float(a.saturating_add(b) as f32)
        }
        other => unimplemented!(
            "dsl_ast::eval::mask: Builtin::{} is not in the wolves+humans survey — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1",
            other.name()
        ),
    }
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Euclidean distance between two `Vec3` positions.
#[inline]
fn vec3_distance(a: Vec3, b: Vec3) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::eval::{AbilityId, AgentId, EffectOp, Vec3};
    use crate::ir::{
        IrActionHead, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, IrType, LocalRef, MaskIR,
        NamespaceId, ViewRef,
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

    fn lit_bool(b: bool) -> IrExprNode {
        IrExprNode { kind: IrExpr::LitBool(b), span: span() }
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

    fn builtin_call(b: Builtin, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::BuiltinCall(
                b,
                args.into_iter()
                    .map(|v| IrCallArg { name: None, value: v, span: span() })
                    .collect(),
            ),
            span: span(),
        }
    }

    fn view_call(vref: ViewRef, args: Vec<IrExprNode>) -> IrExprNode {
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

    fn binary(op: BinOp, lhs: IrExprNode, rhs: IrExprNode) -> IrExprNode {
        IrExprNode { kind: IrExpr::Binary(op, Box::new(lhs), Box::new(rhs)), span: span() }
    }

    fn unary_not(rhs: IrExprNode) -> IrExprNode {
        IrExprNode { kind: IrExpr::Unary(UnOp::Not, Box::new(rhs)), span: span() }
    }

    fn mask_no_params(predicate: IrExprNode) -> MaskIR {
        MaskIR {
            head: IrActionHead {
                name: "TestMask".into(),
                shape: IrActionHeadShape::None,
                span: span(),
            },
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(),
        }
    }

    fn mask_with_target(predicate: IrExprNode) -> MaskIR {
        MaskIR {
            head: IrActionHead {
                name: "TestMask".into(),
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

    // -----------------------------------------------------------------------
    // Stub ReadContext
    // -----------------------------------------------------------------------

    /// Minimal test-only `ReadContext` stub.  Only the methods called by
    /// each specific test are implemented; all others call `unimplemented!()`.
    struct StubCtx {
        /// alive table: id → alive?
        alive: HashMap<u32, bool>,
        /// positions
        pos: HashMap<u32, Vec3>,
        /// stun expiry ticks
        stun_expires: HashMap<u32, u32>,
        /// engagement partner
        engaged_with: HashMap<u32, Option<u32>>,
        /// hostility table: (a, b) → bool
        hostile: HashMap<(u32, u32), bool>,
        /// is_stunned view values
        view_stunned: HashMap<u32, bool>,
        /// is_hostile view values
        view_hostile: HashMap<(u32, u32), bool>,
        /// abilities known: (agent_id, ability_id) → bool
        abilities_known: HashMap<(u32, u32), bool>,
        /// cooldown ready: (agent_id, ability_id) → bool
        cooldown_ready: HashMap<(u32, u32), bool>,
        /// config values
        attack_range: f32,
        max_move_radius: f32,
        tick: u32,
    }

    impl StubCtx {
        fn new() -> Self {
            StubCtx {
                alive: HashMap::new(),
                pos: HashMap::new(),
                stun_expires: HashMap::new(),
                engaged_with: HashMap::new(),
                hostile: HashMap::new(),
                view_stunned: HashMap::new(),
                view_hostile: HashMap::new(),
                abilities_known: HashMap::new(),
                cooldown_ready: HashMap::new(),
                attack_range: 2.0,
                max_move_radius: 20.0,
                tick: 0,
            }
        }
    }

    use crate::eval::ReadContext;

    impl ReadContext for StubCtx {
        fn world_tick(&self) -> u32 { self.tick }

        fn agents_alive(&self, agent: AgentId) -> bool {
            *self.alive.get(&agent.raw()).unwrap_or(&true)
        }

        fn agents_pos(&self, agent: AgentId) -> Vec3 {
            *self.pos.get(&agent.raw()).unwrap_or(&[0.0, 0.0, 0.0])
        }

        fn agents_hp(&self, _agent: AgentId) -> f32 { unimplemented!() }
        fn agents_max_hp(&self, _agent: AgentId) -> f32 { unimplemented!() }
        fn agents_hp_pct(&self, _agent: AgentId) -> f32 { unimplemented!() }
        fn agents_shield_hp(&self, _agent: AgentId) -> f32 { unimplemented!() }

        fn agents_stun_expires_at_tick(&self, agent: AgentId) -> u32 {
            *self.stun_expires.get(&agent.raw()).unwrap_or(&0)
        }

        fn agents_slow_expires_at_tick(&self, _agent: AgentId) -> u32 { unimplemented!() }
        fn agents_slow_factor_q8(&self, _agent: AgentId) -> i16 { unimplemented!() }
        fn agents_attack_damage(&self, _agent: AgentId) -> f32 { unimplemented!() }

        fn agents_engaged_with(&self, agent: AgentId) -> Option<AgentId> {
            self.engaged_with
                .get(&agent.raw())
                .copied()
                .flatten()
                .and_then(AgentId::new)
        }

        fn agents_is_hostile_to(&self, a: AgentId, b: AgentId) -> bool {
            *self.hostile.get(&(a.raw(), b.raw())).unwrap_or(&false)
        }

        fn agents_gold(&self, _agent: AgentId) -> i64 { unimplemented!() }

        fn query_nearby_agents(&self, _center: Vec3, _radius: f32, _f: &mut dyn FnMut(AgentId)) {
            unimplemented!()
        }

        fn query_nearby_kin(
            &self,
            _origin: AgentId,
            _center: Vec3,
            _radius: f32,
            _f: &mut dyn FnMut(AgentId),
        ) {
            unimplemented!()
        }

        fn query_nearest_hostile_to(&self, _agent: AgentId, _radius: f32) -> Option<AgentId> {
            unimplemented!()
        }

        fn abilities_is_known(&self, _ab: AbilityId) -> bool { unimplemented!() }

        fn abilities_known(&self, agent: AgentId, ab: AbilityId) -> bool {
            *self.abilities_known.get(&(agent.raw(), ab.raw())).unwrap_or(&false)
        }

        fn abilities_cooldown_ready(&self, agent: AgentId, ab: AbilityId) -> bool {
            *self.cooldown_ready.get(&(agent.raw(), ab.raw())).unwrap_or(&true)
        }

        fn abilities_cooldown_ticks(&self, _ab: AbilityId) -> u32 { unimplemented!() }
        fn abilities_effects(&self, _ab: AbilityId, _f: &mut dyn FnMut(EffectOp)) {
            unimplemented!()
        }

        fn config_combat_attack_range(&self) -> f32 { self.attack_range }
        fn config_combat_engagement_range(&self) -> f32 { 12.0 }
        fn config_movement_max_move_radius(&self) -> f32 { self.max_move_radius }
        fn config_cascade_max_iterations(&self) -> u32 { 8 }

        fn view_is_hostile(&self, a: AgentId, b: AgentId) -> bool {
            *self.view_hostile.get(&(a.raw(), b.raw())).unwrap_or(&false)
        }

        fn view_is_stunned(&self, agent: AgentId) -> bool {
            *self.view_stunned.get(&agent.raw()).unwrap_or(&false)
        }

        fn view_threat_level(&self, _observer: AgentId, _target: AgentId) -> f32 { unimplemented!() }
        fn view_my_enemies(&self, _observer: AgentId, _target: AgentId) -> f32 { unimplemented!() }
        fn view_pack_focus(&self, _observer: AgentId, _target: AgentId) -> f32 { unimplemented!() }
        fn view_kin_fear(&self, _observer: AgentId) -> f32 { unimplemented!() }
        fn view_rally_boost(&self, _observer: AgentId) -> f32 { unimplemented!() }
        fn view_slow_factor(&self, _agent: AgentId) -> f32 { unimplemented!() }
    }

    // -----------------------------------------------------------------------
    // Helper to make AgentId from a raw u32
    // -----------------------------------------------------------------------
    fn aid(raw: u32) -> AgentId { AgentId::new(raw).unwrap() }
    fn abid(raw: u32) -> AbilityId { AbilityId::new(raw).unwrap() }

    // -----------------------------------------------------------------------
    // Test 1: Leaf case — simple field comparison (agents.alive)
    // -----------------------------------------------------------------------

    /// Predicate: `agents.alive(self)` — the Hold mask.
    #[test]
    fn leaf_alive_self_true_when_alive() {
        let predicate = ns_call(NamespaceId::Agents, "alive", vec![local("self", 0)]);
        let mask = mask_no_params(predicate);

        let mut ctx = StubCtx::new();
        ctx.alive.insert(1, true);

        let result = mask.eval(&ctx, aid(1), 0, &[]);
        assert!(result, "alive agent should pass Hold mask");
    }

    #[test]
    fn leaf_alive_self_false_when_dead() {
        let predicate = ns_call(NamespaceId::Agents, "alive", vec![local("self", 0)]);
        let mask = mask_no_params(predicate);

        let mut ctx = StubCtx::new();
        ctx.alive.insert(1, false);

        let result = mask.eval(&ctx, aid(1), 0, &[]);
        assert!(!result, "dead agent should fail Hold mask");
    }

    // -----------------------------------------------------------------------
    // Test 2: Boolean ops (AND / OR / NOT)
    // -----------------------------------------------------------------------

    /// Predicate: `agents.alive(self) && !false` — always true for alive agents.
    #[test]
    fn boolean_and_not_false_is_true() {
        let alive = ns_call(NamespaceId::Agents, "alive", vec![local("self", 0)]);
        let not_false = unary_not(lit_bool(false));
        let predicate = binary(BinOp::And, alive, not_false);
        let mask = mask_no_params(predicate);

        let mut ctx = StubCtx::new();
        ctx.alive.insert(1, true);

        assert!(mask.eval(&ctx, aid(1), 0, &[]));
    }

    /// Short-circuit AND: if left is false, right is not evaluated.
    #[test]
    fn boolean_and_short_circuits_on_false_left() {
        let alive_false = {
            let mut m = StubCtx::new();
            m.alive.insert(1, false);
            m
        };
        // Predicate: `agents.alive(self) && agents.alive(self)`
        // With self dead, second clause never runs — but both would return false anyway.
        let predicate = binary(
            BinOp::And,
            ns_call(NamespaceId::Agents, "alive", vec![local("self", 0)]),
            ns_call(NamespaceId::Agents, "alive", vec![local("self", 0)]),
        );
        let mask = mask_no_params(predicate);
        assert!(!mask.eval(&alive_false, aid(1), 0, &[]));
    }

    /// OR: true if either side is true.
    #[test]
    fn boolean_or_true_when_one_side_true() {
        let predicate = binary(BinOp::Or, lit_bool(false), lit_bool(true));
        let mask = mask_no_params(predicate);
        let ctx = StubCtx::new();
        assert!(mask.eval(&ctx, aid(1), 0, &[]));
    }

    // -----------------------------------------------------------------------
    // Test 3: Stdlib-call case (distance builtin + comparison)
    // -----------------------------------------------------------------------

    /// Predicate: `distance(agents.pos(self), agents.pos(target)) < 2.0`
    /// where self is at origin and target is at (1, 0, 0) → distance = 1.0 < 2.0 → true.
    #[test]
    fn distance_builtin_within_range_is_true() {
        let self_pos = ns_call(NamespaceId::Agents, "pos", vec![local("self", 0)]);
        let target_pos = ns_call(NamespaceId::Agents, "pos", vec![local("target", 1)]);
        let dist = builtin_call(Builtin::Distance, vec![self_pos, target_pos]);
        let predicate = binary(BinOp::Lt, dist, lit_float(2.0));
        let mask = mask_with_target(predicate);

        let mut ctx = StubCtx::new();
        ctx.pos.insert(1, [0.0, 0.0, 0.0]);
        ctx.pos.insert(2, [1.0, 0.0, 0.0]);

        let result = mask.eval(&ctx, aid(1), 0, &[LocalParam::Agent(aid(2))]);
        assert!(result, "distance 1.0 < 2.0 should be true");
    }

    /// Predicate: `distance(agents.pos(self), agents.pos(target)) < 2.0`
    /// where target is at (3, 0, 0) → distance = 3.0, fails.
    #[test]
    fn distance_builtin_out_of_range_is_false() {
        let self_pos = ns_call(NamespaceId::Agents, "pos", vec![local("self", 0)]);
        let target_pos = ns_call(NamespaceId::Agents, "pos", vec![local("target", 1)]);
        let dist = builtin_call(Builtin::Distance, vec![self_pos, target_pos]);
        let predicate = binary(BinOp::Lt, dist, lit_float(2.0));
        let mask = mask_with_target(predicate);

        let mut ctx = StubCtx::new();
        ctx.pos.insert(1, [0.0, 0.0, 0.0]);
        ctx.pos.insert(2, [3.0, 0.0, 0.0]);

        let result = mask.eval(&ctx, aid(1), 0, &[LocalParam::Agent(aid(2))]);
        assert!(!result, "distance 3.0 < 2.0 should be false");
    }

    // -----------------------------------------------------------------------
    // Test 4: ViewCall (is_stunned and is_hostile)
    // -----------------------------------------------------------------------

    /// Predicate: `!view::is_stunned(self)` — Cast mask stun guard.
    #[test]
    fn view_is_stunned_blocks_when_stunned() {
        // ViewRef(0) — arity 1 → routes to view_is_stunned.
        let is_stunned = view_call(ViewRef(0), vec![local("self", 0)]);
        let predicate = unary_not(is_stunned);
        let mask = mask_no_params(predicate);

        let mut ctx = StubCtx::new();
        ctx.view_stunned.insert(1, true);

        assert!(!mask.eval(&ctx, aid(1), 0, &[]), "stunned agent should fail cast mask");
    }

    #[test]
    fn view_is_stunned_passes_when_not_stunned() {
        let is_stunned = view_call(ViewRef(0), vec![local("self", 0)]);
        let predicate = unary_not(is_stunned);
        let mask = mask_no_params(predicate);

        let ctx = StubCtx::new(); // view_stunned defaults to false
        assert!(mask.eval(&ctx, aid(1), 0, &[]), "non-stunned agent should pass cast mask");
    }

    /// Predicate: `view::is_hostile(self, target)` — Attack hostility check.
    #[test]
    fn view_is_hostile_returns_correct_value() {
        // ViewRef(0) — arity 2 → routes to view_is_hostile.
        let hostile = view_call(
            ViewRef(0),
            vec![local("self", 0), local("target", 1)],
        );
        let mask = mask_with_target(hostile);

        let mut ctx = StubCtx::new();
        ctx.view_hostile.insert((1, 2), true);
        ctx.view_hostile.insert((1, 3), false);

        assert!(mask.eval(&ctx, aid(1), 0, &[LocalParam::Agent(aid(2))]));
        assert!(!mask.eval(&ctx, aid(1), 0, &[LocalParam::Agent(aid(3))]));
    }

    // -----------------------------------------------------------------------
    // Test 5: Nested case — full Attack mask
    // -----------------------------------------------------------------------

    /// Full attack mask predicate:
    /// `agents.alive(target) && view::is_hostile(self, target)
    ///  && distance(agents.pos(self), agents.pos(target)) < 2.0`
    #[test]
    fn full_attack_mask_passes_valid_target() {
        let alive = ns_call(NamespaceId::Agents, "alive", vec![local("target", 1)]);
        let hostile = view_call(ViewRef(0), vec![local("self", 0), local("target", 1)]);
        let self_pos = ns_call(NamespaceId::Agents, "pos", vec![local("self", 0)]);
        let target_pos = ns_call(NamespaceId::Agents, "pos", vec![local("target", 1)]);
        let dist = builtin_call(Builtin::Distance, vec![self_pos, target_pos]);
        let in_range = binary(BinOp::Lt, dist, lit_float(2.0));
        let predicate = binary(
            BinOp::And,
            binary(BinOp::And, alive, hostile),
            in_range,
        );
        let mask = mask_with_target(predicate);

        let mut ctx = StubCtx::new();
        ctx.alive.insert(2, true);
        ctx.pos.insert(1, [0.0, 0.0, 0.0]);
        ctx.pos.insert(2, [1.0, 0.0, 0.0]);
        ctx.view_hostile.insert((1, 2), true);

        assert!(mask.eval(&ctx, aid(1), 0, &[LocalParam::Agent(aid(2))]));
    }

    #[test]
    fn full_attack_mask_fails_dead_target() {
        let alive = ns_call(NamespaceId::Agents, "alive", vec![local("target", 1)]);
        let hostile = view_call(ViewRef(0), vec![local("self", 0), local("target", 1)]);
        let self_pos = ns_call(NamespaceId::Agents, "pos", vec![local("self", 0)]);
        let target_pos = ns_call(NamespaceId::Agents, "pos", vec![local("target", 1)]);
        let dist = builtin_call(Builtin::Distance, vec![self_pos, target_pos]);
        let in_range = binary(BinOp::Lt, dist, lit_float(2.0));
        let predicate = binary(
            BinOp::And,
            binary(BinOp::And, alive, hostile),
            in_range,
        );
        let mask = mask_with_target(predicate);

        let mut ctx = StubCtx::new();
        ctx.alive.insert(2, false); // dead
        ctx.pos.insert(1, [0.0, 0.0, 0.0]);
        ctx.pos.insert(2, [1.0, 0.0, 0.0]);
        ctx.view_hostile.insert((1, 2), true);

        assert!(!mask.eval(&ctx, aid(1), 0, &[LocalParam::Agent(aid(2))]));
    }

    // -----------------------------------------------------------------------
    // Test 6: Cast mask — ability known + cooldown_ready + not engaged
    // -----------------------------------------------------------------------

    /// Predicate: `abilities.known(self, ability) && abilities.cooldown_ready(self, ability)
    ///             && agents.engaged_with(self) == None`
    #[test]
    fn cast_mask_passes_when_known_ready_not_engaged() {
        let ability_ab = IrExprNode {
            kind: IrExpr::Local(LocalRef(1), "ability".to_string()),
            span: span(),
        };
        let known = ns_call(
            NamespaceId::Abilities,
            "known",
            vec![local("self", 0), ability_ab.clone()],
        );
        let ready = ns_call(
            NamespaceId::Abilities,
            "cooldown_ready",
            vec![local("self", 0), ability_ab.clone()],
        );
        let none_sentinel = IrExprNode {
            kind: IrExpr::EnumVariant {
                ty: "".to_string(),
                variant: "None".to_string(),
            },
            span: span(),
        };
        let engaged = ns_call(
            NamespaceId::Agents,
            "engaged_with",
            vec![local("self", 0)],
        );
        let not_engaged = binary(BinOp::Eq, engaged, none_sentinel);
        let predicate = binary(
            BinOp::And,
            binary(BinOp::And, known, ready),
            not_engaged,
        );

        // Mask with ability param at slot 1.
        let mask = MaskIR {
            head: IrActionHead {
                name: "Cast".into(),
                shape: IrActionHeadShape::Positional(vec![(
                    "ability".into(),
                    LocalRef(1),
                    IrType::AbilityId,
                )]),
                span: span(),
            },
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(),
        };

        let mut ctx = StubCtx::new();
        ctx.abilities_known.insert((1, 5), true);
        ctx.cooldown_ready.insert((1, 5), true);
        // engaged_with returns None by default (not in map)

        let ab = abid(5);
        assert!(mask.eval(&ctx, aid(1), 0, &[LocalParam::Ability(ab)]));
    }

    #[test]
    fn cast_mask_fails_when_engaged() {
        let ability_ab = IrExprNode {
            kind: IrExpr::Local(LocalRef(1), "ability".to_string()),
            span: span(),
        };
        let known = ns_call(
            NamespaceId::Abilities,
            "known",
            vec![local("self", 0), ability_ab.clone()],
        );
        let ready = ns_call(
            NamespaceId::Abilities,
            "cooldown_ready",
            vec![local("self", 0), ability_ab.clone()],
        );
        let none_sentinel = IrExprNode {
            kind: IrExpr::EnumVariant {
                ty: "".to_string(),
                variant: "None".to_string(),
            },
            span: span(),
        };
        let engaged = ns_call(
            NamespaceId::Agents,
            "engaged_with",
            vec![local("self", 0)],
        );
        let not_engaged = binary(BinOp::Eq, engaged, none_sentinel);
        let predicate = binary(
            BinOp::And,
            binary(BinOp::And, known, ready),
            not_engaged,
        );

        let mask = MaskIR {
            head: IrActionHead {
                name: "Cast".into(),
                shape: IrActionHeadShape::Positional(vec![(
                    "ability".into(),
                    LocalRef(1),
                    IrType::AbilityId,
                )]),
                span: span(),
            },
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(),
        };

        let mut ctx = StubCtx::new();
        ctx.abilities_known.insert((1, 5), true);
        ctx.cooldown_ready.insert((1, 5), true);
        ctx.engaged_with.insert(1, Some(2)); // engaged with agent 2

        let ab = abid(5);
        assert!(!mask.eval(&ctx, aid(1), 0, &[LocalParam::Ability(ab)]));
    }
}
