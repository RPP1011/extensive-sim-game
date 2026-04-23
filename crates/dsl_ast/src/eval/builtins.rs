//! Shared numeric helpers reused by multiple IR interpreter modules.
//!
//! ## Scope
//!
//! This module contains helpers that are **not** specific to a single rule
//! class.  The wolves+humans coverage survey
//! (`docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md`)
//! classifies these as:
//!
//! - **§1.1 / §2.1** — Arithmetic binary operators (`Add`, `Sub`, `Mul`,
//!   `Div`, `Mod`): appear in mask and scoring expressions.
//! - **§3.3** — Numeric builtins (`min`, `max`, `saturating_add`,
//!   `distance`): physics-scope per the survey, but also exercised in
//!   masks. Extracted here so Tasks 4 (ScoringIR) and 5 (PhysicsIR) can
//!   reuse rather than duplicate.
//!
//! Any arithmetic op not handled here returns an `unimplemented!()` panic
//! with a survey pointer at the call site.

use crate::ast::BinOp;
use crate::eval::Vec3;

// ---------------------------------------------------------------------------
// Intermediate value (shared across interpreter modules)
// ---------------------------------------------------------------------------

/// A dynamically-typed intermediate value produced while walking any IR
/// expression tree.
///
/// Only the shapes reachable in the wolves+humans fixture are represented.
/// Out-of-survey paths panic with a pointer to the survey document.
///
/// Named `EvalVal` rather than `Val` to avoid ambiguity when imported
/// alongside other `Val` types in future modules.
#[derive(Debug, Clone)]
pub(crate) enum EvalVal {
    Bool(bool),
    Float(f32),
    Vec3(Vec3),
    Agent(Option<crate::eval::AgentId>),
    Ability(crate::eval::AbilityId),
}

// Allow dead_code on the full impl: as_bool/as_agent_id/as_ability_id are not
// called from mask.rs (only as_f32/as_vec3 are needed there), but Tasks 4 and
// 5 will use the full surface.  Suppress the warning now rather than littering
// per-method attributes.
#[allow(dead_code)]
impl EvalVal {
    /// Coerce to `bool`. Panics on non-boolean shapes.
    #[inline]
    pub(crate) fn as_bool(&self) -> bool {
        match self {
            EvalVal::Bool(b) => *b,
            other => panic!(
                "dsl_ast::eval: expected Bool, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }

    /// Coerce to `f32`. Panics on non-numeric shapes.
    #[inline]
    pub(crate) fn as_f32(&self) -> f32 {
        match self {
            EvalVal::Float(f) => *f,
            other => panic!(
                "dsl_ast::eval: expected Float, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }

    /// Coerce to `Vec3`. Panics on non-vector shapes.
    #[inline]
    pub(crate) fn as_vec3(&self) -> Vec3 {
        match self {
            EvalVal::Vec3(v) => *v,
            other => panic!(
                "dsl_ast::eval: expected Vec3, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }

    /// Coerce to `AgentId` (non-optional). Panics on non-agent or absent-agent shapes.
    #[inline]
    pub(crate) fn as_agent_id(&self) -> crate::eval::AgentId {
        match self {
            EvalVal::Agent(Some(id)) => *id,
            EvalVal::Agent(None) => panic!(
                "dsl_ast::eval: expected AgentId, got None; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
            other => panic!(
                "dsl_ast::eval: expected Agent, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }

    /// Coerce to `AbilityId`. Panics on non-ability shapes.
    #[inline]
    pub(crate) fn as_ability_id(&self) -> crate::eval::AbilityId {
        match self {
            EvalVal::Ability(ab) => *ab,
            other => panic!(
                "dsl_ast::eval: expected Ability, got {other:?}; \
                 see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic binary operators
// ---------------------------------------------------------------------------

/// Evaluate an arithmetic binary operator over two `f32` operands.
///
/// Handles `Add`, `Sub`, `Mul`, `Div`, `Mod`.  All other `BinOp` variants
/// are not arithmetic and must be handled by the caller (boolean short-circuit
/// ops, comparison ops, etc.).
///
/// # Panics
///
/// Panics with a survey-pointer message if `op` is not one of the five
/// arithmetic variants.  Callers should only invoke this function after
/// confirming `op` is arithmetic.
#[inline]
pub(crate) fn eval_arithmetic_binop(op: BinOp, l: f32, r: f32) -> f32 {
    match op {
        BinOp::Add => l + r,
        BinOp::Sub => l - r,
        BinOp::Mul => l * r,
        BinOp::Div => l / r,
        BinOp::Mod => l % r,
        other => unimplemented!(
            "dsl_ast::eval::builtins: BinOp::{other:?} is not an arithmetic op — \
             see docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md §1"
        ),
    }
}

// ---------------------------------------------------------------------------
// Numeric builtin functions
// ---------------------------------------------------------------------------

/// Evaluate a numeric builtin function by name.
///
/// Handles `min`, `max`, `saturating_add`, and `distance`.  Returns
/// `Some(EvalVal)` when the name matches; returns `None` when the name is
/// unrecognised so the caller can fall through to its class-specific dispatch.
///
/// Per the coverage survey (§3.3), `min`/`max`/`saturating_add` are
/// physics-scope builtins, but they are also exercised in mask expressions,
/// so they are extracted here for reuse by Tasks 3, 4, and 5.
///
/// # Panics
///
/// Panics with a survey-pointer message if argument counts are wrong.
pub(crate) fn eval_numeric_builtin(name: &str, args: &[EvalVal]) -> Option<EvalVal> {
    match name {
        "min" => {
            assert_eq!(args.len(), 2, "min expects 2 args");
            Some(EvalVal::Float(args[0].as_f32().min(args[1].as_f32())))
        }
        "max" => {
            assert_eq!(args.len(), 2, "max expects 2 args");
            Some(EvalVal::Float(args[0].as_f32().max(args[1].as_f32())))
        }
        "saturating_add" => {
            assert_eq!(args.len(), 2, "saturating_add expects 2 args");
            let a = args[0].as_f32() as u32;
            let b = args[1].as_f32() as u32;
            Some(EvalVal::Float(a.saturating_add(b) as f32))
        }
        "distance" => {
            assert_eq!(args.len(), 2, "distance expects 2 args");
            let a = args[0].as_vec3();
            let b = args[1].as_vec3();
            Some(EvalVal::Float(vec3_distance(a, b)))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Euclidean distance between two `Vec3` positions.
#[inline]
pub(crate) fn vec3_distance(a: Vec3, b: Vec3) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
