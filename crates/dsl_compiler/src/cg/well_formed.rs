//! `check_well_formed` — IR invariants pass.
//!
//! Every consumer of the Compute-Graph IR (HIR/MIR lowering, schedule
//! synthesis, emit) assumes the program it walks has no dangling ids,
//! no type-incoherent expressions, no schedule-blocking cycles in the
//! read/write graph, and conforms to the constitution's mutation
//! channels. The lowering passes that produce a [`CgProgram`] use the
//! [`CgProgramBuilder`] which catches the structural-id failures *at
//! insertion time*, but the broader semantic invariants (type checking,
//! P6 mutation channel, dispatch-shape compatibility, cycle freedom)
//! aren't checked there. This pass is the single gatekeeper that runs
//! all of those invariants in one walk and returns a structured error
//! list.
//!
//! The pass NEVER panics: it operates on potentially-malformed programs
//! (the whole point of well-formedness checking) and returns a typed
//! error variant for every defect rather than indexing past arena
//! bounds. P10.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 1.6, for the design rationale.
//!
//! # P6 — corrected reading
//!
//! The plan body wording for the P6-related check ("every `writes` field
//! on a non-event-fold op references an `EventRing`") was the wrong
//! reading of constitution P6. P6 says: agent state mutations flow
//! through events; only event-fold dispatchers may write fields
//! directly. The implementation enforces the corrected form:
//!
//! - Mask predicates / scoring / physics rules / spatial queries /
//!   plumbing may NOT contain a `CgStmt::Assign` whose target is a
//!   `DataHandle::AgentField` — they must `Emit` an event into an event
//!   ring instead.
//! - View folds MAY contain `AgentField` writes (folds are the
//!   permitted writers).
//!
//! See the commit message + the test `view_fold_may_write_agent_field`
//! for the inverse case.

use std::collections::{BTreeMap, BTreeSet};

use crate::cg::data_handle::{AgentRef, CgExprId, DataHandle};
use crate::cg::dispatch::DispatchShape;
use crate::cg::expr::{type_check, CgTy, TypeCheckCtx, TypeError};
use crate::cg::op::{ComputeOp, ComputeOpKind, OpId, ScoringRowOp, SpatialQueryKind};
use crate::cg::program::CgProgram;
use crate::cg::stmt::{CgStmt, CgStmtId, CgStmtListId};

// ---------------------------------------------------------------------------
// HandleConsistencyReason — typed reasons a DataHandle can be malformed.
// ---------------------------------------------------------------------------

/// Why a [`DataHandle`] failed the structural-consistency check. Each
/// variant carries the typed payload needed to point at the specific
/// reference that's broken — no `String` reasons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandleConsistencyReason {
    /// `DataHandle::AgentField { target: AgentRef::Target(expr_id), .. }`
    /// referenced an expression id past the end of the program's expr
    /// arena.
    AgentRefTargetExprOutOfRange {
        referenced: CgExprId,
        arena_len: u32,
    },
}

// ---------------------------------------------------------------------------
// CgError — typed error vocabulary for the well-formed pass.
// ---------------------------------------------------------------------------

/// Every defect [`check_well_formed`] can report. Every variant pins
/// the offending op/id with typed fields. No `String` reasons; the
/// `kind_label` / `shape_label` strings carried by `P6Violation` and
/// `KindShapeMismatch` are `&'static str` tags pulled from the small,
/// closed set of op kinds and dispatch shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CgError {
    /// An op references an [`OpId`] that's out of range. (Reserved for
    /// future cross-op references — today the only places `OpId`s are
    /// referenced are the `ops:` array's positions, which can't be
    /// out-of-range. Variant exists so downstream passes that *do*
    /// embed `OpId` in the IR have a place to report.)
    OpIdOutOfRange {
        op: OpId,
        referenced: OpId,
        arena_len: u32,
    },

    /// An expression-id referenced inside an op or statement is past
    /// the end of the program's expression arena.
    ExprIdOutOfRange {
        op: OpId,
        referenced: CgExprId,
        arena_len: u32,
    },

    /// A statement-list-id referenced by an op is past the end of the
    /// program's stmt-list arena.
    StmtListIdOutOfRange {
        op: OpId,
        referenced: CgStmtListId,
        arena_len: u32,
    },

    /// A statement-id referenced by a [`crate::cg::stmt::CgStmtList`] is
    /// past the end of the program's stmt arena. `list` names which
    /// list contained the dangling reference.
    StmtIdOutOfRange {
        list: CgStmtListId,
        referenced: CgStmtId,
        arena_len: u32,
    },

    /// A [`DataHandle`] is internally inconsistent. The structural form
    /// of the check today validates the embedded `CgExprId` in
    /// `AgentRef::Target(expr_id)`; future variants of this error
    /// variant accrete more checks here.
    DataHandleIdInconsistent {
        op: OpId,
        handle: DataHandle,
        reason: HandleConsistencyReason,
    },

    /// The IR's read/write graph contains a cycle (an SCC of size > 1
    /// where each member writes a handle the next member reads). The
    /// schedule synthesizer can't topologically sort programs with
    /// cycles. Self-edges (one op reading what it writes) are NOT
    /// reported as cycles — that's a legitimate event-fold pattern.
    Cycle { ops: Vec<OpId> },

    /// An expression's claimed [`CgTy`] doesn't match its operands'
    /// types. Wraps the existing [`TypeError`] vocabulary so the
    /// well-formed pass doesn't duplicate it.
    TypeMismatch { op: OpId, error: TypeError },

    /// A non-fold op writes an [`DataHandle::AgentField`]. Per
    /// constitution P6, agent-state mutations must flow through events
    /// (`Emit` into an `EventRing`); only [`ComputeOpKind::ViewFold`]
    /// dispatchers may write `AgentField` directly. `kind_label` is the
    /// op's [`ComputeOpKind::label_static`]; `write` is the offending
    /// handle.
    P6Violation {
        op: OpId,
        kind_label: &'static str,
        write: DataHandle,
    },

    /// A [`ComputeOpKind`] is paired with a [`DispatchShape`] not in
    /// its allowed set. Subsumes the Task 1.5 reviewer concern I-3
    /// (silent kind/shape mismatches at builder time).
    KindShapeMismatch {
        op: OpId,
        kind_label: &'static str,
        shape_label: &'static str,
    },
}

// ---------------------------------------------------------------------------
// DispatchShapeLabel — lightweight tag for kind/shape compatibility.
// ---------------------------------------------------------------------------

/// Lightweight identifier for a dispatch shape — strips the data each
/// shape carries (`source_ring`, `PerPairSource`) so the compatibility
/// table is a trivial enum-vs-enum match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchShapeLabel {
    PerAgent,
    PerEvent,
    PerPair,
    OneShot,
    PerWord,
}

impl DispatchShapeLabel {
    fn from_shape(shape: &DispatchShape) -> Self {
        match shape {
            DispatchShape::PerAgent => DispatchShapeLabel::PerAgent,
            DispatchShape::PerEvent { .. } => DispatchShapeLabel::PerEvent,
            DispatchShape::PerPair { .. } => DispatchShapeLabel::PerPair,
            DispatchShape::OneShot => DispatchShapeLabel::OneShot,
            DispatchShape::PerWord => DispatchShapeLabel::PerWord,
        }
    }

    fn snake(self) -> &'static str {
        match self {
            DispatchShapeLabel::PerAgent => "per_agent",
            DispatchShapeLabel::PerEvent => "per_event",
            DispatchShapeLabel::PerPair => "per_pair",
            DispatchShapeLabel::OneShot => "one_shot",
            DispatchShapeLabel::PerWord => "per_word",
        }
    }
}

/// Stable kind-label used in `P6Violation` and `KindShapeMismatch`.
/// Decoupled from [`ComputeOpKind::label`] (which is instance-specific
/// — includes ids in the rendered string); this returns one of the
/// closed set of kind tags.
fn kind_label(kind: &ComputeOpKind) -> &'static str {
    match kind {
        ComputeOpKind::MaskPredicate { .. } => "mask_predicate",
        ComputeOpKind::ScoringArgmax { .. } => "scoring_argmax",
        ComputeOpKind::PhysicsRule { .. } => "physics_rule",
        ComputeOpKind::ViewFold { .. } => "view_fold",
        ComputeOpKind::SpatialQuery { .. } => "spatial_query",
        ComputeOpKind::Plumbing { kind } => match *kind {},
    }
}

/// Allowed [`DispatchShapeLabel`]s for a given [`ComputeOpKind`].
/// `MaskPredicate` accepts `PerAgent` (the dominant case) and `PerPair`
/// (pair-masks driven by spatial neighborhoods). `ScoringArgmax` is
/// `PerAgent` only. `PhysicsRule`/`ViewFold` are `PerEvent` only.
/// `SpatialQuery`'s allowed shapes depend on the spatial-query kind:
/// `BuildHash` is `PerAgent` (every agent is hashed); `KinQuery` /
/// `EngagementQuery` are `PerAgent` (one query per agent — they write
/// into the per-agent query-results scratch).
fn allowed_shapes_for_kind(kind: &ComputeOpKind) -> &'static [DispatchShapeLabel] {
    match kind {
        ComputeOpKind::MaskPredicate { .. } => {
            &[DispatchShapeLabel::PerAgent, DispatchShapeLabel::PerPair]
        }
        ComputeOpKind::ScoringArgmax { .. } => &[DispatchShapeLabel::PerAgent],
        ComputeOpKind::PhysicsRule { .. } => &[DispatchShapeLabel::PerEvent],
        ComputeOpKind::ViewFold { .. } => &[DispatchShapeLabel::PerEvent],
        ComputeOpKind::SpatialQuery { kind } => match kind {
            // BuildHash hashes every agent into the grid — per-agent
            // dispatch.
            SpatialQueryKind::BuildHash => &[DispatchShapeLabel::PerAgent],
            // Kin/engagement queries run one walk per agent and write
            // the per-agent query-results scratch.
            SpatialQueryKind::KinQuery | SpatialQueryKind::EngagementQuery => {
                &[DispatchShapeLabel::PerAgent]
            }
        },
        ComputeOpKind::Plumbing { kind } => match *kind {},
    }
}

// ---------------------------------------------------------------------------
// check_well_formed
// ---------------------------------------------------------------------------

/// Run every well-formedness check on `prog`. Returns `Ok(())` if the
/// program passes every invariant; otherwise returns `Err(errors)` with
/// every defect found. The pass collects all errors (it does NOT
/// short-circuit on first); downstream tooling wants the full list.
///
/// The pass NEVER panics — even on programs with corrupted arenas,
/// dangling ids, malformed handles, etc. Indexing the program's arenas
/// uses `.get(...)`-style range checks and reports each out-of-range
/// reference as a typed `CgError` variant.
///
/// Checks performed (in order — but each contributes errors
/// independently to the output list):
///
/// 1. Arena id-range checks for every `OpId`, `CgExprId`, `CgStmtListId`,
///    `CgStmtId` reference embedded in the program's ops, statements,
///    and lists.
/// 2. `DataHandle` structural consistency (every embedded `CgExprId`
///    inside an `AgentRef::Target` resolves).
/// 3. Type checking of every expression referenced by an op (delegates
///    to [`type_check`]).
/// 4. P6 mutation channel: only `ViewFold` ops may contain a
///    `CgStmt::Assign` to a `DataHandle::AgentField`.
/// 5. Kind/shape compatibility: each `ComputeOpKind` only pairs with
///    its declared set of `DispatchShape`s.
/// 6. Cycle detection in the read/write graph.
pub fn check_well_formed(prog: &CgProgram) -> Result<(), Vec<CgError>> {
    let mut errors: Vec<CgError> = Vec::new();

    let expr_arena_len = prog.exprs.len() as u32;
    let stmt_arena_len = prog.stmts.len() as u32;
    let list_arena_len = prog.stmt_lists.len() as u32;

    // -----------------------------------------------------------------
    // Pass A: per-list stmt-id range checks. Run before per-op walks
    // so that each op's list-walks see lists with already-validated
    // contents. Errors here do NOT prevent later passes from running;
    // each pass guards its own indexing.
    // -----------------------------------------------------------------
    for (list_index, list) in prog.stmt_lists.iter().enumerate() {
        let list_id = CgStmtListId(list_index as u32);
        for stmt_id in &list.stmts {
            if stmt_id.0 >= stmt_arena_len {
                errors.push(CgError::StmtIdOutOfRange {
                    list: list_id,
                    referenced: *stmt_id,
                    arena_len: stmt_arena_len,
                });
            }
        }
    }

    // -----------------------------------------------------------------
    // Pass B: per-op walks. Each op contributes error-set entries for
    // every check that fires.
    // -----------------------------------------------------------------
    for (op_index, op) in prog.ops.iter().enumerate() {
        let op_id = OpId(op_index as u32);
        check_op(
            op,
            op_id,
            prog,
            expr_arena_len,
            stmt_arena_len,
            list_arena_len,
            &mut errors,
        );
    }

    // -----------------------------------------------------------------
    // Pass C: cycle detection. Build the read/write graph from the
    // per-op `reads` / `writes` fields — these may include
    // record_read/record_write-injected handles, which is intentional
    // (cycle freedom must hold over the *full* dependency set, not
    // just the auto-walker portion).
    // -----------------------------------------------------------------
    detect_cycles(prog, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Per-op checks — id-range, handle consistency, type checking, P6, and
/// kind/shape compatibility.
fn check_op(
    op: &ComputeOp,
    op_id: OpId,
    prog: &CgProgram,
    expr_arena_len: u32,
    stmt_arena_len: u32,
    list_arena_len: u32,
    errors: &mut Vec<CgError>,
) {
    // --- id-range checks per kind ------------------------------------

    match &op.kind {
        ComputeOpKind::MaskPredicate { predicate, .. } => {
            if predicate.0 >= expr_arena_len {
                errors.push(CgError::ExprIdOutOfRange {
                    op: op_id,
                    referenced: *predicate,
                    arena_len: expr_arena_len,
                });
            }
        }
        ComputeOpKind::ScoringArgmax { rows, .. } => {
            for row in rows {
                if row.utility.0 >= expr_arena_len {
                    errors.push(CgError::ExprIdOutOfRange {
                        op: op_id,
                        referenced: row.utility,
                        arena_len: expr_arena_len,
                    });
                }
                if row.target.0 >= expr_arena_len {
                    errors.push(CgError::ExprIdOutOfRange {
                        op: op_id,
                        referenced: row.target,
                        arena_len: expr_arena_len,
                    });
                }
            }
        }
        ComputeOpKind::PhysicsRule { body, .. } | ComputeOpKind::ViewFold { body, .. } => {
            if body.0 >= list_arena_len {
                errors.push(CgError::StmtListIdOutOfRange {
                    op: op_id,
                    referenced: *body,
                    arena_len: list_arena_len,
                });
            } else {
                // Body resolves; walk it for inner expr/list/stmt
                // references.
                walk_list_id_ranges(
                    *body,
                    op_id,
                    prog,
                    expr_arena_len,
                    stmt_arena_len,
                    list_arena_len,
                    errors,
                );
            }
        }
        ComputeOpKind::SpatialQuery { .. } => {
            // No id refs to check.
        }
        ComputeOpKind::Plumbing { kind } => {
            // Uninhabited — no instance reachable.
            match *kind {}
        }
    }

    // --- DataHandle structural consistency on reads + writes ---------

    for handle in op.reads.iter().chain(op.writes.iter()) {
        if let Some(reason) = check_data_handle_consistency(handle, expr_arena_len) {
            errors.push(CgError::DataHandleIdInconsistent {
                op: op_id,
                handle: handle.clone(),
                reason,
            });
        }
    }

    // --- Type checking on embedded expressions -----------------------

    let ctx = TypeCheckCtx::new(prog);
    type_check_op(op, op_id, prog, &ctx, expr_arena_len, errors);

    // --- P6 mutation channel: AgentField writes only in ViewFold ----

    p6_check_op(op, op_id, prog, errors);

    // --- Kind/shape compatibility ------------------------------------

    let allowed = allowed_shapes_for_kind(&op.kind);
    let label = DispatchShapeLabel::from_shape(&op.shape);
    if !allowed.contains(&label) {
        errors.push(CgError::KindShapeMismatch {
            op: op_id,
            kind_label: kind_label(&op.kind),
            shape_label: label.snake(),
        });
    }
}

/// Walk a `CgStmtListId` recursively for id-range defects in its
/// statements + nested lists. The list-id itself has already been
/// range-checked by the caller.
fn walk_list_id_ranges(
    list_id: CgStmtListId,
    op_id: OpId,
    prog: &CgProgram,
    expr_arena_len: u32,
    stmt_arena_len: u32,
    list_arena_len: u32,
    errors: &mut Vec<CgError>,
) {
    let Some(list) = prog.stmt_lists.get(list_id.0 as usize) else {
        // Defensive — caller checked range already, but if ever called
        // out-of-range we don't panic.
        return;
    };
    for stmt_id in &list.stmts {
        if stmt_id.0 >= stmt_arena_len {
            // Already reported in Pass A; skip the recursion to avoid
            // indexing past the arena.
            continue;
        }
        let Some(stmt) = prog.stmts.get(stmt_id.0 as usize) else {
            continue;
        };
        match stmt {
            CgStmt::Assign { value, target } => {
                if value.0 >= expr_arena_len {
                    errors.push(CgError::ExprIdOutOfRange {
                        op: op_id,
                        referenced: *value,
                        arena_len: expr_arena_len,
                    });
                }
                // Target's structural consistency is covered by the
                // op-level reads/writes walk; checking again here
                // would double-report.
                let _ = target;
            }
            CgStmt::Emit { fields, .. } => {
                for (_, expr_id) in fields {
                    if expr_id.0 >= expr_arena_len {
                        errors.push(CgError::ExprIdOutOfRange {
                            op: op_id,
                            referenced: *expr_id,
                            arena_len: expr_arena_len,
                        });
                    }
                }
            }
            CgStmt::If { cond, then, else_ } => {
                if cond.0 >= expr_arena_len {
                    errors.push(CgError::ExprIdOutOfRange {
                        op: op_id,
                        referenced: *cond,
                        arena_len: expr_arena_len,
                    });
                }
                if then.0 >= list_arena_len {
                    errors.push(CgError::StmtListIdOutOfRange {
                        op: op_id,
                        referenced: *then,
                        arena_len: list_arena_len,
                    });
                } else {
                    walk_list_id_ranges(
                        *then,
                        op_id,
                        prog,
                        expr_arena_len,
                        stmt_arena_len,
                        list_arena_len,
                        errors,
                    );
                }
                if let Some(else_id) = else_ {
                    if else_id.0 >= list_arena_len {
                        errors.push(CgError::StmtListIdOutOfRange {
                            op: op_id,
                            referenced: *else_id,
                            arena_len: list_arena_len,
                        });
                    } else {
                        walk_list_id_ranges(
                            *else_id,
                            op_id,
                            prog,
                            expr_arena_len,
                            stmt_arena_len,
                            list_arena_len,
                            errors,
                        );
                    }
                }
            }
        }
    }
}

/// Structural consistency of a [`DataHandle`]: every embedded
/// [`CgExprId`] resolves into the program's expression arena. Returns
/// `Some(reason)` if the handle has an out-of-range embedded id.
///
/// Future work: a registry-aware variant that cross-checks `MaskId` /
/// `ViewId` / `EventRingId` etc. against a lowering-populated registry.
/// Today the IR has no such registry as a hard contract — see the
/// module-level doc on the structural-only choice.
fn check_data_handle_consistency(
    handle: &DataHandle,
    expr_arena_len: u32,
) -> Option<HandleConsistencyReason> {
    match handle {
        DataHandle::AgentField { target, .. } => match target {
            AgentRef::Target(expr_id) => {
                if expr_id.0 >= expr_arena_len {
                    Some(HandleConsistencyReason::AgentRefTargetExprOutOfRange {
                        referenced: *expr_id,
                        arena_len: expr_arena_len,
                    })
                } else {
                    None
                }
            }
            AgentRef::Self_ | AgentRef::Actor | AgentRef::EventTarget => None,
        },
        DataHandle::ViewStorage { .. }
        | DataHandle::EventRing { .. }
        | DataHandle::ConfigConst { .. }
        | DataHandle::MaskBitmap { .. }
        | DataHandle::ScoringOutput
        | DataHandle::SpatialStorage { .. }
        | DataHandle::Rng { .. } => None,
    }
}

/// Type-check every expression embedded in the op's payload. Each
/// dispatch hands the offending [`CgExprId`] to [`type_check`] and
/// converts a [`TypeError`] into a [`CgError::TypeMismatch`] tagged
/// with the op id.
///
/// Skips ids that are already known to be out-of-range — those have
/// been reported by the id-range pass and re-resolving them here would
/// either index past the arena (which we never do) or duplicate the
/// report.
fn type_check_op(
    op: &ComputeOp,
    op_id: OpId,
    prog: &CgProgram,
    ctx: &TypeCheckCtx<'_>,
    expr_arena_len: u32,
    errors: &mut Vec<CgError>,
) {
    match &op.kind {
        ComputeOpKind::MaskPredicate { predicate, .. } => {
            if predicate.0 < expr_arena_len {
                let expr = &prog.exprs[predicate.0 as usize];
                match type_check(expr, *predicate, ctx) {
                    Ok(ty) => {
                        if ty != CgTy::Bool {
                            errors.push(CgError::TypeMismatch {
                                op: op_id,
                                error: TypeError::ClaimedResultMismatch {
                                    node: *predicate,
                                    expected: CgTy::Bool,
                                    got: ty,
                                },
                            });
                        }
                    }
                    Err(err) => {
                        errors.push(CgError::TypeMismatch {
                            op: op_id,
                            error: err,
                        });
                    }
                }
            }
        }
        ComputeOpKind::ScoringArgmax { rows, .. } => {
            for row in rows {
                type_check_row(row, op_id, prog, ctx, expr_arena_len, errors);
            }
        }
        ComputeOpKind::PhysicsRule { body, .. } | ComputeOpKind::ViewFold { body, .. } => {
            // Walk the body; type-check every expression we encounter.
            // The body's stmt-list-id has been range-checked by the
            // caller's id-range pass; we still guard here to never
            // panic.
            type_check_list(*body, op_id, prog, ctx, expr_arena_len, errors);
        }
        ComputeOpKind::SpatialQuery { .. } => {
            // No expressions to type-check — kernel is built-in.
        }
        ComputeOpKind::Plumbing { kind } => match *kind {},
    }
}

fn type_check_row(
    row: &ScoringRowOp,
    op_id: OpId,
    prog: &CgProgram,
    ctx: &TypeCheckCtx<'_>,
    expr_arena_len: u32,
    errors: &mut Vec<CgError>,
) {
    if row.utility.0 < expr_arena_len {
        let expr = &prog.exprs[row.utility.0 as usize];
        if let Err(err) = type_check(expr, row.utility, ctx) {
            errors.push(CgError::TypeMismatch {
                op: op_id,
                error: err,
            });
        }
    }
    if row.target.0 < expr_arena_len {
        let expr = &prog.exprs[row.target.0 as usize];
        if let Err(err) = type_check(expr, row.target, ctx) {
            errors.push(CgError::TypeMismatch {
                op: op_id,
                error: err,
            });
        }
    }
}

fn type_check_list(
    list_id: CgStmtListId,
    op_id: OpId,
    prog: &CgProgram,
    ctx: &TypeCheckCtx<'_>,
    expr_arena_len: u32,
    errors: &mut Vec<CgError>,
) {
    let Some(list) = prog.stmt_lists.get(list_id.0 as usize) else {
        return;
    };
    for stmt_id in &list.stmts {
        let Some(stmt) = prog.stmts.get(stmt_id.0 as usize) else {
            continue;
        };
        match stmt {
            CgStmt::Assign { value, .. } => {
                if value.0 < expr_arena_len {
                    let expr = &prog.exprs[value.0 as usize];
                    if let Err(err) = type_check(expr, *value, ctx) {
                        errors.push(CgError::TypeMismatch {
                            op: op_id,
                            error: err,
                        });
                    }
                }
            }
            CgStmt::Emit { fields, .. } => {
                for (_, expr_id) in fields {
                    if expr_id.0 < expr_arena_len {
                        let expr = &prog.exprs[expr_id.0 as usize];
                        if let Err(err) = type_check(expr, *expr_id, ctx) {
                            errors.push(CgError::TypeMismatch {
                                op: op_id,
                                error: err,
                            });
                        }
                    }
                }
            }
            CgStmt::If { cond, then, else_ } => {
                if cond.0 < expr_arena_len {
                    let expr = &prog.exprs[cond.0 as usize];
                    match type_check(expr, *cond, ctx) {
                        Ok(ty) => {
                            if ty != CgTy::Bool {
                                errors.push(CgError::TypeMismatch {
                                    op: op_id,
                                    error: TypeError::ClaimedResultMismatch {
                                        node: *cond,
                                        expected: CgTy::Bool,
                                        got: ty,
                                    },
                                });
                            }
                        }
                        Err(err) => errors.push(CgError::TypeMismatch {
                            op: op_id,
                            error: err,
                        }),
                    }
                }
                type_check_list(*then, op_id, prog, ctx, expr_arena_len, errors);
                if let Some(else_id) = else_ {
                    type_check_list(*else_id, op_id, prog, ctx, expr_arena_len, errors);
                }
            }
        }
    }
}

/// P6 (corrected reading): only [`ComputeOpKind::ViewFold`] ops may
/// contain an `Assign` whose target is a [`DataHandle::AgentField`].
/// Mask predicates / scoring argmax / physics rules / spatial queries
/// must use `Emit` to write through events.
fn p6_check_op(op: &ComputeOp, op_id: OpId, prog: &CgProgram, errors: &mut Vec<CgError>) {
    let allow_agent_field_writes = matches!(op.kind, ComputeOpKind::ViewFold { .. });
    if allow_agent_field_writes {
        return;
    }
    // Walk the body (if any) for `Assign { target: AgentField, .. }`.
    match &op.kind {
        ComputeOpKind::PhysicsRule { body, .. } => {
            p6_walk_list(
                *body,
                op_id,
                kind_label(&op.kind),
                prog,
                errors,
            );
        }
        // Non-body kinds: nothing to walk. (Mask predicates and
        // scoring rows are pure expressions; spatial queries have no
        // user-authored body.)
        ComputeOpKind::MaskPredicate { .. }
        | ComputeOpKind::ScoringArgmax { .. }
        | ComputeOpKind::SpatialQuery { .. } => {}
        // Plumbing is uninhabited; ViewFold returns above.
        ComputeOpKind::ViewFold { .. } => {}
        ComputeOpKind::Plumbing { kind } => match *kind {},
    }
}

fn p6_walk_list(
    list_id: CgStmtListId,
    op_id: OpId,
    kind_label: &'static str,
    prog: &CgProgram,
    errors: &mut Vec<CgError>,
) {
    let Some(list) = prog.stmt_lists.get(list_id.0 as usize) else {
        return;
    };
    for stmt_id in &list.stmts {
        let Some(stmt) = prog.stmts.get(stmt_id.0 as usize) else {
            continue;
        };
        match stmt {
            CgStmt::Assign { target, .. } => {
                if matches!(target, DataHandle::AgentField { .. }) {
                    errors.push(CgError::P6Violation {
                        op: op_id,
                        kind_label,
                        write: target.clone(),
                    });
                }
            }
            CgStmt::Emit { .. } => {
                // Allowed — `Emit` is the P6 mutation channel.
            }
            CgStmt::If { then, else_, .. } => {
                p6_walk_list(*then, op_id, kind_label, prog, errors);
                if let Some(else_id) = else_ {
                    p6_walk_list(*else_id, op_id, kind_label, prog, errors);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cycle detection
// ---------------------------------------------------------------------------

/// Build the read/write graph and detect cycles. Edges are
/// `producer → consumer` for each handle that op `producer` writes and
/// op `consumer` reads. Self-edges (one op reading what it writes) do
/// NOT count as cycles — that's a legitimate event-fold pattern (read
/// prior tick's storage, write next tick's).
///
/// Implementation: build a `HashMap<DataHandle, Vec<OpId>>` of writers
/// per handle; for each op's reads, add edges from each writer of that
/// handle to this op. Then run a DFS-based SCC finder; report any SCC
/// of size > 1.
fn detect_cycles(prog: &CgProgram, errors: &mut Vec<CgError>) {
    let n = prog.ops.len();
    if n == 0 {
        return;
    }

    // Map handle → writers.
    let mut writers: BTreeMap<HandleKey, Vec<OpId>> = BTreeMap::new();
    for (op_index, op) in prog.ops.iter().enumerate() {
        let id = OpId(op_index as u32);
        for w in &op.writes {
            writers
                .entry(HandleKey::from_handle(w))
                .or_default()
                .push(id);
        }
    }

    // adjacency: out-edges per op as a sorted set of `u32` indices
    // (OpId doesn't implement Ord, so store the inner u32 directly).
    let mut adj: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
    for (op_index, op) in prog.ops.iter().enumerate() {
        let consumer = OpId(op_index as u32);
        for r in &op.reads {
            let key = HandleKey::from_handle(r);
            if let Some(producers) = writers.get(&key) {
                for &producer in producers {
                    if producer == consumer {
                        // Self-edge — not a cycle (event-fold pattern).
                        continue;
                    }
                    adj[producer.0 as usize].insert(consumer.0);
                }
            }
        }
    }

    // Tarjan's SCC. Iterative form so deep graphs don't blow the
    // stack — we never panic.
    let sccs = tarjan_scc(&adj);
    for scc in sccs {
        if scc.len() > 1 {
            // SCC is yielded in arbitrary order from Tarjan; sort by
            // op index for deterministic error output.
            let mut ops = scc;
            ops.sort_by_key(|o| o.0);
            errors.push(CgError::Cycle { ops });
        }
    }
}

/// Strict-ordered key for [`DataHandle`] — `DataHandle` is `Eq + Hash`
/// but not `Ord`. We project into a small `(u32, ...)` tuple via
/// serialization. JSON serialization is overkill; build a lightweight
/// canonical bytestring instead.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HandleKey(String);

impl HandleKey {
    fn from_handle(handle: &DataHandle) -> Self {
        // Use the handle's `Display` form — every variant prints a
        // distinct, structurally complete string (the Display impls
        // were tested in Task 1.1 to be unique per variant).
        // `AgentRef::Target(expr_id)` includes the expr id, so two
        // handles with different targets produce different keys.
        Self(format!("{}", handle))
    }
}

/// Tarjan's strongly-connected-components algorithm. Iterative
/// implementation; never panics. Returns each SCC as a `Vec<OpId>`.
///
/// Adapted from the canonical recursive form to use an explicit stack —
/// the recursion depth on a linear chain of n ops is n, which would
/// blow the default 8 MB stack at ~32K ops. Iterative is robust.
fn tarjan_scc(adj: &[BTreeSet<u32>]) -> Vec<Vec<OpId>> {
    let n = adj.len();
    let mut indices: Vec<i64> = vec![-1; n];
    let mut lowlinks: Vec<i64> = vec![0; n];
    let mut on_stack: Vec<bool> = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut sccs: Vec<Vec<OpId>> = Vec::new();
    let mut index_counter: i64 = 0;

    // Iterative DFS frame: (node, iterator into out-edges as a
    // collected Vec).
    struct Frame {
        node: usize,
        edges: Vec<usize>,
        next_edge: usize,
    }

    for start in 0..n {
        if indices[start] != -1 {
            continue;
        }
        let mut call_stack: Vec<Frame> = Vec::new();
        // initial visit
        indices[start] = index_counter;
        lowlinks[start] = index_counter;
        index_counter += 1;
        stack.push(start);
        on_stack[start] = true;
        let edges: Vec<usize> = adj[start].iter().map(|o| *o as usize).collect();
        call_stack.push(Frame {
            node: start,
            edges,
            next_edge: 0,
        });

        while let Some(frame) = call_stack.last_mut() {
            if frame.next_edge < frame.edges.len() {
                let w = frame.edges[frame.next_edge];
                frame.next_edge += 1;
                if w >= n {
                    // Defensive — adjacency cannot reference an out-of-
                    // range op (we built it from in-range OpIds), but
                    // never panic.
                    continue;
                }
                if indices[w] == -1 {
                    // Recurse into w.
                    indices[w] = index_counter;
                    lowlinks[w] = index_counter;
                    index_counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    let edges_w: Vec<usize> = adj[w].iter().map(|o| *o as usize).collect();
                    call_stack.push(Frame {
                        node: w,
                        edges: edges_w,
                        next_edge: 0,
                    });
                    continue;
                } else if on_stack[w] {
                    // Back-edge — update lowlink of the current node.
                    if indices[w] < lowlinks[frame.node] {
                        lowlinks[frame.node] = indices[w];
                    }
                }
            } else {
                // Done with this node — possibly close an SCC.
                let v = frame.node;
                if lowlinks[v] == indices[v] {
                    let mut scc = Vec::new();
                    while let Some(w) = stack.pop() {
                        on_stack[w] = false;
                        scc.push(OpId(w as u32));
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(scc);
                }
                // Pop the frame, propagate lowlink upward.
                call_stack.pop();
                if let Some(parent) = call_stack.last_mut() {
                    if lowlinks[v] < lowlinks[parent.node] {
                        lowlinks[parent.node] = lowlinks[v];
                    }
                }
            }
        }
    }

    sccs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cg::data_handle::{
        AgentFieldId, AgentRef, EventRingId, MaskId, ViewId, ViewStorageSlot,
    };
    use crate::cg::expr::{BinaryOp, CgExpr, CgTy, LitValue};
    use crate::cg::op::{
        ActionId, ComputeOp, EventKindId, PhysicsRuleId, ScoringId, ScoringRowOp, Span,
    };
    use crate::cg::program::{CgProgram, CgProgramBuilder};
    use crate::cg::stmt::{CgStmt, CgStmtId, CgStmtList, EventField};

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    fn read_self_hp() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        })
    }

    fn lit_f32(v: f32) -> CgExpr {
        CgExpr::Lit(LitValue::F32(v))
    }

    fn lit_bool(b: bool) -> CgExpr {
        CgExpr::Lit(LitValue::Bool(b))
    }

    /// Build a program containing a well-formed mask predicate
    /// `hp < 0.5`. Returned for reuse + as the happy-path fixture.
    fn happy_mask_program() -> CgProgram {
        let mut b = CgProgramBuilder::new();
        let hp = b.add_expr(read_self_hp()).unwrap();
        let half = b.add_expr(lit_f32(0.5)).unwrap();
        let pred = b
            .add_expr(CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: half,
                ty: CgTy::Bool,
            })
            .unwrap();
        b.add_op(
            ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: pred,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap();
        b.finish()
    }

    // -----------------------------------------------------------------
    // 1. Happy path
    // -----------------------------------------------------------------

    #[test]
    fn happy_path_single_mask_predicate() {
        let prog = happy_mask_program();
        assert_eq!(check_well_formed(&prog), Ok(()));
    }

    #[test]
    fn happy_path_multi_op_program() {
        // Build a small program with one of each top-level kind that's
        // exercisable today: a mask predicate, a scoring argmax, a
        // physics rule, a view fold, a spatial query.
        let mut b = CgProgramBuilder::new();

        // --- mask: hp < 0.5 ---
        let hp = b.add_expr(read_self_hp()).unwrap();
        let half = b.add_expr(lit_f32(0.5)).unwrap();
        let pred = b
            .add_expr(CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: half,
                ty: CgTy::Bool,
            })
            .unwrap();
        b.add_op(
            ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: pred,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap();

        // --- scoring: utility = 1.0, target = AgentId(0) ---
        let util = b.add_expr(lit_f32(1.0)).unwrap();
        let tgt = b.add_expr(CgExpr::Lit(LitValue::AgentId(0))).unwrap();
        b.add_op(
            ComputeOpKind::ScoringArgmax {
                scoring: ScoringId(0),
                rows: vec![ScoringRowOp {
                    action: ActionId(0),
                    utility: util,
                    target: tgt,
                }],
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap();

        // --- physics rule: emit-only body (P6-compliant) ---
        let physics_emit_payload = b.add_expr(lit_f32(0.0)).unwrap();
        let physics_emit = b
            .add_stmt(CgStmt::Emit {
                event: EventKindId(7),
                fields: vec![(
                    EventField {
                        event: EventKindId(7),
                        index: 0,
                    },
                    physics_emit_payload,
                )],
            })
            .unwrap();
        let physics_body = b.add_stmt_list(CgStmtList::new(vec![physics_emit])).unwrap();
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(7),
                body: physics_body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(7),
            },
            Span::dummy(),
        )
        .unwrap();

        // --- view fold: assigns view storage (allowed) ---
        let view_value = b.add_expr(lit_f32(2.0)).unwrap();
        let view_assign = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::ViewStorage {
                    view: ViewId(0),
                    slot: ViewStorageSlot::Primary,
                },
                value: view_value,
            })
            .unwrap();
        let view_body = b.add_stmt_list(CgStmtList::new(vec![view_assign])).unwrap();
        b.add_op(
            ComputeOpKind::ViewFold {
                view: ViewId(0),
                on_event: EventKindId(7),
                body: view_body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(7),
            },
            Span::dummy(),
        )
        .unwrap();

        // --- spatial query (BuildHash) ---
        b.add_op(
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::BuildHash,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap();

        let prog = b.finish();
        assert_eq!(check_well_formed(&prog), Ok(()));
    }

    // -----------------------------------------------------------------
    // 2. Out-of-range expr id
    // -----------------------------------------------------------------

    #[test]
    fn out_of_range_predicate_expr_id_reports_expr_id_out_of_range() {
        // Bypass the builder — the builder rejects this. We're testing
        // the post-construction defense.
        let mut prog = CgProgram::new();
        // No exprs in the arena. Construct a ComputeOp that references
        // expr#99.
        let op = ComputeOp {
            id: OpId(0),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: CgExprId(99),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(0) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        };
        prog.ops.push(op);
        let errs = check_well_formed(&prog).expect_err("dangling expr ref");
        assert!(
            errs.contains(&CgError::ExprIdOutOfRange {
                op: OpId(0),
                referenced: CgExprId(99),
                arena_len: 0,
            }),
            "missing ExprIdOutOfRange in {:?}",
            errs
        );
    }

    // -----------------------------------------------------------------
    // 3. Out-of-range stmt-list id
    // -----------------------------------------------------------------

    #[test]
    fn out_of_range_physics_body_list_id_reports_stmt_list_out_of_range() {
        // Bypass builder.
        let mut prog = CgProgram::new();
        let op = ComputeOp {
            id: OpId(0),
            kind: ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body: CgStmtListId(99),
            },
            reads: vec![],
            writes: vec![],
            shape: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            span: Span::dummy(),
        };
        prog.ops.push(op);
        let errs = check_well_formed(&prog).expect_err("dangling list ref");
        assert!(
            errs.contains(&CgError::StmtListIdOutOfRange {
                op: OpId(0),
                referenced: CgStmtListId(99),
                arena_len: 0,
            }),
            "missing StmtListIdOutOfRange in {:?}",
            errs
        );
    }

    // -----------------------------------------------------------------
    // 4. Out-of-range stmt id (referenced from a list)
    // -----------------------------------------------------------------

    #[test]
    fn out_of_range_stmt_id_in_list_reports_stmt_id_out_of_range() {
        // Build via builder to get a valid op pointing at a list, then
        // poke the list to reference a dangling stmt id.
        let mut b = CgProgramBuilder::new();
        let body = b.add_stmt_list(CgStmtList::new(vec![])).unwrap();
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap();
        let mut prog = b.finish();
        // Bypass: poke the list directly with a dangling stmt id.
        prog.stmt_lists[0].stmts.push(CgStmtId(99));

        let errs = check_well_formed(&prog).expect_err("dangling stmt ref");
        assert!(
            errs.contains(&CgError::StmtIdOutOfRange {
                list: CgStmtListId(0),
                referenced: CgStmtId(99),
                arena_len: 0,
            }),
            "missing StmtIdOutOfRange in {:?}",
            errs
        );
    }

    // -----------------------------------------------------------------
    // 5. AgentRef::Target dangling expr id (DataHandle consistency)
    // -----------------------------------------------------------------

    #[test]
    fn agent_ref_target_dangling_expr_reports_data_handle_inconsistent() {
        // Build a program with a valid op, then poke an
        // AgentRef::Target(CgExprId(99)) into its reads.
        let prog = happy_mask_program();
        let mut prog = prog;
        let dangling = DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Target(CgExprId(99)),
        };
        // arena has 3 exprs (hp, lit, binary), so #99 is dangling.
        let arena_len = prog.exprs.len() as u32;
        prog.ops[0].reads.push(dangling.clone());

        let errs = check_well_formed(&prog).expect_err("dangling AgentRef target");
        assert!(
            errs.contains(&CgError::DataHandleIdInconsistent {
                op: OpId(0),
                handle: dangling,
                reason: HandleConsistencyReason::AgentRefTargetExprOutOfRange {
                    referenced: CgExprId(99),
                    arena_len,
                },
            }),
            "missing DataHandleIdInconsistent in {:?}",
            errs
        );
    }

    // -----------------------------------------------------------------
    // 6. Type mismatch — MaskPredicate with non-Bool predicate
    // -----------------------------------------------------------------

    #[test]
    fn mask_predicate_non_bool_reports_type_mismatch() {
        // Predicate is `hp` (F32) instead of a Bool expression.
        let mut b = CgProgramBuilder::new();
        let hp = b.add_expr(read_self_hp()).unwrap(); // F32-typed read
        b.add_op(
            ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: hp,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap();
        let prog = b.finish();

        let errs = check_well_formed(&prog).expect_err("non-bool mask predicate");
        let saw_type_error = errs.iter().any(|e| {
            matches!(
                e,
                CgError::TypeMismatch {
                    op: OpId(0),
                    error: TypeError::ClaimedResultMismatch {
                        node: _,
                        expected: CgTy::Bool,
                        got: CgTy::F32,
                    },
                }
            )
        });
        assert!(saw_type_error, "missing Bool-typed TypeMismatch in {:?}", errs);
    }

    // -----------------------------------------------------------------
    // 7. Cycle detection
    // -----------------------------------------------------------------

    #[test]
    fn cycle_two_ops_writing_each_others_masks() {
        // Op#0 writes mask(3), reads mask(5). Op#1 writes mask(5),
        // reads mask(3). The graph forms 0 → 1 → 0 (cycle).
        //
        // Bypass the builder to inject the cross-mask reads (the
        // auto-walker doesn't synthesize them — they're emit-time
        // dependencies that lowering would record_read).
        //
        // Build two valid mask predicates first via the builder, then
        // splice reads into each.
        let mut b = CgProgramBuilder::new();
        let hp = b.add_expr(read_self_hp()).unwrap();
        let half = b.add_expr(lit_f32(0.5)).unwrap();
        let pred = b
            .add_expr(CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: half,
                ty: CgTy::Bool,
            })
            .unwrap();
        b.add_op(
            ComputeOpKind::MaskPredicate {
                mask: MaskId(3),
                predicate: pred,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap();
        b.add_op(
            ComputeOpKind::MaskPredicate {
                mask: MaskId(5),
                predicate: pred,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap();
        let mut prog = b.finish();

        // Inject cross-mask reads via record_read.
        prog.ops[0].record_read(DataHandle::MaskBitmap { mask: MaskId(5) });
        prog.ops[1].record_read(DataHandle::MaskBitmap { mask: MaskId(3) });

        let errs = check_well_formed(&prog).expect_err("cycle expected");
        let saw_cycle = errs.iter().any(|e| match e {
            CgError::Cycle { ops } => {
                ops.contains(&OpId(0)) && ops.contains(&OpId(1)) && ops.len() == 2
            }
            _ => false,
        });
        assert!(saw_cycle, "missing 2-op Cycle in {:?}", errs);
    }

    #[test]
    fn self_edge_is_not_a_cycle() {
        // An op that reads what it writes (event-fold pattern) should
        // NOT be reported as a cycle.
        //
        // Build a view fold that writes view-primary and reads
        // view-primary in its body's expression.
        let mut b = CgProgramBuilder::new();
        let read_prior = b
            .add_expr(CgExpr::Read(DataHandle::ViewStorage {
                view: ViewId(0),
                slot: ViewStorageSlot::Primary,
            }))
            .unwrap();
        let assign = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::ViewStorage {
                    view: ViewId(0),
                    slot: ViewStorageSlot::Primary,
                },
                value: read_prior,
            })
            .unwrap();
        let body = b.add_stmt_list(CgStmtList::new(vec![assign])).unwrap();
        b.add_op(
            ComputeOpKind::ViewFold {
                view: ViewId(0),
                on_event: EventKindId(0),
                body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap();
        let prog = b.finish();
        // Single op writes + reads view[0].primary — that's a self-edge,
        // not a cycle. check_well_formed should pass.
        assert_eq!(check_well_formed(&prog), Ok(()));
    }

    // -----------------------------------------------------------------
    // 8. P6 violation — physics rule writing AgentField directly
    // -----------------------------------------------------------------

    #[test]
    fn physics_rule_assigning_agent_field_reports_p6_violation() {
        // Build a physics rule whose body assigns hp directly. This is
        // the P6-corrected violation: only ViewFold may write
        // AgentField; physics rules must Emit instead.
        let mut b = CgProgramBuilder::new();
        let new_hp = b.add_expr(lit_f32(0.0)).unwrap();
        let assign = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: new_hp,
            })
            .unwrap();
        let body = b.add_stmt_list(CgStmtList::new(vec![assign])).unwrap();
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap();
        let prog = b.finish();

        let errs = check_well_formed(&prog).expect_err("P6 violation expected");
        let agent_field_handle = DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        };
        assert!(
            errs.contains(&CgError::P6Violation {
                op: OpId(0),
                kind_label: "physics_rule",
                write: agent_field_handle,
            }),
            "missing P6Violation(physics_rule) in {:?}",
            errs
        );
    }

    #[test]
    fn view_fold_may_write_agent_field() {
        // The inverse: a view-fold body writing AgentField is allowed.
        // (Folds are the permitted writers under the corrected P6.)
        let mut b = CgProgramBuilder::new();
        let new_hp = b.add_expr(lit_f32(0.0)).unwrap();
        let assign = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: new_hp,
            })
            .unwrap();
        let body = b.add_stmt_list(CgStmtList::new(vec![assign])).unwrap();
        b.add_op(
            ComputeOpKind::ViewFold {
                view: ViewId(0),
                on_event: EventKindId(0),
                body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap();
        let prog = b.finish();
        assert_eq!(check_well_formed(&prog), Ok(()));
    }

    // -----------------------------------------------------------------
    // 9. Kind/shape mismatch — MaskPredicate + PerEvent
    // -----------------------------------------------------------------

    #[test]
    fn mask_predicate_with_per_event_shape_reports_kind_shape_mismatch() {
        // A mask predicate paired with PerEvent dispatch is structurally
        // wrong — masks are per-agent (or per-pair) only.
        // Bypass the builder (the builder doesn't currently catch this,
        // which is precisely the I-3 reviewer concern this check
        // subsumes).
        let mut prog = CgProgram::new();
        let pred_expr = CgExpr::Lit(LitValue::Bool(true));
        prog.exprs.push(pred_expr);
        let op = ComputeOp {
            id: OpId(0),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: CgExprId(0),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(0) }],
            shape: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            span: Span::dummy(),
        };
        prog.ops.push(op);

        let errs = check_well_formed(&prog).expect_err("kind/shape mismatch");
        assert!(
            errs.contains(&CgError::KindShapeMismatch {
                op: OpId(0),
                kind_label: "mask_predicate",
                shape_label: "per_event",
            }),
            "missing KindShapeMismatch in {:?}",
            errs
        );
    }

    // -----------------------------------------------------------------
    // 10. Multi-error — collects all errors before returning
    // -----------------------------------------------------------------

    #[test]
    fn multi_error_collects_dangling_expr_and_type_mismatch() {
        // Build a program that has BOTH a type mismatch (mask predicate
        // with F32 body) AND a dangling expr id in another op. Both
        // errors must surface in the output Vec.
        let mut prog = CgProgram::new();
        // Op#0: type mismatch (predicate references hp, F32-typed)
        prog.exprs.push(read_self_hp()); // expr#0
        prog.ops.push(ComputeOp {
            id: OpId(0),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: CgExprId(0),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(0) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });
        // Op#1: dangling expr id
        prog.ops.push(ComputeOp {
            id: OpId(1),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(1),
                predicate: CgExprId(99),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(1) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });

        let errs = check_well_formed(&prog).expect_err("multi-error expected");

        // ExprIdOutOfRange for op#1.
        let saw_dangling = errs.contains(&CgError::ExprIdOutOfRange {
            op: OpId(1),
            referenced: CgExprId(99),
            arena_len: 1,
        });
        // TypeMismatch for op#0 (predicate is F32, expected Bool).
        let saw_type_mismatch = errs.iter().any(|e| {
            matches!(
                e,
                CgError::TypeMismatch {
                    op: OpId(0),
                    error: TypeError::ClaimedResultMismatch {
                        expected: CgTy::Bool,
                        got: CgTy::F32,
                        ..
                    },
                }
            )
        });

        assert!(
            saw_dangling && saw_type_mismatch,
            "expected both errors; got {:?}",
            errs
        );
        // Strict: at least 2 distinct errors collected.
        assert!(errs.len() >= 2, "expected ≥2 errors, got {:?}", errs);
    }

    // -----------------------------------------------------------------
    // 11. Empty program
    // -----------------------------------------------------------------

    #[test]
    fn empty_program_is_well_formed() {
        let prog = CgProgram::new();
        assert_eq!(check_well_formed(&prog), Ok(()));
    }

    // -----------------------------------------------------------------
    // Bonus: record_read-injected handle still passes well-formedness
    // -----------------------------------------------------------------

    #[test]
    fn record_read_injected_handle_passes_well_formedness() {
        // An op whose reads include a record_read-injected EventRing
        // (not in the auto-walker's output) is still well-formed.
        let mut b = CgProgramBuilder::new();
        let body = b.add_stmt_list(CgStmtList::new(vec![])).unwrap();
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(7),
            },
            Span::dummy(),
        )
        .unwrap();
        let mut prog = b.finish();
        // Lowering would inject the source-ring read.
        prog.ops[0].record_read(DataHandle::EventRing {
            ring: EventRingId(7),
            kind: crate::cg::data_handle::EventRingAccess::Read,
        });
        assert_eq!(check_well_formed(&prog), Ok(()));
    }

    // -----------------------------------------------------------------
    // Determinism: same program → same errors in same order across
    // multiple runs.
    // -----------------------------------------------------------------

    #[test]
    fn deterministic_error_ordering() {
        // Reuse the multi-error fixture.
        let mut prog = CgProgram::new();
        prog.exprs.push(read_self_hp());
        prog.ops.push(ComputeOp {
            id: OpId(0),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: CgExprId(0),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(0) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });
        prog.ops.push(ComputeOp {
            id: OpId(1),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(1),
                predicate: CgExprId(99),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(1) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });

        let a = check_well_formed(&prog).expect_err("multi-error");
        let b = check_well_formed(&prog).expect_err("multi-error (replay)");
        assert_eq!(a, b, "well-formed pass must be deterministic");
    }

    // -----------------------------------------------------------------
    // Bonus: lit_bool helper coverage to satisfy unused-import warning.
    // -----------------------------------------------------------------

    #[test]
    fn lit_bool_helper_compiles() {
        // Just exercise the helper so Cargo doesn't warn unused.
        let _ = lit_bool(true);
    }
}
