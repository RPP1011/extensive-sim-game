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
use std::fmt;

use crate::cg::data_handle::{AgentRef, CgExprId, CycleEdgeKey, DataHandle};
use crate::cg::dispatch::DispatchShape;
use crate::cg::expr::{data_handle_ty, type_check, CgExpr, CgTy, TypeCheckCtx, TypeError};
use crate::cg::op::{
    ComputeOp, ComputeOpKind, OpId, PlumbingKind, ScoringRowOp, Span, SpatialQueryKind,
};
use crate::cg::program::CgProgram;
use crate::cg::stmt::{CgStmt, CgStmtId, CgStmtListId, VariantId};

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

    /// A [`CgStmt::Assign`]'s value type does not match the storage
    /// type of its target [`DataHandle`]. The well-formed pass
    /// type-checks the assignment value with [`type_check`] and then
    /// compares the result to [`data_handle_ty`] of the target. A
    /// `Bool` value assigned to an `AgentField::Hp` (which is `F32`)
    /// is the canonical defect this variant catches.
    AssignTypeMismatch {
        op: OpId,
        target: DataHandle,
        expected: CgTy,
        got: CgTy,
    },

    /// A [`ScoringRowOp`]'s `target` expression has a type other than
    /// [`CgTy::AgentId`]. Per-ability scoring rows produce an agent-id
    /// candidate; a row whose `target` is `Some(expr)` typing to a
    /// non-agent type is structurally wrong and is rejected here.
    /// Standard rows leave `target` as `None`, which never fires this
    /// check. `row_index` pins which row inside the op carried the
    /// offending target.
    ScoringTargetNotAgentId {
        op: OpId,
        row_index: u32,
        got: CgTy,
    },

    /// A [`ScoringRowOp`]'s `guard` expression has a type other than
    /// [`CgTy::Bool`]. Per-ability scoring rows guard their score with
    /// a boolean predicate; a `Some(expr)` guard typing to anything
    /// other than `Bool` is structurally wrong. Standard rows leave
    /// `guard` as `None`, which never fires this check.
    ScoringGuardNotBool {
        op: OpId,
        row_index: u32,
        got: CgTy,
    },

    /// A [`ScoringRowOp`]'s `utility` expression has a type other than
    /// [`CgTy::F32`]. Scoring utilities are scalar floats —
    /// `engine::scoring` accumulates them with `+` and picks the
    /// argmax; an integer / agent-id / bool utility is rejected.
    /// `row_index` pins which row inside the op carried the offending
    /// utility.
    ScoringUtilityNotF32 {
        op: OpId,
        row_index: u32,
        got: CgTy,
    },

    /// Two arms of a [`crate::cg::stmt::CgStmt::Match`] reference the
    /// same [`VariantId`]. The
    /// [`crate::cg::stmt::CgMatchArm`]'s docstring promises the
    /// well-formed pass enforces this — duplicate arms would make
    /// dispatch order-dependent and indistinguishable to downstream
    /// emit. The AST resolver normally rejects duplicate source-level
    /// arms, so this defends against synthetic IRs that bypass the
    /// resolver. `span` is the enclosing op's span — the inner CgStmt
    /// does not carry one of its own.
    MatchDuplicateVariant {
        op: OpId,
        variant: VariantId,
        span: Span,
    },
}

impl fmt::Display for HandleConsistencyReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HandleConsistencyReason::AgentRefTargetExprOutOfRange {
                referenced,
                arena_len,
            } => write!(
                f,
                "AgentRef::Target(expr#{}) out of range (expr arena holds {} entries)",
                referenced.0, arena_len
            ),
        }
    }
}

impl fmt::Display for CgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CgError::OpIdOutOfRange {
                op,
                referenced,
                arena_len,
            } => write!(
                f,
                "op#{}: referenced op#{} out of range (op arena holds {} entries)",
                op.0, referenced.0, arena_len
            ),
            CgError::ExprIdOutOfRange {
                op,
                referenced,
                arena_len,
            } => write!(
                f,
                "op#{}: referenced expr#{} out of range (expr arena holds {} entries)",
                op.0, referenced.0, arena_len
            ),
            CgError::StmtListIdOutOfRange {
                op,
                referenced,
                arena_len,
            } => write!(
                f,
                "op#{}: referenced stmt-list#{} out of range (stmt-list arena holds {} entries)",
                op.0, referenced.0, arena_len
            ),
            CgError::StmtIdOutOfRange {
                list,
                referenced,
                arena_len,
            } => write!(
                f,
                "stmt-list#{}: referenced stmt#{} out of range (stmt arena holds {} entries)",
                list.0, referenced.0, arena_len
            ),
            CgError::DataHandleIdInconsistent {
                op,
                handle,
                reason,
            } => write!(
                f,
                "op#{}: data handle {} inconsistent — {}",
                op.0, handle, reason
            ),
            CgError::Cycle { ops } => {
                f.write_str("cycle in read/write graph: [")?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "op#{}", op.0)?;
                }
                f.write_str("]")
            }
            CgError::TypeMismatch { op, error } => {
                write!(f, "op#{}: type mismatch — {}", op.0, error)
            }
            CgError::P6Violation {
                op,
                kind_label,
                write,
            } => write!(
                f,
                "op#{}: P6 violation — {} writes agent field {}",
                op.0, kind_label, write
            ),
            CgError::KindShapeMismatch {
                op,
                kind_label,
                shape_label,
            } => write!(
                f,
                "op#{}: kind/shape mismatch — {} cannot dispatch as {}",
                op.0, kind_label, shape_label
            ),
            CgError::AssignTypeMismatch {
                op,
                target,
                expected,
                got,
            } => write!(
                f,
                "op#{}: assign type mismatch — target {} expects {}, got {}",
                op.0, target, expected, got
            ),
            CgError::ScoringTargetNotAgentId {
                op,
                row_index,
                got,
            } => write!(
                f,
                "op#{}: scoring row#{} target must be agent_id, got {}",
                op.0, row_index, got
            ),
            CgError::ScoringGuardNotBool {
                op,
                row_index,
                got,
            } => write!(
                f,
                "op#{}: scoring row#{} guard must be bool, got {}",
                op.0, row_index, got
            ),
            CgError::ScoringUtilityNotF32 {
                op,
                row_index,
                got,
            } => write!(
                f,
                "op#{}: scoring row#{} utility must be f32, got {}",
                op.0, row_index, got
            ),
            CgError::MatchDuplicateVariant { op, variant, span } => write!(
                f,
                "op#{}: match arms duplicate {} (span {}..{})",
                op.0, variant, span.start, span.end
            ),
        }
    }
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
        ComputeOpKind::Plumbing { .. } => "plumbing",
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
        // Plumbing kinds each have a single canonical dispatch shape,
        // pinned by `PlumbingKind::dispatch_shape`. The well-formed
        // pass mirrors that table — any drift between the lowering's
        // chosen shape and the IR-canonical shape surfaces as a
        // `KindShapeMismatch`.
        ComputeOpKind::Plumbing { kind } => match kind {
            PlumbingKind::PackAgents | PlumbingKind::UnpackAgents => {
                &[DispatchShapeLabel::PerAgent]
            }
            PlumbingKind::AliveBitmap => &[DispatchShapeLabel::PerWord],
            PlumbingKind::DrainEvents { .. } => &[DispatchShapeLabel::PerEvent],
            PlumbingKind::UploadSimCfg
            | PlumbingKind::KickSnapshot
            | PlumbingKind::SeedIndirectArgs { .. } => &[DispatchShapeLabel::OneShot],
        },
    }
}

// ---------------------------------------------------------------------------
// collect_subexpr_ids — adversarial-input expr-id reachability walk.
// ---------------------------------------------------------------------------

/// Recursively collect every [`CgExprId`] reachable from `root` (the
/// root id itself + every sub-id embedded in the [`CgExpr`] tree under
/// it). The walk is panic-free: arena lookups use `slice::get`, and
/// every recursion guards against cycles via a `visited` set so that a
/// corrupted arena with self-referential ids (`Binary { lhs: self, .. }`)
/// can't loop forever.
///
/// This is the load-bearing helper that closes the P10 panic gap: the
/// well-formed pass calls this BEFORE any [`type_check`] dispatch and
/// validates that every reachable id is in-range. If any sub-id is out
/// of range, the offending op skips its type-check pass (the type
/// checker would index past the arena) but still runs the structural
/// id-range / P6 / kind-shape / cycle checks.
fn collect_subexpr_ids(arena: &[CgExpr], root: CgExprId, out: &mut Vec<CgExprId>) {
    let mut visited: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut stack: Vec<CgExprId> = vec![root];
    while let Some(id) = stack.pop() {
        if !visited.insert(id.0) {
            // Already walked — avoids infinite loops on adversarially
            // self-referential arenas.
            continue;
        }
        out.push(id);
        // Use `slice::get` — never index with `[..]`.
        let Some(expr) = arena.get(id.0 as usize) else {
            // Out-of-range id; nothing to recurse into. The id itself
            // has already been recorded for the caller's range check.
            continue;
        };
        match expr {
            CgExpr::Binary { lhs, rhs, .. } => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            CgExpr::Unary { arg, .. } => {
                stack.push(*arg);
            }
            CgExpr::Builtin { args, .. } => {
                for a in args {
                    stack.push(*a);
                }
            }
            CgExpr::Select { cond, then, else_, .. } => {
                stack.push(*cond);
                stack.push(*then);
                stack.push(*else_);
            }
            CgExpr::Read(handle) => {
                // `AgentRef::Target(expr_id)` embeds an expression id
                // inside a data handle reachable from a `Read`. Fold
                // it into the reachability walk so the OOR check
                // catches it too.
                if let DataHandle::AgentField {
                    target: AgentRef::Target(target_expr),
                    ..
                } = handle
                {
                    stack.push(*target_expr);
                }
            }
            CgExpr::Lit(_) | CgExpr::Rng { .. } => {}
        }
    }
}

/// Validate every id collected by [`collect_subexpr_ids`] against the
/// arena length. Pushes a [`CgError::ExprIdOutOfRange`] for each id
/// past the arena end.
///
/// This pass is the source of [`CgError::ExprIdOutOfRange`] (with op
/// context). It does NOT gate the type-check pass — `type_check` itself
/// returns [`TypeError::DanglingExprId`] when it encounters an
/// out-of-range child, so running both is safe and produces non-
/// overlapping diagnostics (this pass reports the structural id-range
/// defect; `type_check` reports the type-coherence defect with a
/// different node anchor).
fn validate_expr_subtree(
    arena: &[CgExpr],
    root: CgExprId,
    op_id: OpId,
    expr_arena_len: u32,
    errors: &mut Vec<CgError>,
) {
    let mut ids = Vec::new();
    collect_subexpr_ids(arena, root, &mut ids);
    for id in ids {
        if id.0 >= expr_arena_len {
            // Dedup: only report the same offending id once per op.
            // (A repeated reference inside the tree would otherwise
            // produce duplicate errors.) We use `errors.iter()`
            // because the error set is small per op; the comparison
            // is an Eq match on the variant.
            let candidate = CgError::ExprIdOutOfRange {
                op: op_id,
                referenced: id,
                arena_len: expr_arena_len,
            };
            if !errors.contains(&candidate) {
                errors.push(candidate);
            }
        }
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
///
/// Returns nothing — all defects are pushed into `errors`. Each check
/// is independent: structural id-range checks (this pass's
/// [`validate_expr_subtree`]) report [`CgError::ExprIdOutOfRange`] with
/// op context; type checking ([`type_check`]) reports
/// [`CgError::TypeMismatch`] including [`TypeError::DanglingExprId`]
/// for the same arena defect from the type-coherence side. Both run
/// unconditionally — `type_check` is panic-free now that `ExprArena::get`
/// returns `Option`.
fn check_op(
    op: &ComputeOp,
    op_id: OpId,
    prog: &CgProgram,
    expr_arena_len: u32,
    stmt_arena_len: u32,
    list_arena_len: u32,
    errors: &mut Vec<CgError>,
) {
    // --- structural pre-pass: validate every reachable expr id. ---------
    //
    // Records [`CgError::ExprIdOutOfRange`] for every out-of-range id
    // reachable from the op's expression roots. We walk:
    //   - mask predicate id (root)
    //   - each scoring row's utility + target ids (roots)
    //   - each AgentRef::Target(expr_id) inside the op's reads/writes
    //   - every expression embedded in the op's body statements (Assign
    //     value, Emit field expr, If cond)
    //
    // Every sub-id reached recursively (Binary lhs/rhs, Unary arg,
    // Builtin args, Select cond/then/else_, Read(AgentField target))
    // is also validated.
    let arena: &[CgExpr] = prog.exprs.as_slice();

    match &op.kind {
        ComputeOpKind::MaskPredicate { predicate, .. } => {
            validate_expr_subtree(arena, *predicate, op_id, expr_arena_len, errors);
        }
        ComputeOpKind::ScoringArgmax { rows, .. } => {
            for row in rows {
                validate_expr_subtree(arena, row.utility, op_id, expr_arena_len, errors);
                if let Some(target_id) = row.target {
                    validate_expr_subtree(arena, target_id, op_id, expr_arena_len, errors);
                }
                if let Some(guard_id) = row.guard {
                    validate_expr_subtree(arena, guard_id, op_id, expr_arena_len, errors);
                }
            }
        }
        ComputeOpKind::PhysicsRule { body, .. } | ComputeOpKind::ViewFold { body, .. } => {
            // Body subexpressions are reached via `walk_body_expr_subtrees`
            // — see below. List-id-range is checked in the dedicated
            // pass below this match.
            if body.0 < list_arena_len {
                walk_body_expr_subtrees(
                    *body,
                    op_id,
                    prog,
                    expr_arena_len,
                    list_arena_len,
                    errors,
                );
            }
        }
        ComputeOpKind::SpatialQuery { .. } => {}
        // Plumbing kinds carry no embedded `CgExpr` / `CgStmt` — every
        // variant's reads/writes are typed `DataHandle`s sourced from
        // `PlumbingKind::dependencies()`. Nothing to walk.
        ComputeOpKind::Plumbing { .. } => {}
    }

    // Validate AgentRef::Target ids on every read/write handle.
    for handle in op.reads.iter().chain(op.writes.iter()) {
        if let DataHandle::AgentField {
            target: AgentRef::Target(expr_id),
            ..
        } = handle
        {
            validate_expr_subtree(arena, *expr_id, op_id, expr_arena_len, errors);
        }
    }

    // --- list/stmt id-range checks per kind --------------------------

    match &op.kind {
        ComputeOpKind::PhysicsRule { body, .. } | ComputeOpKind::ViewFold { body, .. } => {
            if body.0 >= list_arena_len {
                errors.push(CgError::StmtListIdOutOfRange {
                    op: op_id,
                    referenced: *body,
                    arena_len: list_arena_len,
                });
            } else {
                // Body resolves; walk it for inner stmt/list id-range
                // references (expr ids are already covered above).
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
        ComputeOpKind::MaskPredicate { .. }
        | ComputeOpKind::ScoringArgmax { .. }
        | ComputeOpKind::SpatialQuery { .. } => {}
        // Plumbing kinds reference no `CgStmtListId` — no list-range
        // check to perform.
        ComputeOpKind::Plumbing { .. } => {}
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
    //
    // Runs unconditionally — `type_check` is panic-free since
    // `ExprArena::get` returns `Option`. Out-of-range sub-ids surface
    // as [`TypeError::DanglingExprId`] wrapped in a
    // [`CgError::TypeMismatch`]; the structural id-range pass above
    // produced the [`CgError::ExprIdOutOfRange`] form independently.

    let ctx = TypeCheckCtx::new(prog);
    type_check_op(op, op_id, prog, &ctx, expr_arena_len, errors);

    // --- P6 mutation channel: AgentField writes only in ViewFold ----

    p6_check_op(op, op_id, prog, errors);

    // --- Match arm uniqueness -----------------------------------------

    match_uniqueness_check_op(op, op_id, prog, errors);

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

/// Walk a body stmt-list and validate every embedded expression
/// subtree against the arena (recursing through nested If branches).
/// Pushes [`CgError::ExprIdOutOfRange`] into `errors` for each reachable
/// out-of-range id.
///
/// Always returns when the list-id is out-of-range — that's a separate
/// defect reported by the structural pass; here we just stop walking
/// and never panic.
fn walk_body_expr_subtrees(
    list_id: CgStmtListId,
    op_id: OpId,
    prog: &CgProgram,
    expr_arena_len: u32,
    list_arena_len: u32,
    errors: &mut Vec<CgError>,
) {
    let Some(list) = prog.stmt_lists.get(list_id.0 as usize) else {
        return;
    };
    let arena: &[CgExpr] = prog.exprs.as_slice();
    for stmt_id in &list.stmts {
        let Some(stmt) = prog.stmts.get(stmt_id.0 as usize) else {
            continue;
        };
        match stmt {
            CgStmt::Assign { value, target } => {
                validate_expr_subtree(arena, *value, op_id, expr_arena_len, errors);
                // Validate AgentRef::Target embedded in the assign
                // target handle (the value-side is captured by the
                // op-level reads/writes pass too, but body-targets
                // can hold target-pointers that lowering computed).
                if let DataHandle::AgentField {
                    target: AgentRef::Target(target_expr),
                    ..
                } = target
                {
                    validate_expr_subtree(arena, *target_expr, op_id, expr_arena_len, errors);
                }
            }
            CgStmt::Emit { fields, .. } => {
                for (_, expr_id) in fields {
                    validate_expr_subtree(arena, *expr_id, op_id, expr_arena_len, errors);
                }
            }
            CgStmt::If { cond, then, else_ } => {
                validate_expr_subtree(arena, *cond, op_id, expr_arena_len, errors);
                if then.0 < list_arena_len {
                    walk_body_expr_subtrees(
                        *then,
                        op_id,
                        prog,
                        expr_arena_len,
                        list_arena_len,
                        errors,
                    );
                }
                if let Some(else_id) = else_ {
                    if else_id.0 < list_arena_len {
                        walk_body_expr_subtrees(
                            *else_id,
                            op_id,
                            prog,
                            expr_arena_len,
                            list_arena_len,
                            errors,
                        );
                    }
                }
            }
            CgStmt::Match { scrutinee, arms } => {
                validate_expr_subtree(arena, *scrutinee, op_id, expr_arena_len, errors);
                for arm in arms {
                    if arm.body.0 < list_arena_len {
                        walk_body_expr_subtrees(
                            arm.body,
                            op_id,
                            prog,
                            expr_arena_len,
                            list_arena_len,
                            errors,
                        );
                    }
                }
            }
            CgStmt::Let { value, .. } => {
                validate_expr_subtree(arena, *value, op_id, expr_arena_len, errors);
            }
        }
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
        // NB: expr-id range checks (Assign.value, Emit field, If.cond)
        // are handled by `walk_body_expr_subtrees` /
        // `validate_expr_subtree`, which also recurse into sub-ids
        // (Binary lhs/rhs etc.). This walk only handles list/stmt-id
        // ranges to avoid double-reporting expr defects.
        match stmt {
            CgStmt::Assign { .. } | CgStmt::Emit { .. } | CgStmt::Let { .. } => {
                // Nothing to range-check at the list-walk level —
                // these statements only embed expr-ids and (for
                // Assign) a target handle, all of which are validated
                // elsewhere. `Let` carries only an expr-id payload
                // plus a typed (LocalId, CgTy); the expr-id is caught
                // by the expr-subtree walk.
            }
            CgStmt::If { then, else_, .. } => {
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
            CgStmt::Match { arms, .. } => {
                for arm in arms {
                    if arm.body.0 >= list_arena_len {
                        errors.push(CgError::StmtListIdOutOfRange {
                            op: op_id,
                            referenced: arm.body,
                            arena_len: list_arena_len,
                        });
                    } else {
                        walk_list_id_ranges(
                            arm.body,
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
            AgentRef::Self_
            | AgentRef::Actor
            | AgentRef::EventTarget
            | AgentRef::PerPairCandidate => None,
        },
        DataHandle::ViewStorage { .. }
        | DataHandle::EventRing { .. }
        | DataHandle::ConfigConst { .. }
        | DataHandle::MaskBitmap { .. }
        | DataHandle::ScoringOutput
        | DataHandle::SpatialStorage { .. }
        | DataHandle::Rng { .. }
        | DataHandle::AliveBitmap
        | DataHandle::IndirectArgs { .. }
        | DataHandle::AgentScratch { .. }
        | DataHandle::SimCfgBuffer
        | DataHandle::SnapshotKick => None,
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
            // Safe indexing: caller guards by `expr_corrupted`, so any
            // sub-id reachable from `predicate` is in-range; still use
            // `.get` to never index past arena.
            let Some(expr) = prog.exprs.get(predicate.0 as usize) else {
                return;
            };
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
        ComputeOpKind::ScoringArgmax { rows, .. } => {
            for (row_index, row) in rows.iter().enumerate() {
                type_check_row(row, row_index as u32, op_id, prog, ctx, expr_arena_len, errors);
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
        ComputeOpKind::Plumbing { .. } => {
            // Plumbing kinds carry no embedded expressions; their
            // structural reads/writes are typed `DataHandle`s sourced
            // from `PlumbingKind::dependencies()`. Nothing to walk.
        }
    }
}

/// Type-check a single [`ScoringRowOp`]. Validates:
///   - `row.utility` type-checks AND its result is [`CgTy::F32`]
///     (a non-F32 utility is rejected as
///     [`CgError::ScoringUtilityNotF32`]).
///   - `row.target` (if `Some`) type-checks AND its result is
///     [`CgTy::AgentId`] (a non-AgentId target is rejected as
///     [`CgError::ScoringTargetNotAgentId`]). Standard rows whose
///     `target` is `None` skip this check.
///   - `row.guard` (if `Some`) type-checks AND its result is
///     [`CgTy::Bool`] (a non-Bool guard is rejected as
///     [`CgError::ScoringGuardNotBool`]). Rows without a guard skip
///     this check.
fn type_check_row(
    row: &ScoringRowOp,
    row_index: u32,
    op_id: OpId,
    prog: &CgProgram,
    ctx: &TypeCheckCtx<'_>,
    _expr_arena_len: u32,
    errors: &mut Vec<CgError>,
) {
    if let Some(expr) = prog.exprs.get(row.utility.0 as usize) {
        match type_check(expr, row.utility, ctx) {
            Ok(utility_ty) => {
                if utility_ty != CgTy::F32 {
                    errors.push(CgError::ScoringUtilityNotF32 {
                        op: op_id,
                        row_index,
                        got: utility_ty,
                    });
                }
            }
            Err(err) => errors.push(CgError::TypeMismatch {
                op: op_id,
                error: err,
            }),
        }
    }
    if let Some(target_id) = row.target {
        if let Some(expr) = prog.exprs.get(target_id.0 as usize) {
            match type_check(expr, target_id, ctx) {
                Ok(target_ty) => {
                    if target_ty != CgTy::AgentId {
                        errors.push(CgError::ScoringTargetNotAgentId {
                            op: op_id,
                            row_index,
                            got: target_ty,
                        });
                    }
                }
                Err(err) => errors.push(CgError::TypeMismatch {
                    op: op_id,
                    error: err,
                }),
            }
        }
    }
    if let Some(guard_id) = row.guard {
        if let Some(expr) = prog.exprs.get(guard_id.0 as usize) {
            match type_check(expr, guard_id, ctx) {
                Ok(guard_ty) => {
                    if guard_ty != CgTy::Bool {
                        errors.push(CgError::ScoringGuardNotBool {
                            op: op_id,
                            row_index,
                            got: guard_ty,
                        });
                    }
                }
                Err(err) => errors.push(CgError::TypeMismatch {
                    op: op_id,
                    error: err,
                }),
            }
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
            CgStmt::Assign { value, target } => {
                if let Some(expr) = prog.exprs.get(value.0 as usize) {
                    match type_check(expr, *value, ctx) {
                        Ok(value_ty) => {
                            // Compare value type vs the target
                            // handle's storage type. A mismatch
                            // surfaces as a structured
                            // `AssignTypeMismatch` (distinct from
                            // `TypeMismatch`, which wraps inner
                            // expression typing defects).
                            let expected = data_handle_ty(target);
                            if value_ty != expected {
                                errors.push(CgError::AssignTypeMismatch {
                                    op: op_id,
                                    target: target.clone(),
                                    expected,
                                    got: value_ty,
                                });
                            }
                        }
                        Err(err) => errors.push(CgError::TypeMismatch {
                            op: op_id,
                            error: err,
                        }),
                    }
                }
            }
            CgStmt::Emit { fields, .. } => {
                for (_, expr_id) in fields {
                    if let Some(expr) = prog.exprs.get(expr_id.0 as usize) {
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
                if let Some(expr) = prog.exprs.get(cond.0 as usize) {
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
            CgStmt::Match { scrutinee, arms } => {
                // Type-check the scrutinee expression (any well-typed
                // CgExpr is acceptable — sum-type scrutinees aren't
                // a distinct CgTy in v1, so the well-formed pass
                // doesn't gate on the scrutinee's type variant; it
                // only catches inner expression-tree defects).
                if let Some(expr) = prog.exprs.get(scrutinee.0 as usize) {
                    if let Err(err) = type_check(expr, *scrutinee, ctx) {
                        errors.push(CgError::TypeMismatch {
                            op: op_id,
                            error: err,
                        });
                    }
                }
                for arm in arms {
                    type_check_list(arm.body, op_id, prog, ctx, expr_arena_len, errors);
                }
            }
            CgStmt::Let { value, ty, .. } => {
                // Type-check the bound value expression. The Let's
                // declared `ty` must match the value expression's
                // computed type — a mismatch surfaces as a typed
                // `TypeMismatch` carrying a `ClaimedResultMismatch`.
                if let Some(expr) = prog.exprs.get(value.0 as usize) {
                    match type_check(expr, *value, ctx) {
                        Ok(value_ty) => {
                            if value_ty != *ty {
                                errors.push(CgError::TypeMismatch {
                                    op: op_id,
                                    error: TypeError::ClaimedResultMismatch {
                                        node: *value,
                                        expected: *ty,
                                        got: value_ty,
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
        // ViewFold returns above; Plumbing carries no body so there is
        // nothing to walk for the P6 mutation-channel check. Plumbing
        // ops do write `DataHandle::AgentField` directly when packing/
        // unpacking, but those writes appear on `op.writes` (sourced
        // from `PlumbingKind::dependencies`) rather than via a
        // `CgStmt::Assign`, so the P6 walker — which only inspects
        // `Assign` statements — has nothing to flag here.
        ComputeOpKind::ViewFold { .. } => {}
        ComputeOpKind::Plumbing { .. } => {}
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
            CgStmt::Let { .. } => {
                // Allowed — `Let` introduces a local binding only;
                // the binding has no AgentField write surface for
                // P6 to police.
            }
            CgStmt::If { then, else_, .. } => {
                p6_walk_list(*then, op_id, kind_label, prog, errors);
                if let Some(else_id) = else_ {
                    p6_walk_list(*else_id, op_id, kind_label, prog, errors);
                }
            }
            CgStmt::Match { arms, .. } => {
                for arm in arms {
                    p6_walk_list(arm.body, op_id, kind_label, prog, errors);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Match-arm variant uniqueness
// ---------------------------------------------------------------------------

/// Walk every [`crate::cg::stmt::CgStmt::Match`] reachable from this op's
/// body and report a [`CgError::MatchDuplicateVariant`] for each arm-set
/// that contains the same [`VariantId`] twice. The
/// [`crate::cg::stmt::CgMatchArm`]'s docstring promises this guarantee;
/// the AST resolver normally rejects duplicates, so this is the
/// defense-in-depth gate against synthetic IRs.
///
/// Walks bodies of [`ComputeOpKind::PhysicsRule`] and
/// [`ComputeOpKind::ViewFold`]; other kinds carry no statement bodies
/// and are skipped.
fn match_uniqueness_check_op(
    op: &ComputeOp,
    op_id: OpId,
    prog: &CgProgram,
    errors: &mut Vec<CgError>,
) {
    let body = match &op.kind {
        ComputeOpKind::PhysicsRule { body, .. } | ComputeOpKind::ViewFold { body, .. } => *body,
        ComputeOpKind::MaskPredicate { .. }
        | ComputeOpKind::ScoringArgmax { .. }
        | ComputeOpKind::SpatialQuery { .. }
        | ComputeOpKind::Plumbing { .. } => return,
    };
    match_uniqueness_walk_list(body, op, op_id, prog, errors);
}

/// Recursive walker — visits every nested list (If branches, Match arm
/// bodies, …) and inspects each `Match` for duplicate variant ids.
fn match_uniqueness_walk_list(
    list_id: CgStmtListId,
    op: &ComputeOp,
    op_id: OpId,
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
            CgStmt::Assign { .. } | CgStmt::Emit { .. } | CgStmt::Let { .. } => {
                // Leaves — no nested bodies to descend into.
            }
            CgStmt::If { then, else_, .. } => {
                match_uniqueness_walk_list(*then, op, op_id, prog, errors);
                if let Some(else_id) = else_ {
                    match_uniqueness_walk_list(*else_id, op, op_id, prog, errors);
                }
            }
            CgStmt::Match { arms, .. } => {
                let mut seen: BTreeSet<VariantId> = BTreeSet::new();
                for arm in arms {
                    if !seen.insert(arm.variant) {
                        errors.push(CgError::MatchDuplicateVariant {
                            op: op_id,
                            variant: arm.variant,
                            span: op.span,
                        });
                    }
                    match_uniqueness_walk_list(arm.body, op, op_id, prog, errors);
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
/// Implementation: build a `BTreeMap<CycleEdgeKey, Vec<OpId>>` of
/// writers per handle; for each op's reads, add edges from each writer
/// of that handle to this op. Then run a DFS-based SCC finder; report
/// any SCC of size > 1.
///
/// **Projection note.** Writers and readers are keyed by
/// [`DataHandle::cycle_edge_key`], not by raw `DataHandle`, so
/// [`DataHandle::EventRing`]'s `Read`/`Append`/`Drain` access discriminants
/// collapse to a single per-ring key. The access mode is a
/// producer/consumer marker, not a separate resource: a producer
/// `Append` and a consumer `Read` on the same ring still form a real
/// dependency edge, and the cycle gate must close that edge. Other
/// variants keep their full-identity keys via `CycleEdgeKey::Other`;
/// see `data_handle.rs`.
fn detect_cycles(prog: &CgProgram, errors: &mut Vec<CgError>) {
    let n = prog.ops.len();
    if n == 0 {
        return;
    }

    // Map projected handle → writers. We project via
    // `DataHandle::cycle_edge_key` so that producer/consumer ring
    // accesses with different `EventRingAccess` modes resolve to the
    // same key (see the docstring above). Other variants are wrapped
    // as `CycleEdgeKey::Other` and keep full-identity equality.
    let mut writers: BTreeMap<CycleEdgeKey, Vec<OpId>> = BTreeMap::new();
    for (op_index, op) in prog.ops.iter().enumerate() {
        let id = OpId(op_index as u32);
        for w in &op.writes {
            writers.entry(w.cycle_edge_key()).or_default().push(id);
        }
    }

    // adjacency: out-edges per op as a sorted set of `u32` indices
    // (OpId doesn't implement Ord, so store the inner u32 directly).
    let mut adj: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
    for (op_index, op) in prog.ops.iter().enumerate() {
        let consumer = OpId(op_index as u32);
        for r in &op.reads {
            if let Some(producers) = writers.get(&r.cycle_edge_key()) {
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
        AgentFieldId, AgentRef, EventRingAccess, EventRingId, MaskId, ViewId, ViewStorageSlot,
    };
    use crate::cg::expr::{BinaryOp, CgExpr, CgTy, LitValue};
    use crate::cg::op::{
        ActionId, ComputeOp, EventKindId, PhysicsRuleId, ReplayabilityFlag, ScoringId,
        ScoringRowOp, Span,
    };
    use crate::cg::program::{CgProgram, CgProgramBuilder};
    use crate::cg::stmt::{CgMatchArm, CgStmt, CgStmtId, CgStmtList, EventField};

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
                    target: Some(tgt),
                    guard: None,
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
                replayable: ReplayabilityFlag::Replayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(7),
            },
            Span::dummy(),
        )
        .unwrap();

        // --- view fold: assigns view storage (allowed) ---
        // ViewStorage::Primary's storage type is `ViewKey { view }`,
        // so we feed the slot a `Read` of the prior tick's primary
        // (the canonical fold pattern: read prior, transform, write
        // back). Picking a value of the right type avoids triggering
        // `AssignTypeMismatch`.
        let view_value = b
            .add_expr(CgExpr::Read(DataHandle::ViewStorage {
                view: ViewId(0),
                slot: ViewStorageSlot::Primary,
            }))
            .unwrap();
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
                replayable: ReplayabilityFlag::Replayable,
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
                replayable: ReplayabilityFlag::Replayable,
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
    fn cycle_two_ops_event_ring_read_and_append() {
        // Spec-shape ring cycle: op#0 reads ring(0) (Read) and writes
        // ring(1) (Append); op#1 reads ring(1) (Read) and writes
        // ring(0) (Append). The producer-side `Append` and the
        // consumer-side `Read` for each ring carry different
        // `EventRingAccess` discriminants, so a `DataHandle`
        // full-identity match would NOT close the edge — only the
        // `cycle_edge_key` projection by ring identity does.
        //
        // This is exactly what the driver's Phase 4 wiring emits;
        // verifying the gate fires here is the unit-level guarantee
        // the projection works for cycles in event-ring traffic.
        //
        // Build two trivial PhysicsRule ops with empty bodies, then
        // splice the ring reads/writes via record_read / record_write
        // (mirrors the lowering driver's wiring of source-ring reads
        // and destination-ring writes).
        let mut b = CgProgramBuilder::new();
        let body_a = b.add_stmt_list(CgStmtList::new(Vec::new())).unwrap();
        let body_b = b.add_stmt_list(CgStmtList::new(Vec::new())).unwrap();
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body: body_a,
                replayable: ReplayabilityFlag::Replayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap();
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(1),
                on_event: EventKindId(1),
                body: body_b,
                replayable: ReplayabilityFlag::Replayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(1),
            },
            Span::dummy(),
        )
        .unwrap();
        let mut prog = b.finish();

        // Op 0: consumer of ring 0 (Read), producer of ring 1 (Append).
        prog.ops[0].record_read(DataHandle::EventRing {
            ring: EventRingId(0),
            kind: EventRingAccess::Read,
        });
        prog.ops[0].record_write(DataHandle::EventRing {
            ring: EventRingId(1),
            kind: EventRingAccess::Append,
        });
        // Op 1: consumer of ring 1 (Read), producer of ring 0 (Append).
        prog.ops[1].record_read(DataHandle::EventRing {
            ring: EventRingId(1),
            kind: EventRingAccess::Read,
        });
        prog.ops[1].record_write(DataHandle::EventRing {
            ring: EventRingId(0),
            kind: EventRingAccess::Append,
        });

        let errs = check_well_formed(&prog)
            .expect_err("ring-symmetric cycle should trip the cycle gate");
        let saw_cycle = errs.iter().any(|e| match e {
            CgError::Cycle { ops } => {
                ops.contains(&OpId(0)) && ops.contains(&OpId(1)) && ops.len() == 2
            }
            _ => false,
        });
        assert!(
            saw_cycle,
            "missing 2-op Cycle on Read/Append ring edges in {:?}",
            errs
        );
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
                replayable: ReplayabilityFlag::Replayable,
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
                replayable: ReplayabilityFlag::Replayable,
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

    // -----------------------------------------------------------------
    // 12. If.cond non-Bool — closes Task 1.6 spec gap
    // -----------------------------------------------------------------

    #[test]
    fn if_cond_non_bool_returns_type_mismatch() {
        // A `CgStmt::If` whose `cond` is a non-Bool literal must
        // surface as a `TypeMismatch` with `ClaimedResultMismatch
        // { expected: Bool, got: F32 }`. Build the If inside a
        // physics-rule body (the only place If can live today).
        let mut b = CgProgramBuilder::new();
        // F32-typed cond — this is the defect we're testing.
        let cond = b.add_expr(lit_f32(1.0)).unwrap();
        // An empty inner body (no statements). The If's then/else
        // need real list ids; an empty list is fine.
        let inner = b.add_stmt_list(CgStmtList::new(vec![])).unwrap();
        let if_stmt = b
            .add_stmt(CgStmt::If {
                cond,
                then: inner,
                else_: None,
            })
            .unwrap();
        let body = b.add_stmt_list(CgStmtList::new(vec![if_stmt])).unwrap();
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body,
                replayable: ReplayabilityFlag::Replayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap();
        let prog = b.finish();

        let errs = check_well_formed(&prog).expect_err("If.cond non-Bool");
        let saw = errs.iter().any(|e| {
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
        assert!(saw, "missing If-cond Bool TypeMismatch in {:?}", errs);
    }

    // -----------------------------------------------------------------
    // 13. Assign value type vs target type
    // -----------------------------------------------------------------

    #[test]
    fn assign_wrong_value_type_returns_assign_type_mismatch() {
        // Assign a `Bool` value to AgentField::Hp (an F32 slot). The
        // pass must surface `AssignTypeMismatch { expected: F32, got:
        // Bool }`. Use a ViewFold (the one place AgentField writes
        // are allowed under P6).
        let mut b = CgProgramBuilder::new();
        let bool_val = b.add_expr(lit_bool(true)).unwrap();
        let assign = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: bool_val,
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

        let errs = check_well_formed(&prog).expect_err("Assign wrong type");
        let target_handle = DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        };
        assert!(
            errs.contains(&CgError::AssignTypeMismatch {
                op: OpId(0),
                target: target_handle,
                expected: CgTy::F32,
                got: CgTy::Bool,
            }),
            "missing AssignTypeMismatch in {:?}",
            errs
        );
    }

    // -----------------------------------------------------------------
    // 14. Scoring row target must be AgentId
    // -----------------------------------------------------------------

    #[test]
    fn scoring_row_target_non_agent_id_returns_error() {
        // A scoring row whose `target` is F32-typed (instead of
        // AgentId) must surface as `ScoringTargetNotAgentId`.
        let mut b = CgProgramBuilder::new();
        let util = b.add_expr(lit_f32(1.0)).unwrap();
        // F32-typed target — this is the defect.
        let bad_target = b.add_expr(lit_f32(0.0)).unwrap();
        b.add_op(
            ComputeOpKind::ScoringArgmax {
                scoring: ScoringId(0),
                rows: vec![ScoringRowOp {
                    action: ActionId(0),
                    utility: util,
                    target: Some(bad_target),
                    guard: None,
                }],
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap();
        let prog = b.finish();

        let errs = check_well_formed(&prog).expect_err("scoring target non-AgentId");
        assert!(
            errs.contains(&CgError::ScoringTargetNotAgentId {
                op: OpId(0),
                row_index: 0,
                got: CgTy::F32,
            }),
            "missing ScoringTargetNotAgentId in {:?}",
            errs
        );
    }

    // -----------------------------------------------------------------
    // 15. Nested expr-id OOR (the P10 panic gap)
    // -----------------------------------------------------------------

    #[test]
    fn nested_binary_lhs_out_of_range_does_not_panic() {
        // Construct a program where a top-level mask predicate
        // references a Binary whose `lhs` points at an out-of-range
        // expr id. The pre-fix implementation would PANIC here (the
        // type checker recursively dereferences `lhs` via
        // `ctx.arena.get(...)` which panics on a `Vec<CgExpr>` arena
        // with an OOR id, see expr.rs:684). Post-fix: the well-formed
        // pass surfaces a typed `ExprIdOutOfRange` and skips
        // type-check for this op.
        let mut prog = CgProgram::new();
        // expr#0: rhs (F32 lit)
        prog.exprs.push(CgExpr::Lit(LitValue::F32(0.5)));
        // expr#1: Binary with lhs = #99 (OOR), rhs = #0 (in-range)
        prog.exprs.push(CgExpr::Binary {
            op: BinaryOp::LtF32,
            lhs: CgExprId(99),
            rhs: CgExprId(0),
            ty: CgTy::Bool,
        });
        prog.ops.push(ComputeOp {
            id: OpId(0),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: CgExprId(1),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(0) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });

        let errs = check_well_formed(&prog).expect_err("nested OOR must not panic");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                CgError::ExprIdOutOfRange {
                    op: OpId(0),
                    referenced: CgExprId(99),
                    arena_len: 2,
                }
            )),
            "missing nested ExprIdOutOfRange in {:?}",
            errs
        );
    }

    #[test]
    fn nested_select_cond_out_of_range_does_not_panic() {
        // Similar to the Binary case but for Select.cond.
        let mut prog = CgProgram::new();
        // expr#0: F32 lit (then)
        prog.exprs.push(CgExpr::Lit(LitValue::F32(1.0)));
        // expr#1: F32 lit (else)
        prog.exprs.push(CgExpr::Lit(LitValue::F32(0.0)));
        // expr#2: Select with OOR cond
        prog.exprs.push(CgExpr::Select {
            cond: CgExprId(99),
            then: CgExprId(0),
            else_: CgExprId(1),
            ty: CgTy::F32,
        });
        // expr#3: Use expr#2 as a scoring row utility (the row's
        // target must be in-range AgentId; otherwise we'd shadow this
        // assertion with a separate row-target defect).
        prog.exprs.push(CgExpr::Lit(LitValue::AgentId(0)));
        prog.ops.push(ComputeOp {
            id: OpId(0),
            kind: ComputeOpKind::ScoringArgmax {
                scoring: ScoringId(0),
                rows: vec![ScoringRowOp {
                    action: ActionId(0),
                    utility: CgExprId(2),
                    target: Some(CgExprId(3)),
                    guard: None,
                }],
            },
            reads: vec![],
            writes: vec![DataHandle::ScoringOutput],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });

        let errs = check_well_formed(&prog).expect_err("nested Select OOR must not panic");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                CgError::ExprIdOutOfRange {
                    op: OpId(0),
                    referenced: CgExprId(99),
                    arena_len: 4,
                }
            )),
            "missing nested Select-cond ExprIdOutOfRange in {:?}",
            errs
        );
    }

    #[test]
    fn nested_oor_inside_agent_ref_target_does_not_panic() {
        // An AgentRef::Target(expr_id) embedded in a Read whose
        // expr_id is out-of-range. Pre-fix would panic (the Read's
        // sub-id is not validated by the structural pass). Post-fix:
        // the reachability walk picks up the embedded id and reports
        // it as `ExprIdOutOfRange`.
        let mut prog = CgProgram::new();
        // expr#0: a Read of AgentField with Target(#99) — OOR
        prog.exprs.push(CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Target(CgExprId(99)),
        }));
        // expr#1: Lit F32 (rhs)
        prog.exprs.push(CgExpr::Lit(LitValue::F32(0.5)));
        // expr#2: Binary with the Read as lhs — produces a Bool
        prog.exprs.push(CgExpr::Binary {
            op: BinaryOp::LtF32,
            lhs: CgExprId(0),
            rhs: CgExprId(1),
            ty: CgTy::Bool,
        });
        prog.ops.push(ComputeOp {
            id: OpId(0),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: CgExprId(2),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(0) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });

        let errs = check_well_formed(&prog).expect_err("AgentRef::Target OOR must not panic");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                CgError::ExprIdOutOfRange {
                    op: OpId(0),
                    referenced: CgExprId(99),
                    arena_len: 3,
                }
            )),
            "missing AgentRef::Target ExprIdOutOfRange in {:?}",
            errs
        );
    }

    // -----------------------------------------------------------------
    // 16. Cumulative-corruption test — load-bearing "never panics"
    // assertion. Stresses every check simultaneously.
    // -----------------------------------------------------------------

    #[test]
    fn cumulative_corruption_collects_all_error_kinds_without_panic() {
        // Build a program that simultaneously contains:
        //   (a) a nested expr-id OOR in op#0 (Binary lhs = #99)
        //   (b) a type mismatch in op#1 (mask predicate is F32-typed,
        //       not Bool)
        //   (c) a cycle in the read/write graph (op#2 + op#3 cross-
        //       reading each other's masks)
        //   (d) a P6 violation in op#4 (physics rule writing
        //       AgentField directly)
        //   (e) a kind/shape mismatch in op#5 (mask predicate paired
        //       with PerEvent dispatch)
        //
        // Asserts that `check_well_formed` returns Err (never panic)
        // and that AT LEAST ONE error of each kind is present.
        let mut prog = CgProgram::new();

        // --- Shared expressions used by multiple ops ------------------
        // expr#0: hp read (F32)
        prog.exprs.push(read_self_hp());
        // expr#1: 0.5 (F32)
        prog.exprs.push(lit_f32(0.5));
        // expr#2: bool lit (true) — used by op#5's predicate
        prog.exprs.push(lit_bool(true));
        // expr#3: Binary with OOR lhs — used by op#0 predicate
        prog.exprs.push(CgExpr::Binary {
            op: BinaryOp::LtF32,
            lhs: CgExprId(99),
            rhs: CgExprId(1),
            ty: CgTy::Bool,
        });
        // expr#4: Lit F32(0.0) — used by op#4's Assign value
        prog.exprs.push(lit_f32(0.0));

        // --- op#0: nested expr-id OOR (predicate #3 → Binary{lhs=#99}) ---
        prog.ops.push(ComputeOp {
            id: OpId(0),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: CgExprId(3),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(0) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });

        // --- op#1: type mismatch (predicate = #0 which is F32) ---
        prog.ops.push(ComputeOp {
            id: OpId(1),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(1),
                predicate: CgExprId(0),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(1) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });

        // --- op#2: writes mask(2), reads mask(7) — pair of cycle ---
        prog.ops.push(ComputeOp {
            id: OpId(2),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(2),
                predicate: CgExprId(2), // Bool-typed lit — well-typed
            },
            reads: vec![DataHandle::MaskBitmap { mask: MaskId(7) }],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(2) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });

        // --- op#3: writes mask(7), reads mask(2) — closes the cycle ---
        prog.ops.push(ComputeOp {
            id: OpId(3),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(7),
                predicate: CgExprId(2),
            },
            reads: vec![DataHandle::MaskBitmap { mask: MaskId(2) }],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(7) }],
            shape: DispatchShape::PerAgent,
            span: Span::dummy(),
        });

        // --- op#4: P6 violation — physics rule assigns AgentField ---
        // Build a stmt list with one Assign{AgentField, F32}.
        let assign_id = CgStmtId(0);
        prog.stmts.push(CgStmt::Assign {
            target: DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            },
            value: CgExprId(4),
        });
        let body_list_id = CgStmtListId(0);
        prog.stmt_lists.push(CgStmtList::new(vec![assign_id]));
        prog.ops.push(ComputeOp {
            id: OpId(4),
            kind: ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body: body_list_id,
                replayable: ReplayabilityFlag::Replayable,
            },
            reads: vec![],
            writes: vec![],
            shape: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            span: Span::dummy(),
        });

        // --- op#5: kind/shape mismatch — MaskPredicate + PerEvent ---
        prog.ops.push(ComputeOp {
            id: OpId(5),
            kind: ComputeOpKind::MaskPredicate {
                mask: MaskId(8),
                predicate: CgExprId(2),
            },
            reads: vec![],
            writes: vec![DataHandle::MaskBitmap { mask: MaskId(8) }],
            shape: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            span: Span::dummy(),
        });

        // The pass must NOT panic on this adversarial input.
        let errs = check_well_formed(&prog).expect_err("cumulative corruption");

        let saw_oor = errs
            .iter()
            .any(|e| matches!(e, CgError::ExprIdOutOfRange { .. }));
        let saw_type_mismatch = errs
            .iter()
            .any(|e| matches!(e, CgError::TypeMismatch { .. }));
        let saw_cycle = errs.iter().any(|e| matches!(e, CgError::Cycle { .. }));
        let saw_p6 = errs
            .iter()
            .any(|e| matches!(e, CgError::P6Violation { .. }));
        let saw_kind_shape = errs
            .iter()
            .any(|e| matches!(e, CgError::KindShapeMismatch { .. }));

        assert!(
            saw_oor && saw_type_mismatch && saw_cycle && saw_p6 && saw_kind_shape,
            "missing one or more error kinds; got {:?}",
            errs
        );
    }

    // -----------------------------------------------------------------
    // 17. Display impl on CgError — every variant produces a non-empty
    // human-readable message containing the variant tag.
    // -----------------------------------------------------------------

    #[test]
    fn cg_error_display_covers_every_variant() {
        let cases: Vec<(CgError, &str)> = vec![
            (
                CgError::OpIdOutOfRange {
                    op: OpId(1),
                    referenced: OpId(7),
                    arena_len: 3,
                },
                "op#1: referenced op#7 out of range",
            ),
            (
                CgError::ExprIdOutOfRange {
                    op: OpId(0),
                    referenced: CgExprId(99),
                    arena_len: 5,
                },
                "op#0: referenced expr#99 out of range",
            ),
            (
                CgError::StmtListIdOutOfRange {
                    op: OpId(2),
                    referenced: CgStmtListId(4),
                    arena_len: 1,
                },
                "op#2: referenced stmt-list#4 out of range",
            ),
            (
                CgError::StmtIdOutOfRange {
                    list: CgStmtListId(0),
                    referenced: CgStmtId(8),
                    arena_len: 2,
                },
                "stmt-list#0: referenced stmt#8 out of range",
            ),
            (
                CgError::DataHandleIdInconsistent {
                    op: OpId(0),
                    handle: DataHandle::AgentField {
                        field: AgentFieldId::Hp,
                        target: AgentRef::Target(CgExprId(99)),
                    },
                    reason: HandleConsistencyReason::AgentRefTargetExprOutOfRange {
                        referenced: CgExprId(99),
                        arena_len: 1,
                    },
                },
                "data handle",
            ),
            (
                CgError::Cycle {
                    ops: vec![OpId(0), OpId(1)],
                },
                "cycle in read/write graph",
            ),
            (
                CgError::TypeMismatch {
                    op: OpId(0),
                    error: TypeError::ClaimedResultMismatch {
                        node: CgExprId(2),
                        expected: CgTy::Bool,
                        got: CgTy::F32,
                    },
                },
                "type mismatch",
            ),
            (
                CgError::P6Violation {
                    op: OpId(0),
                    kind_label: "physics_rule",
                    write: DataHandle::AgentField {
                        field: AgentFieldId::Hp,
                        target: AgentRef::Self_,
                    },
                },
                "P6 violation",
            ),
            (
                CgError::KindShapeMismatch {
                    op: OpId(0),
                    kind_label: "mask_predicate",
                    shape_label: "per_event",
                },
                "kind/shape mismatch",
            ),
            (
                CgError::AssignTypeMismatch {
                    op: OpId(0),
                    target: DataHandle::AgentField {
                        field: AgentFieldId::Hp,
                        target: AgentRef::Self_,
                    },
                    expected: CgTy::F32,
                    got: CgTy::Bool,
                },
                "assign type mismatch",
            ),
            (
                CgError::ScoringTargetNotAgentId {
                    op: OpId(0),
                    row_index: 0,
                    got: CgTy::F32,
                },
                "scoring row#0 target must be agent_id",
            ),
            (
                CgError::MatchDuplicateVariant {
                    op: OpId(0),
                    variant: VariantId(7),
                    span: Span::new(3, 9),
                },
                "match arms duplicate",
            ),
        ];
        for (err, needle) in cases {
            let s = format!("{}", err);
            assert!(!s.is_empty(), "Display of {err:?} produced an empty string");
            assert!(
                s.contains(needle),
                "Display of {err:?} = {s:?} missing substring {needle:?}"
            );
        }
    }

    /// Build a `PhysicsRule` op whose body contains a `Match` with two
    /// arms that share the same `VariantId(0)`. The well-formed pass
    /// must surface a [`CgError::MatchDuplicateVariant`].
    #[test]
    fn match_arms_with_duplicate_variant_id_rejected() {
        let mut b = CgProgramBuilder::new();
        // Scrutinee — any well-typed expression suffices; the pass
        // doesn't gate on the scrutinee's variant.
        let scrutinee = b
            .add_expr(CgExpr::Lit(LitValue::F32(0.0)))
            .expect("add scrutinee");
        // Two arms, both `VariantId(0)`. Empty bodies.
        let empty_body = b
            .add_stmt_list(CgStmtList::new(vec![]))
            .expect("empty arm body");
        let match_stmt = b
            .add_stmt(CgStmt::Match {
                scrutinee,
                arms: vec![
                    CgMatchArm {
                        variant: VariantId(0),
                        bindings: vec![],
                        body: empty_body,
                    },
                    CgMatchArm {
                        variant: VariantId(0),
                        bindings: vec![],
                        body: empty_body,
                    },
                ],
            })
            .expect("add match");
        let body = b
            .add_stmt_list(CgStmtList::new(vec![match_stmt]))
            .expect("body list");
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body,
                replayable: ReplayabilityFlag::NonReplayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .expect("add op");
        let prog = b.finish();
        let errs = check_well_formed(&prog).expect_err("duplicate variant must surface");
        let dup = errs
            .iter()
            .find(|e| matches!(e, CgError::MatchDuplicateVariant { .. }))
            .expect("MatchDuplicateVariant present in error vec");
        match dup {
            CgError::MatchDuplicateVariant { op, variant, .. } => {
                assert_eq!(*op, OpId(0));
                assert_eq!(*variant, VariantId(0));
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    /// Inverse: a `Match` whose arms are all distinct passes the
    /// uniqueness check (no `MatchDuplicateVariant` in the error
    /// vec).
    #[test]
    fn match_arms_with_distinct_variant_ids_pass() {
        let mut b = CgProgramBuilder::new();
        let scrutinee = b
            .add_expr(CgExpr::Lit(LitValue::F32(0.0)))
            .expect("add scrutinee");
        let empty_body = b
            .add_stmt_list(CgStmtList::new(vec![]))
            .expect("empty arm body");
        let match_stmt = b
            .add_stmt(CgStmt::Match {
                scrutinee,
                arms: vec![
                    CgMatchArm {
                        variant: VariantId(0),
                        bindings: vec![],
                        body: empty_body,
                    },
                    CgMatchArm {
                        variant: VariantId(1),
                        bindings: vec![],
                        body: empty_body,
                    },
                ],
            })
            .expect("add match");
        let body = b
            .add_stmt_list(CgStmtList::new(vec![match_stmt]))
            .expect("body list");
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body,
                replayable: ReplayabilityFlag::NonReplayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .expect("add op");
        let prog = b.finish();
        // The well-formed pass may surface other defects, but it must
        // NOT include a MatchDuplicateVariant for this op.
        let result = check_well_formed(&prog);
        if let Err(errs) = result {
            assert!(
                !errs
                    .iter()
                    .any(|e| matches!(e, CgError::MatchDuplicateVariant { .. })),
                "no MatchDuplicateVariant expected, got: {errs:?}"
            );
        }
    }

    /// well_formed validates `CgStmt::Let.value`'s expression-id
    /// reference against the program's expression arena. An
    /// out-of-range value id surfaces as
    /// [`CgError::ExprIdOutOfRange`] (caught by the expr-subtree
    /// walk this Task 5.5b extends to cover the `Let` arm).
    #[test]
    fn let_value_out_of_range_reports_expr_id_out_of_range() {
        // Build a CgProgram with a `Let` whose `value` references a
        // dangling expression id. We bypass `add_stmt`'s validation
        // by mutating the arena directly — the well_formed pass is
        // the gate.
        use crate::cg::stmt::LocalId;
        let mut b = CgProgramBuilder::new();
        let real = b
            .add_expr(CgExpr::Lit(LitValue::F32(1.0)))
            .expect("add expr");
        // Real Let pointing at an in-range expr (so `add_stmt` accepts).
        let let_stmt = b
            .add_stmt(CgStmt::Let {
                local: LocalId(0),
                value: real,
                ty: CgTy::F32,
            })
            .expect("add Let");
        let body = b
            .add_stmt_list(CgStmtList::new(vec![let_stmt]))
            .expect("body list");
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body,
                replayable: ReplayabilityFlag::NonReplayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .expect("add op");
        let mut prog = b.finish();
        // Now corrupt the Let's value id past the arena's end.
        let dangling = CgExprId(prog.exprs.len() as u32 + 5);
        if let CgStmt::Let { value, .. } = &mut prog.stmts[let_stmt.0 as usize] {
            *value = dangling;
        }
        let errs = check_well_formed(&prog).expect_err("dangling Let.value");
        assert!(
            errs.iter().any(
                |e| matches!(e, CgError::ExprIdOutOfRange { referenced, .. } if *referenced == dangling)
            ),
            "expected ExprIdOutOfRange for the dangling Let.value, got: {errs:?}"
        );
    }
}
