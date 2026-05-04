//! `CgStmt` — the compute-graph statement tree.
//!
//! Statements are the "body" form for compute ops that mutate state in
//! a structured way: physics rules and view-fold handlers. They are
//! sibling to [`crate::cg::CgExpr`]: expressions are pure and produce a
//! value, statements are effectful and produce a side effect on a
//! [`DataHandle`] or an event ring.
//!
//! # Variant set
//!
//! The CG layer ships four statement forms — assignment, event emit,
//! conditional, and typed pattern match. The DSL surface AST has
//! additional control-flow and binding forms (`for`, `let`, `belief
//! observe`); the AST → CG lowering (Phase 2 of the plan) is
//! responsible for desugaring them — `for` unrolls or fuses into the
//! dispatch shape, `let` flattens via SSA expression sharing,
//! `BeliefObserve` decomposes into a sequence of `Assign`s against the
//! BeliefState SoA fields. The `match` form survives lowering: physics
//! rules like `cast` dispatch on stdlib sum types (`EffectOp` variant)
//! and the typed `CgStmt::Match` pins the variant id + binders so emit
//! can produce the correct match arms without re-resolving variant
//! names. Adding a new CG-level statement variant is therefore a
//! deliberate choice: it widens the set of shapes every later layer
//! (HIR/MIR/LIR + emit) must handle.
//!
//! # Arena vs node
//!
//! Like [`crate::cg::CgExpr`], statements form a tree but reference
//! their children by id (`CgStmtListId` for sub-bodies, `CgStmtId` for
//! sibling list entries). The [`StmtArena`] trait resolves statement
//! ids to nodes; [`crate::cg::expr::ExprArena`] resolves expression
//! ids. The actual `Vec<CgStmt>` arena lives in `CgProgram` (Task 1.5);
//! tests use `Vec<CgStmt>` directly via the `impl StmtArena for Vec`
//! provided here.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::data_handle::{CgExprId, DataHandle};
use super::expr::{CgExpr, CgTy, ExprArena};
use super::op::EventKindId;

// ---------------------------------------------------------------------------
// Newtype IDs
// ---------------------------------------------------------------------------

/// Stable id for a statement node in a [`StmtArena`]. Sibling to
/// [`CgExprId`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CgStmtId(pub u32);

/// Stable id for a [`CgStmtList`] in the program. The list arena is
/// separate from the statement arena: `CgStmtListId` indexes into a
/// `Vec<CgStmtList>`, each list holds its `Vec<CgStmtId>` of statements
/// in execution order.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CgStmtListId(pub u32);

/// Stable id for a field within an event variant. `event` names which
/// event variant; `index` is the 0-based position of the field in the
/// variant's declared field list. The pair maps to a concrete struct
/// field at emit time (the lowering resolves it against the program's
/// event table).
///
/// Encoding the (event, index) pair structurally — rather than a flat
/// `EventFieldId(u32)` — keeps the IR honest about the fact that field
/// indices only make sense relative to a particular event variant.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EventField {
    pub event: EventKindId,
    pub index: u8,
}

impl fmt::Display for EventField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "event[#{}].field#{}", self.event.0, self.index)
    }
}

/// Stable id for a sum-type variant — used by [`CgStmt::Match`] to name
/// the case its arm matches. The id resolves through a driver-supplied
/// registry (the lowering context's `variant_ids` map) at AST → CG
/// lowering time; emit then maps the id to the concrete `<EnumName>::<Variant>`
/// path in the generated code.
///
/// Today the only sum type the IR matches over is the stdlib `EffectOp`
/// (the `cast` physics rule's `match op { Damage { amount } => … }` shape);
/// future plumbing-level scrutinees (e.g., a generic enum surface) reuse
/// the same id space — the registry is the single source of truth.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VariantId(pub u32);

impl fmt::Display for VariantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "variant#{}", self.0)
    }
}

/// Stable id for a local binding introduced inside a body (a match-arm
/// pattern binder, a `let`-introduced local, a fold binder once
/// supported). The id resolves through a driver-supplied registry
/// (the lowering context's `local_ids` map). The CG IR carries the id
/// rather than the source-level name so consumers (emit, fusion) don't
/// re-parse strings to disambiguate two bindings sharing a name across
/// nested scopes.
///
/// Locals are not first-class CG expressions today — bindings only
/// surface as match-arm payload destinations (a future task wires
/// `CgExpr::Local(LocalId)` reads when expression lowering grows local
/// resolution).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LocalId(pub u32);

impl fmt::Display for LocalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "local#{}", self.0)
    }
}

/// One field-binding in a [`CgMatchArm`]'s pattern.
///
/// `field_name` is the source-level identifier on the variant
/// (`"amount"` for `Damage { amount }`, `"factor_q8"` for `Slow { …,
/// factor_q8 }`). It stays a `String` because variant field names are
/// free-form per-variant identifiers; unlike [`EventField`] (which uses
/// the variant's declared field-index because every event variant is
/// authored in the user's DSL with a fixed schema), the stdlib sum
/// types matched over here (`EffectOp`) carry their fields by name in
/// the resolver's IR — the typed-index registry doesn't exist for them.
/// Wrapping a single name in an opaque newtype would not buy any
/// type-safety improvement; the `String` is the honest carrier.
///
/// `local` is the typed [`LocalId`] the binding introduces. Body
/// references to the binder will resolve against this id once
/// expression lowering grows local-binding support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchArmBinding {
    pub field_name: String,
    pub local: LocalId,
}

impl fmt::Display for MatchArmBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.field_name, self.local)
    }
}

/// One arm of a [`CgStmt::Match`].
///
/// `variant` selects the sum-type case this arm matches; `bindings`
/// names each captured field; `body` is the statement list executed
/// when the variant matches.
///
/// The lowering produces arms in source order. The well-formed pass
/// validates that no two arms share a variant id (today the only
/// sum type matched is `EffectOp`, where the resolver already rejects
/// duplicate arms; defense-in-depth at the CG level catches a
/// synthetic IR that bypasses resolve).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CgMatchArm {
    pub variant: VariantId,
    pub bindings: Vec<MatchArmBinding>,
    pub body: CgStmtListId,
}

impl fmt::Display for CgMatchArm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "arm({}, [", self.variant)?;
        for (i, b) in self.bindings.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", b)?;
        }
        write!(f, "], body=stmts#{})", self.body.0)
    }
}

// ---------------------------------------------------------------------------
// CgStmt
// ---------------------------------------------------------------------------

/// Statement node in the compute-graph statement tree.
///
/// Every variant produces a side effect:
/// - `Assign` writes a value into a [`DataHandle`].
/// - `Emit` appends an event record into an event ring (the ring is
///   identified by the event variant's [`EventKindId`]; the lowering
///   resolves which ring at emit time).
/// - `If` branches between two sub-bodies. Both arms are themselves
///   `CgStmtList`s; the `else` branch is optional.
/// - `Match` dispatches on a typed sum-type scrutinee. Each arm names
///   a [`VariantId`], a list of field-binders, and a body statement
///   list. Used by the `cast` physics rule and similar — the AST
///   resolves the variant + binder names to typed ids before lowering
///   so emit can produce match arms without re-resolving strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CgStmt {
    /// Write `value` into the slot named by `target`. The slot type is
    /// determined by the data handle (see [`crate::cg::data_handle_ty`]);
    /// the type checker (Task 1.6 well-formed pass) ensures `value`'s
    /// CgTy matches.
    Assign { target: DataHandle, value: CgExprId },

    /// Emit an event record. `event` selects the event variant;
    /// `fields` lists each declared field of that variant in order
    /// with the expression that computes its payload. Field indices
    /// not present in `fields` carry the variant's default value at
    /// emit time (the well-formed pass enforces full coverage).
    Emit {
        event: EventKindId,
        fields: Vec<(EventField, CgExprId)>,
    },

    /// Conditional execution. `cond` is a Bool-typed expression;
    /// `then` runs when it is true, `else_` (if present) runs when
    /// false.
    If {
        cond: CgExprId,
        then: CgStmtListId,
        else_: Option<CgStmtListId>,
    },

    /// Pattern-match dispatch. `scrutinee` is the value being matched
    /// (a sum-type-shaped expression — today only the stdlib
    /// `EffectOp` from `cast` rules); `arms` enumerates the cases in
    /// source order. Each arm names a [`VariantId`], its field
    /// binders, and the statement list to execute on match.
    ///
    /// The arm body sees its [`MatchArmBinding`] locals in scope. The
    /// CG IR does not yet thread local references through expression
    /// reads — bodies that reference a binder lower as
    /// [`crate::cg::lower::LoweringError::UnsupportedLocalBinding`]
    /// at the expression layer until expression lowering grows local
    /// resolution.
    Match {
        scrutinee: CgExprId,
        arms: Vec<CgMatchArm>,
    },

    /// Local binding statement — `let local_<N>: <ty> = <value>;`.
    /// Lowered from `IrStmt::Let { name, local, value, span }` inside
    /// physics rule bodies. The [`LocalId`] identifies the binder;
    /// expression-tree references to the same local will resolve
    /// through the lowering context's `local_ids` map once
    /// `IrExpr::Local` resolution lands at the expression layer
    /// (Task 5.5d).
    ///
    /// `value` is the bound expression; its CG-typed result must
    /// match `ty`. Emit consults `ty` to declare the local in WGSL
    /// (`let local_<N>: <wgsl-ty> = ...;`) and in Rust.
    ///
    /// # Limitations
    ///
    /// References to the bound local from later statements still
    /// surface as
    /// [`crate::cg::lower::LoweringError::UnsupportedLocalBinding`]
    /// at the expression layer — wiring the read-side resolution is
    /// Task 5.5d. Today's Let arm represents the binding
    /// structurally; consumers are a follow-up.
    Let {
        local: LocalId,
        value: CgExprId,
        ty: CgTy,
    },

    /// N² fold over alive agents — produces a single accumulator value
    /// by walking every agent slot and (optionally filtered) summing a
    /// per-candidate projection.
    ///
    /// Lowered from `IrExpr::Fold { kind: Sum|Count, binder, iter:
    /// agents, body }`. The fold expression itself returns a
    /// [`CgExpr::ReadLocal`] reading `acc_local`; this stmt is
    /// injected before the consumer (via the lowering context's
    /// `pending_pre_stmts` buffer) so the value is already populated
    /// at the read site.
    ///
    /// # WGSL emit shape
    ///
    /// ```text
    /// var local_<acc_local>: <acc_ty> = <init>;
    /// for (var per_pair_candidate: u32 = 0u; per_pair_candidate < cfg.agent_cap; per_pair_candidate = per_pair_candidate + 1u) {
    ///     local_<acc_local> = local_<acc_local> + <projection>;
    /// }
    /// ```
    ///
    /// Where the projection is whatever the fold body resolved to —
    /// for Count, it's `select(0i, 1i, body)` (body is the predicate);
    /// for Sum, it's `body` directly (body is the projection). The
    /// emit layer assembles the boolean projection at lowering time
    /// (Task 33), so this stmt only sees the final WGSL-level
    /// "addend" expression.
    ///
    /// # Loop variable
    ///
    /// The implicit loop variable is named `per_pair_candidate` —
    /// reused from the existing pair-bound emit convention so reads
    /// of `binder.<field>` inside the body lower to
    /// `DataHandle::AgentField { target: AgentRef::PerPairCandidate }`
    /// without inventing a parallel naming scheme. Folds inside
    /// pair-bound contexts (mask predicates etc.) would shadow the
    /// outer pair-candidate; today the boids fixture does not nest
    /// folds inside pair-bound contexts, so the shadow is benign and
    /// well-formed. If a future fixture nests them, the emit will
    /// need to allocate a fresh loop-var name per ForEachAgent.
    ForEachAgent {
        acc_local: LocalId,
        acc_ty: CgTy,
        init: CgExprId,
        projection: CgExprId,
    },

    /// **Spatial-grid-bounded** fold over agents. Same shape as
    /// [`Self::ForEachAgent`] but the walk visits only the cells
    /// inside `radius_cells` of `self`'s cell — for the boids
    /// fixture's perception-radius fold, that's the 3³=27
    /// neighborhood (`radius_cells = 1`). The host pre-populates
    /// `spatial_grid_offsets` / `spatial_grid_cells` via the
    /// `SpatialQuery::BuildHash` kernel earlier in the schedule.
    ///
    /// # WGSL emit shape
    ///
    /// ```text
    /// var local_<acc_local>: <acc_ty> = <init>;
    /// let self_cell_xyz = pos_to_cell_xyz(agent_pos[agent_id]);
    /// for (var dz: i32 = -<radius>; dz <= <radius>; dz = dz + 1) {
    ///   for (var dy: i32 = -<radius>; dy <= <radius>; dy = dy + 1) {
    ///     for (var dx: i32 = -<radius>; dx <= <radius>; dx = dx + 1) {
    ///       let cell = cell_index(...);
    ///       let count = min(atomicLoad(&spatial_grid_offsets[cell]),
    ///                       SPATIAL_MAX_PER_CELL);
    ///       for (var i: u32 = 0u; i < count; i = i + 1u) {
    ///         let per_pair_candidate =
    ///           spatial_grid_cells[cell * SPATIAL_MAX_PER_CELL + i];
    ///         local_<acc_local> = local_<acc_local> + <projection>;
    ///       }
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// # Cell radius
    ///
    /// `radius_cells` is the inclusive half-width of the cell
    /// neighborhood — `1` walks 27 cells, `2` walks 125. For boids,
    /// `CELL_SIZE` is set equal to `perception_radius`, so a single-
    /// cell radius covers every potential neighbor. Larger fold
    /// radii (relative to `CELL_SIZE`) bump this up at the lowering
    /// site.
    ForEachNeighbor {
        acc_local: LocalId,
        acc_ty: CgTy,
        init: CgExprId,
        projection: CgExprId,
        radius_cells: u32,
    },

    /// **Spatial-grid-bounded body iteration** — same per-cell walk
    /// as [`Self::ForEachNeighbor`], but the body is a per-candidate
    /// statement list rather than a fold accumulator. Lowered from
    /// `IrStmt::For { binder, iter: spatial.<query>(self), body, .. }`
    /// in physics rule bodies (slice 2b of the stdlib-into-CG-IR
    /// plan).
    ///
    /// Unlike the fold form, the body can carry side effects
    /// (typically `emit <Event> { … }` to surface a per-pair record)
    /// and references to `<binder>.<field>` / `<binder>` resolve
    /// through [`crate::cg::data_handle::AgentRef::PerPairCandidate`]
    /// just like the fold's projection.
    ///
    /// # WGSL emit shape
    ///
    /// Mirrors [`Self::ForEachNeighbor`]'s 27-cell global-memory walk
    /// scaffolding — the body executes once per neighbour candidate
    /// with `per_pair_candidate` bound to the candidate's slot id.
    ///
    /// # Binder
    ///
    /// `binder` is the [`LocalId`] introduced by the source-level
    /// `for <binder> in …` clause. The lowering registers
    /// `local_ids[ast_ref] = binder` and records
    /// `local_tys[binder] = AgentId` — the WGSL emit does NOT consult
    /// `binder` directly because `per_pair_candidate` is the
    /// kernel-side name for the same slot id; the field is carried
    /// for diagnostic / future-emit-shape clarity.
    ///
    /// # Cell radius
    ///
    /// `radius_cells` mirrors [`Self::ForEachNeighbor`]'s field —
    /// inclusive half-width of the cell neighborhood walked.
    ForEachNeighborBody {
        binder: LocalId,
        body: CgStmtListId,
        radius_cells: u32,
    },
}

impl fmt::Display for CgStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CgStmt::Assign { target, value } => {
                write!(f, "assign({} <- expr#{})", target, value.0)
            }
            CgStmt::Emit { event, fields } => {
                write!(f, "emit(event[#{}]", event.0)?;
                for (fld, expr) in fields {
                    write!(f, ", {}=expr#{}", fld, expr.0)?;
                }
                f.write_str(")")
            }
            CgStmt::If { cond, then, else_ } => match else_ {
                Some(else_id) => write!(
                    f,
                    "if(cond=expr#{}, then=stmts#{}, else=stmts#{})",
                    cond.0, then.0, else_id.0
                ),
                None => write!(f, "if(cond=expr#{}, then=stmts#{})", cond.0, then.0),
            },
            CgStmt::Match { scrutinee, arms } => {
                write!(f, "match(scrutinee=expr#{}, arms=[", scrutinee.0)?;
                for (i, arm) in arms.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", arm)?;
                }
                f.write_str("])")
            }
            CgStmt::Let { local, value, ty } => {
                write!(f, "let({}: {} = expr#{})", local, ty, value.0)
            }
            CgStmt::ForEachAgent {
                acc_local,
                acc_ty,
                init,
                projection,
            } => {
                write!(
                    f,
                    "for_each_agent(acc={}: {}, init=expr#{}, proj=expr#{})",
                    acc_local, acc_ty, init.0, projection.0
                )
            }
            CgStmt::ForEachNeighbor {
                acc_local,
                acc_ty,
                init,
                projection,
                radius_cells,
            } => {
                write!(
                    f,
                    "for_each_neighbor(acc={}: {}, init=expr#{}, proj=expr#{}, r={})",
                    acc_local, acc_ty, init.0, projection.0, radius_cells
                )
            }
            CgStmt::ForEachNeighborBody {
                binder,
                body,
                radius_cells,
            } => {
                write!(
                    f,
                    "for_each_neighbor_body(binder={}, body=stmts#{}, r={})",
                    binder, body.0, radius_cells
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CgStmtList
// ---------------------------------------------------------------------------

/// Ordered list of statement ids — one body in the program. The list
/// arena (in `CgProgram`, Task 1.5) holds many of these, indexed by
/// [`CgStmtListId`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CgStmtList {
    pub stmts: Vec<CgStmtId>,
}

impl CgStmtList {
    pub fn new(stmts: Vec<CgStmtId>) -> Self {
        Self { stmts }
    }

    pub fn is_empty(&self) -> bool {
        self.stmts.is_empty()
    }

    pub fn len(&self) -> usize {
        self.stmts.len()
    }
}

impl fmt::Display for CgStmtList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, id) in self.stmts.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "stmt#{}", id.0)?;
        }
        f.write_str("]")
    }
}

// ---------------------------------------------------------------------------
// StmtArena + ListArena
// ---------------------------------------------------------------------------

/// Resolves a [`CgStmtId`] to its underlying [`CgStmt`]. Sibling to
/// [`crate::cg::expr::ExprArena`]. Implementations:
/// - `CgProgram` (Task 1.5) implements it against its `Vec<CgStmt>`
///   arena.
/// - `[CgStmt]` and `Vec<CgStmt>` impls below let tests carry a tiny
///   arena directly.
///
/// `Option<&CgStmt>` keeps arena lookups panic-free; out-of-range ids
/// return `None` and the caller decides whether that's a defect (the
/// well-formed pass) or the end of a walk.
pub trait StmtArena {
    fn get(&self, id: CgStmtId) -> Option<&CgStmt>;
}

impl StmtArena for [CgStmt] {
    fn get(&self, id: CgStmtId) -> Option<&CgStmt> {
        <[CgStmt]>::get(self, id.0 as usize)
    }
}

impl StmtArena for Vec<CgStmt> {
    fn get(&self, id: CgStmtId) -> Option<&CgStmt> {
        <[CgStmt]>::get(self.as_slice(), id.0 as usize)
    }
}

/// Resolves a [`CgStmtListId`] to its underlying [`CgStmtList`]. Same
/// shape as [`StmtArena`] but for list ids — kept separate so the
/// program can store statement nodes and statement lists in different
/// arenas (which `CgProgram` does — see Task 1.5).
pub trait StmtListArena {
    fn get(&self, id: CgStmtListId) -> Option<&CgStmtList>;
}

impl StmtListArena for [CgStmtList] {
    fn get(&self, id: CgStmtListId) -> Option<&CgStmtList> {
        <[CgStmtList]>::get(self, id.0 as usize)
    }
}

impl StmtListArena for Vec<CgStmtList> {
    fn get(&self, id: CgStmtListId) -> Option<&CgStmtList> {
        <[CgStmtList]>::get(self.as_slice(), id.0 as usize)
    }
}

// ---------------------------------------------------------------------------
// Dependency walks
// ---------------------------------------------------------------------------

/// Walk an expression tree rooted at `id` and append every
/// `CgExpr::Read(handle)` it contains into `out`. Order is depth-first
/// in operand order (lhs before rhs, args left-to-right, condition
/// before then before else), which makes the walk deterministic — the
/// same input arena always produces the same output ordering, including
/// duplicates.
pub fn collect_expr_reads(id: CgExprId, exprs: &dyn ExprArena, out: &mut Vec<DataHandle>) {
    // Out-of-range id (corrupted arena): the auto-walker skips silently.
    // The well-formed pass reports the defect through its dedicated
    // expression-id-range walk; mirroring it here would double-report.
    let Some(node) = exprs.get(id) else {
        return;
    };
    match node {
        CgExpr::Read(h) => {
            out.push(h.clone());
            // Slice 1 (stdlib-into-CG-IR) follow-up: an `AgentField`
            // handle whose `target` is `AgentRef::Target(expr_id)`
            // carries an embedded expression — the per-thread index
            // into the SoA. The kernel binding-scanner consumes
            // `op.reads` to decide which `agent_<field>` buffers a
            // kernel binds; if we don't recurse into the target
            // expression here, any AgentField reads it contains
            // (e.g. `agents.pos(self.engaged_with)` → the inner
            // `Read(AgentField{EngagedWith, Self_})`) won't appear
            // in `op.reads`, and the kernel would reference an
            // undeclared `agent_engaged_with` binding at WGSL
            // validation time. Mirrors the slice-1 emit-side
            // hoisting: every `Target(_)` reference contributes
            // its target expression's reads to the op's binding
            // surface.
            if let DataHandle::AgentField {
                target: crate::cg::data_handle::AgentRef::Target(target_expr_id),
                ..
            } = h
            {
                collect_expr_reads(*target_expr_id, exprs, out);
            }
        }
        CgExpr::Lit(_) => {}
        CgExpr::Binary { lhs, rhs, .. } => {
            collect_expr_reads(*lhs, exprs, out);
            collect_expr_reads(*rhs, exprs, out);
        }
        CgExpr::Unary { arg, .. } => {
            collect_expr_reads(*arg, exprs, out);
        }
        CgExpr::Builtin { args, .. } => {
            for a in args {
                collect_expr_reads(*a, exprs, out);
            }
        }
        CgExpr::Rng { purpose, .. } => {
            // RNG draws are reads of the deterministic per-agent stream;
            // record the handle so dependency analysis knows the op
            // touches the seed.
            out.push(DataHandle::Rng { purpose: *purpose });
        }
        CgExpr::Select {
            cond, then, else_, ..
        } => {
            collect_expr_reads(*cond, exprs, out);
            collect_expr_reads(*then, exprs, out);
            collect_expr_reads(*else_, exprs, out);
        }
        CgExpr::AgentSelfId
        | CgExpr::PerPairCandidateId
        | CgExpr::ReadLocal { .. }
        | CgExpr::EventField { .. }
        | CgExpr::NamespaceField { .. } => {
            // Bare actor / candidate id reads + let-bound local reads
            // do not contribute structural reads of any persisted
            // `DataHandle`. The actor / candidate slot ids are
            // implicit in the dispatch shape; the let-bound local
            // lives in the surrounding body's scope. EventField
            // payload reads are implicit in the dispatch shape's
            // `source_ring` (the surrounding `DispatchShape::PerEvent`)
            // — well_formed flags `EventField` in any other dispatch
            // shape, so the read is structurally accounted for.
            // NamespaceField reads (e.g. `world.tick`) resolve to
            // either kernel-preamble locals or uniform-bound fields
            // per the `WgslAccessForm` registry — neither contributes
            // a `DataHandle::*` read.
        }
        CgExpr::NamespaceCall { args, .. } => {
            // Recurse into argument expressions; the call itself does
            // not read any `DataHandle::*` (the WGSL emit produces a
            // prelude function call whose body operates on its
            // argument values).
            for a in args {
                collect_expr_reads(*a, exprs, out);
            }
        }
    }
}

/// Walk a single statement and append its (reads, writes) into the
/// supplied buffers. For nested `If` bodies the walk recurses through
/// `lists` to fetch the sub-`CgStmtList`s.
///
/// Order: depth-first in source-code order. `Assign` records the RHS
/// expression's reads first, then the assignment's target as a write.
/// `Emit` records each field expression's reads in field order; the
/// destination ring write is **not** synthesized here — the
/// [`EventKindId`] alone doesn't determine which ring stores the
/// event (rings are an emit-time concept resolved by lowering against
/// the event registry). The AST → HIR lowering pass adds the ring
/// write to the enclosing op's `writes` via
/// [`crate::cg::op::ComputeOp::record_write`]. `If` records the
/// condition's reads, then walks the then-arm (in order), then the
/// else-arm (if any).
pub fn collect_stmt_dependencies(
    id: CgStmtId,
    exprs: &dyn ExprArena,
    stmts: &dyn StmtArena,
    lists: &dyn StmtListArena,
    reads: &mut Vec<DataHandle>,
    writes: &mut Vec<DataHandle>,
) {
    // Out-of-range stmt id: skip silently — the well-formed pass owns
    // the structural id-range diagnostic.
    let Some(node) = stmts.get(id) else {
        return;
    };
    match node {
        CgStmt::Assign { target, value } => {
            collect_expr_reads(*value, exprs, reads);
            writes.push(target.clone());
        }
        CgStmt::Emit { event: _, fields } => {
            for (_, expr_id) in fields {
                collect_expr_reads(*expr_id, exprs, reads);
            }
            // No synthesized ring write here — see the doc comment
            // above for the rationale.
        }
        CgStmt::If { cond, then, else_ } => {
            collect_expr_reads(*cond, exprs, reads);
            collect_list_dependencies(*then, exprs, stmts, lists, reads, writes);
            if let Some(else_id) = else_ {
                collect_list_dependencies(*else_id, exprs, stmts, lists, reads, writes);
            }
        }
        CgStmt::Match { scrutinee, arms } => {
            // Match: walk the scrutinee for reads, then descend into
            // each arm's body (in source order) for the union of
            // their reads/writes. Arm bindings introduce locals that
            // are scoped to the arm body — they don't surface as a
            // structural read here (the body's expression walks pick
            // up references to them once expression lowering grows
            // local-binding support; today that path errors out).
            collect_expr_reads(*scrutinee, exprs, reads);
            for arm in arms {
                collect_list_dependencies(arm.body, exprs, stmts, lists, reads, writes);
            }
        }
        CgStmt::Let { value, .. } => {
            // Let: walk the bound value expression for reads. The
            // local binding itself is scoped within the enclosing
            // op's body — locals don't surface as a structural read
            // of any persisted data handle. Expression-level
            // references to the local resolve once `IrExpr::Local`
            // resolution lands (Task 5.5d); until then this walker
            // only contributes the value's reads.
            collect_expr_reads(*value, exprs, reads);
        }
        CgStmt::ForEachAgent {
            init, projection, ..
        } => {
            // Walks every alive agent slot and accumulates a
            // projection. The projection's AgentField reads are
            // captured via collect_expr_reads; the walk itself reads
            // `agent_pos[agent_id]` for the bounds check, but that's
            // already declared elsewhere by the enclosing per-agent
            // op (it reads agent_pos for agent_id resolution).
            collect_expr_reads(*init, exprs, reads);
            collect_expr_reads(*projection, exprs, reads);
        }
        CgStmt::ForEachNeighbor {
            init, projection, ..
        } => {
            // Same shape as ForEachAgent for expr reads, plus three
            // structural reads for the spatial bindings the WGSL
            // emit walks: `spatial_grid_starts[cell..cell+1]` for
            // the start/end of each cell's slice in
            // `spatial_grid_cells`, and `spatial_grid_offsets`
            // (still bound, atomic-counted in the build, used by
            // the diagnostic kernel and as the cooperative-load
            // gate). Surfacing them here is what tells the kernel-
            // emit's BGL composer to bind the spatial buffers AND
            // tells the schedule synthesizer this op is a spatial-
            // query consumer (so the three counting-sort phases
            // get dispatched before it).
            collect_expr_reads(*init, exprs, reads);
            collect_expr_reads(*projection, exprs, reads);
            reads.push(DataHandle::SpatialStorage {
                kind: super::data_handle::SpatialStorageKind::GridCells,
            });
            reads.push(DataHandle::SpatialStorage {
                kind: super::data_handle::SpatialStorageKind::GridOffsets,
            });
            reads.push(DataHandle::SpatialStorage {
                kind: super::data_handle::SpatialStorageKind::GridStarts,
            });
        }
        CgStmt::ForEachNeighborBody { body, .. } => {
            // Body-form spatial walk. The body's own statements
            // contribute reads/writes via the recursive list walk
            // (e.g. an inner Emit's field-value reads + the host-
            // recorded ring write). The walker must ALSO surface
            // the three spatial bindings the WGSL emit reads —
            // mirrors `ForEachNeighbor`'s structural surface so
            // the BGL composer + schedule synthesizer treat both
            // forms identically.
            collect_list_dependencies(*body, exprs, stmts, lists, reads, writes);
            reads.push(DataHandle::SpatialStorage {
                kind: super::data_handle::SpatialStorageKind::GridCells,
            });
            reads.push(DataHandle::SpatialStorage {
                kind: super::data_handle::SpatialStorageKind::GridOffsets,
            });
            reads.push(DataHandle::SpatialStorage {
                kind: super::data_handle::SpatialStorageKind::GridStarts,
            });
            // The WGSL template (`emit_for_each_neighbor_body` in
            // emit/wgsl_body.rs) reads `agent_pos[agent_id]` to compute
            // `_self_cell_f` regardless of whether the body itself
            // references `self.pos`. Surface the implicit self.pos read
            // here so the BGL composer binds `agent_pos` to the kernel.
            // Without this, fixtures whose body-form walk doesn't
            // explicitly reference `self.pos` (e.g. `duel_25v25` —
            // walks neighbours to emit damage but never reads its own
            // position) emit naga-invalid WGSL that references an
            // unbound `agent_pos` identifier.
            reads.push(DataHandle::AgentField {
                field: super::data_handle::AgentFieldId::Pos,
                target: super::data_handle::AgentRef::Self_,
            });
        }
    }
}

/// Walk every statement in a [`CgStmtList`] in order and append the
/// union of their (reads, writes).
pub fn collect_list_dependencies(
    list_id: CgStmtListId,
    exprs: &dyn ExprArena,
    stmts: &dyn StmtArena,
    lists: &dyn StmtListArena,
    reads: &mut Vec<DataHandle>,
    writes: &mut Vec<DataHandle>,
) {
    // Out-of-range list id: skip silently — the well-formed pass owns
    // the structural id-range diagnostic.
    let Some(list) = lists.get(list_id) else {
        return;
    };
    for stmt_id in &list.stmts {
        collect_stmt_dependencies(*stmt_id, exprs, stmts, lists, reads, writes);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{AgentFieldId, AgentRef, CgExprId, RngPurpose};
    use crate::cg::expr::{BinaryOp, CgExpr, CgTy, LitValue};
    use crate::cg::op::EventKindId;

    fn assert_roundtrip<T>(v: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        let json = serde_json::to_string(v).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&back, v, "round-trip changed value (json was {json})");
    }

    fn read_self_hp() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        })
    }

    fn read_self_pos() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Pos,
            target: AgentRef::Self_,
        })
    }

    fn lit_f32(v: f32) -> CgExpr {
        CgExpr::Lit(LitValue::F32(v))
    }

    fn lit_bool(b: bool) -> CgExpr {
        CgExpr::Lit(LitValue::Bool(b))
    }

    // ---- newtype roundtrips ----

    #[test]
    fn cg_stmt_id_roundtrip() {
        assert_roundtrip(&CgStmtId(0));
        assert_roundtrip(&CgStmtId(42));
    }

    #[test]
    fn cg_stmt_list_id_roundtrip() {
        assert_roundtrip(&CgStmtListId(0));
        assert_roundtrip(&CgStmtListId(7));
    }

    #[test]
    fn event_field_display_and_roundtrip() {
        let ef = EventField {
            event: EventKindId(2),
            index: 3,
        };
        assert_eq!(format!("{}", ef), "event[#2].field#3");
        assert_roundtrip(&ef);
    }

    // ---- CgStmt display + roundtrip ----

    #[test]
    fn assign_display_and_roundtrip() {
        let s = CgStmt::Assign {
            target: DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            },
            value: CgExprId(4),
        };
        assert_eq!(format!("{}", s), "assign(agent.self.hp <- expr#4)");
        assert_roundtrip(&s);
    }

    #[test]
    fn emit_display_and_roundtrip() {
        let s = CgStmt::Emit {
            event: EventKindId(1),
            fields: vec![
                (
                    EventField {
                        event: EventKindId(1),
                        index: 0,
                    },
                    CgExprId(2),
                ),
                (
                    EventField {
                        event: EventKindId(1),
                        index: 1,
                    },
                    CgExprId(3),
                ),
            ],
        };
        assert_eq!(
            format!("{}", s),
            "emit(event[#1], event[#1].field#0=expr#2, event[#1].field#1=expr#3)"
        );
        assert_roundtrip(&s);
    }

    #[test]
    fn if_with_else_display_and_roundtrip() {
        let s = CgStmt::If {
            cond: CgExprId(1),
            then: CgStmtListId(2),
            else_: Some(CgStmtListId(3)),
        };
        assert_eq!(
            format!("{}", s),
            "if(cond=expr#1, then=stmts#2, else=stmts#3)"
        );
        assert_roundtrip(&s);
    }

    #[test]
    fn if_without_else_display_and_roundtrip() {
        let s = CgStmt::If {
            cond: CgExprId(0),
            then: CgStmtListId(1),
            else_: None,
        };
        assert_eq!(format!("{}", s), "if(cond=expr#0, then=stmts#1)");
        assert_roundtrip(&s);
    }

    // ---- CgStmtList display + roundtrip ----

    #[test]
    fn stmt_list_empty_display() {
        let l = CgStmtList::new(vec![]);
        assert_eq!(format!("{}", l), "[]");
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
        assert_roundtrip(&l);
    }

    #[test]
    fn stmt_list_two_entries_display() {
        let l = CgStmtList::new(vec![CgStmtId(0), CgStmtId(1)]);
        assert_eq!(format!("{}", l), "[stmt#0, stmt#1]");
        assert_eq!(l.len(), 2);
        assert!(!l.is_empty());
        assert_roundtrip(&l);
    }

    // ---- Arena impls ----

    #[test]
    fn stmt_arena_resolves_by_id() {
        let arena: Vec<CgStmt> = vec![
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(0),
            },
            CgStmt::If {
                cond: CgExprId(1),
                then: CgStmtListId(0),
                else_: None,
            },
        ];
        let s0 = StmtArena::get(&arena, CgStmtId(0)).expect("stmt#0 resolves");
        match s0 {
            CgStmt::Assign { value, .. } => assert_eq!(*value, CgExprId(0)),
            _ => panic!("expected Assign at #0"),
        }
        let s1 = StmtArena::get(&arena, CgStmtId(1)).expect("stmt#1 resolves");
        match s1 {
            CgStmt::If { cond, .. } => assert_eq!(*cond, CgExprId(1)),
            _ => panic!("expected If at #1"),
        }
    }

    #[test]
    fn stmt_list_arena_resolves_by_id() {
        let arena: Vec<CgStmtList> = vec![
            CgStmtList::new(vec![]),
            CgStmtList::new(vec![CgStmtId(7)]),
        ];
        let l0 = StmtListArena::get(&arena, CgStmtListId(0)).expect("list#0 resolves");
        assert!(l0.is_empty());
        let l1 = StmtListArena::get(&arena, CgStmtListId(1)).expect("list#1 resolves");
        assert_eq!(l1.stmts, vec![CgStmtId(7)]);
    }

    // ---- Dependency walks ----

    #[test]
    fn collect_expr_reads_lit_is_empty() {
        let exprs: Vec<CgExpr> = vec![lit_f32(1.0)];
        let mut out = Vec::new();
        collect_expr_reads(CgExprId(0), &exprs, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn collect_expr_reads_binary_lhs_then_rhs() {
        // (hp + 1.0) — reads = [agent.self.hp]
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(1.0),   // 1
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::F32,
            }, // 2
        ];
        let mut out = Vec::new();
        collect_expr_reads(CgExprId(2), &exprs, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0],
            DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }
        );
    }

    #[test]
    fn collect_expr_reads_select_walks_all_three_arms() {
        // select(true, hp, pos) -- pos is wrong type for then/else, but
        // the read collector doesn't care about types.
        let exprs: Vec<CgExpr> = vec![
            lit_bool(true), // 0
            read_self_hp(), // 1
            read_self_pos(),// 2
            CgExpr::Select {
                cond: CgExprId(0),
                then: CgExprId(1),
                else_: CgExprId(2),
                ty: CgTy::F32,
            }, // 3
        ];
        let mut out = Vec::new();
        collect_expr_reads(CgExprId(3), &exprs, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }
        );
        assert_eq!(
            out[1],
            DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }
        );
    }

    #[test]
    fn collect_expr_reads_rng_records_handle() {
        let exprs: Vec<CgExpr> = vec![CgExpr::Rng {
            purpose: RngPurpose::Action,
            ty: CgTy::U32,
        }];
        let mut out = Vec::new();
        collect_expr_reads(CgExprId(0), &exprs, &mut out);
        assert_eq!(
            out,
            vec![DataHandle::Rng {
                purpose: RngPurpose::Action,
            }]
        );
    }

    #[test]
    fn collect_stmt_dependencies_assign_reads_value_writes_target() {
        // assign(agent.self.hp <- (agent.self.hp + 1.0))
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(1.0),   // 1
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::F32,
            }, // 2
        ];
        let stmts: Vec<CgStmt> = vec![CgStmt::Assign {
            target: DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            },
            value: CgExprId(2),
        }];
        let lists: Vec<CgStmtList> = vec![];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_stmt_dependencies(CgStmtId(0), &exprs, &stmts, &lists, &mut reads, &mut writes);
        assert_eq!(
            reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(
            writes,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
    }

    #[test]
    fn collect_stmt_dependencies_emit_walks_field_value_reads_only() {
        // The walker descends into each field-value expression for
        // reads, but does NOT synthesize a destination event-ring
        // write — ring binding is a lowering concern (see the doc
        // comment on `collect_stmt_dependencies`).
        let exprs: Vec<CgExpr> = vec![read_self_hp(), lit_f32(0.0)];
        let stmts: Vec<CgStmt> = vec![CgStmt::Emit {
            event: EventKindId(7),
            fields: vec![
                (
                    EventField {
                        event: EventKindId(7),
                        index: 0,
                    },
                    CgExprId(0),
                ),
                (
                    EventField {
                        event: EventKindId(7),
                        index: 1,
                    },
                    CgExprId(1),
                ),
            ],
        }];
        let lists: Vec<CgStmtList> = vec![];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_stmt_dependencies(CgStmtId(0), &exprs, &stmts, &lists, &mut reads, &mut writes);
        // Reads: hp from the first field's value expr.
        assert_eq!(reads.len(), 1);
        assert!(reads.contains(&DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        }));
        // Writes: none — the `Emit`'s ring binding is added by lowering.
        assert!(writes.is_empty(), "expected no writes, got {writes:?}");
    }

    #[test]
    fn collect_stmt_dependencies_if_walks_both_arms() {
        // if (true) { assign(hp <- 1.0) } else { assign(hp <- 0.0) }
        let exprs: Vec<CgExpr> = vec![
            lit_bool(true), // 0
            lit_f32(1.0),   // 1
            lit_f32(0.0),   // 2
        ];
        let stmts: Vec<CgStmt> = vec![
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(1),
            }, // 0
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(2),
            }, // 1
            CgStmt::If {
                cond: CgExprId(0),
                then: CgStmtListId(0),
                else_: Some(CgStmtListId(1)),
            }, // 2
        ];
        let lists: Vec<CgStmtList> = vec![
            CgStmtList::new(vec![CgStmtId(0)]),
            CgStmtList::new(vec![CgStmtId(1)]),
        ];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_stmt_dependencies(CgStmtId(2), &exprs, &stmts, &lists, &mut reads, &mut writes);
        // Both arms write hp; cond is a literal so no reads from there.
        assert!(reads.is_empty());
        assert_eq!(writes.len(), 2);
    }

    // ---- CgStmt::Match Display + roundtrip + walker ----

    #[test]
    fn variant_id_display_and_roundtrip() {
        assert_eq!(format!("{}", VariantId(7)), "variant#7");
        assert_roundtrip(&VariantId(0));
        assert_roundtrip(&VariantId(42));
    }

    #[test]
    fn local_id_display_and_roundtrip() {
        assert_eq!(format!("{}", LocalId(3)), "local#3");
        assert_roundtrip(&LocalId(0));
        assert_roundtrip(&LocalId(11));
    }

    #[test]
    fn match_arm_binding_display_and_roundtrip() {
        let b = MatchArmBinding {
            field_name: "amount".to_string(),
            local: LocalId(2),
        };
        assert_eq!(format!("{}", b), "amount=local#2");
        assert_roundtrip(&b);
    }

    #[test]
    fn cg_match_arm_display_and_roundtrip() {
        let arm = CgMatchArm {
            variant: VariantId(1),
            bindings: vec![
                MatchArmBinding {
                    field_name: "amount".to_string(),
                    local: LocalId(0),
                },
                MatchArmBinding {
                    field_name: "factor_q8".to_string(),
                    local: LocalId(1),
                },
            ],
            body: CgStmtListId(3),
        };
        assert_eq!(
            format!("{}", arm),
            "arm(variant#1, [amount=local#0, factor_q8=local#1], body=stmts#3)"
        );
        assert_roundtrip(&arm);
    }

    #[test]
    fn match_stmt_display_and_roundtrip() {
        let s = CgStmt::Match {
            scrutinee: CgExprId(5),
            arms: vec![
                CgMatchArm {
                    variant: VariantId(0),
                    bindings: vec![MatchArmBinding {
                        field_name: "amount".to_string(),
                        local: LocalId(0),
                    }],
                    body: CgStmtListId(1),
                },
                CgMatchArm {
                    variant: VariantId(1),
                    bindings: vec![],
                    body: CgStmtListId(2),
                },
            ],
        };
        assert_eq!(
            format!("{}", s),
            "match(scrutinee=expr#5, arms=[arm(variant#0, [amount=local#0], body=stmts#1), arm(variant#1, [], body=stmts#2)])"
        );
        assert_roundtrip(&s);
    }

    #[test]
    fn let_stmt_display_and_roundtrip() {
        // CgStmt::Let { local, value, ty } pretty-prints in the canonical
        // `let(<local>: <ty> = expr#<id>)` form and round-trips through
        // serde without losing payload.
        let s = CgStmt::Let {
            local: LocalId(7),
            value: CgExprId(11),
            ty: CgTy::F32,
        };
        assert_eq!(format!("{}", s), "let(local#7: f32 = expr#11)");
        assert_roundtrip(&s);
    }

    #[test]
    fn collect_stmt_dependencies_let_walks_value_expression() {
        // let local#0: f32 = (agent.self.hp + 1.0)
        // Walker: Let.value's expression-tree reads → [agent.self.hp].
        // The local binding itself is scoped — no persisted-handle
        // read or write for it.
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(1.0),   // 1
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::F32,
            }, // 2
        ];
        let stmts: Vec<CgStmt> = vec![CgStmt::Let {
            local: LocalId(0),
            value: CgExprId(2),
            ty: CgTy::F32,
        }];
        let lists: Vec<CgStmtList> = vec![];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_stmt_dependencies(CgStmtId(0), &exprs, &stmts, &lists, &mut reads, &mut writes);
        assert_eq!(
            reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        // No writes — `Let` introduces a local binding, no persisted
        // data handle is mutated.
        assert!(writes.is_empty(), "expected no writes, got {writes:?}");
    }

    #[test]
    fn collect_stmt_dependencies_match_walks_scrutinee_and_arms() {
        // match scrutinee=hp { variant#0 => emit(event#3) }
        //
        // Scrutinee read of hp + the arm body's emit field-expr reads
        // (none here — the emit has no fields). No writes (Emit's
        // ring binding is added by lowering, not by the auto-walker).
        let exprs: Vec<CgExpr> = vec![read_self_hp()];
        let stmts: Vec<CgStmt> = vec![
            // 0: arm body's emit
            CgStmt::Emit {
                event: EventKindId(3),
                fields: vec![],
            },
            // 1: top-level match
            CgStmt::Match {
                scrutinee: CgExprId(0),
                arms: vec![CgMatchArm {
                    variant: VariantId(0),
                    bindings: vec![],
                    body: CgStmtListId(0),
                }],
            },
        ];
        let lists: Vec<CgStmtList> = vec![CgStmtList::new(vec![CgStmtId(0)])];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_stmt_dependencies(CgStmtId(1), &exprs, &stmts, &lists, &mut reads, &mut writes);
        assert_eq!(
            reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        assert!(writes.is_empty(), "Emit's ring write is not auto-derived");
    }

    #[test]
    fn collect_stmt_dependencies_match_unions_arm_writes() {
        // match scrutinee { variant#0 => assign(hp <- 1.0),
        //                   variant#1 => assign(pos <- ...) }
        // Each arm contributes a write — the walker reports the union
        // in source order (arm 0 first, arm 1 second).
        let exprs: Vec<CgExpr> = vec![
            lit_f32(0.0),    // 0  scrutinee — placeholder lit
            lit_f32(1.0),    // 1  arm0 value
            read_self_pos(), // 2  arm1 value
        ];
        let stmts: Vec<CgStmt> = vec![
            // 0: arm0 body — assign hp
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(1),
            },
            // 1: arm1 body — assign pos
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Pos,
                    target: AgentRef::Self_,
                },
                value: CgExprId(2),
            },
            // 2: top-level match
            CgStmt::Match {
                scrutinee: CgExprId(0),
                arms: vec![
                    CgMatchArm {
                        variant: VariantId(0),
                        bindings: vec![],
                        body: CgStmtListId(0),
                    },
                    CgMatchArm {
                        variant: VariantId(1),
                        bindings: vec![],
                        body: CgStmtListId(1),
                    },
                ],
            },
        ];
        let lists: Vec<CgStmtList> = vec![
            CgStmtList::new(vec![CgStmtId(0)]),
            CgStmtList::new(vec![CgStmtId(1)]),
        ];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_stmt_dependencies(CgStmtId(2), &exprs, &stmts, &lists, &mut reads, &mut writes);
        // Reads: arm1's pos read; arm0's value is a lit. Scrutinee is
        // also a lit so contributes nothing.
        assert_eq!(
            reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }]
        );
        // Writes: arm0 hp first, then arm1 pos.
        assert_eq!(writes.len(), 2);
        assert_eq!(
            writes[0],
            DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }
        );
        assert_eq!(
            writes[1],
            DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }
        );
    }

    #[test]
    fn collect_list_dependencies_walks_in_order() {
        // [assign(hp <- 1.0), emit(event#3)]
        let exprs: Vec<CgExpr> = vec![
            lit_f32(1.0),   // 0
            read_self_hp(), // 1
        ];
        let stmts: Vec<CgStmt> = vec![
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(0),
            }, // 0
            CgStmt::Emit {
                event: EventKindId(3),
                fields: vec![(
                    EventField {
                        event: EventKindId(3),
                        index: 0,
                    },
                    CgExprId(1),
                )],
            }, // 1
        ];
        let lists: Vec<CgStmtList> = vec![CgStmtList::new(vec![CgStmtId(0), CgStmtId(1)])];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_list_dependencies(CgStmtListId(0), &exprs, &stmts, &lists, &mut reads, &mut writes);
        // Reads: emit's field expr reads hp.
        assert_eq!(reads.len(), 1);
        // Writes: just assign's hp — the emit's destination ring is
        // bound by lowering, not by the auto-walker.
        assert_eq!(writes.len(), 1);
        match &writes[0] {
            DataHandle::AgentField { field, .. } => assert_eq!(*field, AgentFieldId::Hp),
            _ => panic!("expected hp write"),
        }
    }
}
