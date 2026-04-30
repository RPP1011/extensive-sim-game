//! `CgProgram` — top-level container for the Compute-Graph IR.
//!
//! `CgProgram` owns every arena the IR layer indexes into: the op
//! arena, the expression arena, the statement arena, and the statement
//! list arena. It also owns the [`Interner`] (stable id → human-readable
//! name lookup, for diagnostics) and the typed [`CgDiagnostic`] log
//! produced by lowering and well-formedness passes.
//!
//! Every consumer of the IR — well-formed checks (Task 1.6), HIR/MIR
//! lowering, schedule synthesis, emit — walks a `CgProgram`. The struct
//! is the single point of truth for "what compute did the DSL produce
//! this build?".
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 1.5, for the design rationale.
//!
//! # Construction
//!
//! Build a program with [`CgProgramBuilder`]. The builder allocates ids
//! contiguously from zero and validates every reference at insertion
//! time — adding a node that references an out-of-range id returns a
//! typed [`BuilderError`], not a runtime panic at consumption time.
//! The arenas on [`CgProgram`] are `pub` so passes can iterate them
//! without an accessor wall, but **direct mutation of the arenas
//! bypasses the builder's invariants** — only mutate via the builder
//! during construction, or via [`ComputeOp::record_read`] /
//! [`ComputeOp::record_write`] (the documented post-construction seams)
//! during lowering.
//!
//! # `Vec<CgStmt>` arena
//!
//! The plan body lists four arenas (`ops`, `exprs`, `stmt_lists`,
//! diagnostics) but `CgStmtList` references statements by [`CgStmtId`]
//! which indexes into a `Vec<CgStmt>`. The plan's omission of the
//! statement arena is editorial — it is required for the program to
//! resolve `CgStmtListId → CgStmtList → CgStmtId → CgStmt`. We add the
//! `stmts: Vec<CgStmt>` arena here and document its presence at the
//! field level.
//!
//! # `record_read` / `record_write`
//!
//! Lowering passes that need to inject a registry-resolved binding
//! (source event ring read for a [`DispatchShape::PerEvent`] op, ring
//! write for an `Emit`, etc.) reach into `program.ops[op_id.0 as
//! usize]` and call [`ComputeOp::record_read`] /
//! [`ComputeOp::record_write`] **after** the op has been added via
//! [`CgProgramBuilder::add_op`]. This is the documented post-
//! construction seam for the parts of the dependency graph the
//! auto-walker can't synthesize structurally.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::data_handle::{
    CgExprId, ConfigConstId, DataHandle, EventRingId, MaskId, ViewId,
};
use super::dispatch::DispatchShape;
use super::expr::{CgExpr, ExprArena};
use super::op::{
    ActionId, ComputeOp, ComputeOpKind, EventKindId, OpId, PhysicsRuleId, ScoringId, Span,
};
use super::stmt::{CgStmt, CgStmtId, CgStmtList, CgStmtListId, StmtArena, StmtListArena};

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Diagnostic severity. The IR layer only emits non-fatal kinds —
/// errors that block compilation are returned as `Result<_, Vec<…>>`
/// from the well-formed pass (Task 1.6) rather than logged here.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Information — lowering observed a structural curiosity worth
    /// surfacing, but compilation proceeds normally.
    Note,
    /// Warning — lowering detected a likely defect (orphaned emit,
    /// unread view) that does not block compilation.
    Warning,
    /// Lint — style or efficiency suggestion. Currently unused; kept
    /// distinct from `Warning` so future fusion-suggestion passes have
    /// a non-warning channel.
    Lint,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Note => f.write_str("note"),
            Severity::Warning => f.write_str("warning"),
            Severity::Lint => f.write_str("lint"),
        }
    }
}

// ---------------------------------------------------------------------------
// CgDiagnosticKind — typed diagnostic payloads
// ---------------------------------------------------------------------------

/// Typed payload for a [`CgDiagnostic`].
///
/// Each variant captures the structural information the diagnostic
/// reports — no `String` reasons. New diagnostic kinds add a variant
/// here.
///
/// **Production of these diagnostics is deferred to the well-formed
/// pass (Task 1.6) and lowering passes (Phase 2);** Task 1.5 only
/// defines the type and the storage. The [`CgProgramBuilder::add_diagnostic`]
/// entry point exists so those later passes have a place to push.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CgDiagnosticKind {
    /// An `Emit` statement references an event kind that no event-ring
    /// binding ever reads. Flagged for lowering review — likely a
    /// dead-end emit.
    OrphanedEmit { event: EventKindId },
    /// A view storage slot is read but never written. Likely a fold
    /// rule was forgotten.
    UnreadView { view: ViewId },
    /// A mask is computed but never read. Dead op — fusion / DCE
    /// candidate.
    UnusedMask { mask: MaskId },
    /// Multiple ops claim to write the same `DataHandle` without an
    /// explicit ordering. Schedule synthesis will need to serialize
    /// these.
    WriteConflict {
        handle: DataHandle,
        ops: Vec<OpId>,
    },
}

impl fmt::Display for CgDiagnosticKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CgDiagnosticKind::OrphanedEmit { event } => {
                write!(f, "orphaned_emit(event=#{})", event.0)
            }
            CgDiagnosticKind::UnreadView { view } => {
                write!(f, "unread_view(view=#{})", view.0)
            }
            CgDiagnosticKind::UnusedMask { mask } => {
                write!(f, "unused_mask(mask=#{})", mask.0)
            }
            CgDiagnosticKind::WriteConflict { handle, ops } => {
                write!(f, "write_conflict(handle={}, ops=[", handle)?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "op#{}", op.0)?;
                }
                f.write_str("])")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CgDiagnostic
// ---------------------------------------------------------------------------

/// A single IR-level diagnostic. Severity + optional source span +
/// typed kind. The `kind` field is the single source of truth for the
/// diagnostic's structural meaning; severity affects only how callers
/// surface the entry.
///
/// `span` is `serde(skip)`'d for the same reason
/// [`super::op::ComputeOp::span`] is — the AST `Span` does not implement
/// `Deserialize`. Round-tripping a diagnostic resets `span` to
/// `None`; structural fields (`severity`, `kind`) round-trip intact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgDiagnostic {
    pub severity: Severity,
    /// Source location, if the diagnostic has a single anchor. Some
    /// diagnostics are program-wide (e.g. a `WriteConflict` spanning
    /// multiple ops) and carry `None`.
    #[serde(skip)]
    pub span: Option<Span>,
    pub kind: CgDiagnosticKind,
}

impl CgDiagnostic {
    /// Construct a diagnostic with no source span.
    pub fn new(severity: Severity, kind: CgDiagnosticKind) -> Self {
        Self {
            severity,
            span: None,
            kind,
        }
    }

    /// Construct a diagnostic anchored at a specific source span.
    pub fn with_span(severity: Severity, span: Span, kind: CgDiagnosticKind) -> Self {
        Self {
            severity,
            span: Some(span),
            kind,
        }
    }
}

impl fmt::Display for CgDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.severity, self.kind)
    }
}

// ---------------------------------------------------------------------------
// Interner — stable id → human-readable name tables
// ---------------------------------------------------------------------------

/// Stable interned ids → human-readable names, used by diagnostics and
/// by [`CgProgram::display_with_names`] to render `DataHandle`s with
/// source-level names instead of opaque `#N` ids.
///
/// Per-id-type tables. Adding a new id type adds a `BTreeMap` field
/// here plus a `get_*_name` / `intern_*_name` pair. `BTreeMap` keys keep
/// the [`fmt::Display`] output deterministic (sorted by id) without an
/// explicit sort step.
///
/// Entries are populated by lowering passes via the builder's
/// `intern_*_name` methods; an id with no entry is rendered in `#N`
/// form by [`CgProgram::display_with_names`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Interner {
    pub views: BTreeMap<u32, String>,
    pub masks: BTreeMap<u32, String>,
    pub scorings: BTreeMap<u32, String>,
    pub physics_rules: BTreeMap<u32, String>,
    pub event_kinds: BTreeMap<u32, String>,
    pub actions: BTreeMap<u32, String>,
    pub config_consts: BTreeMap<u32, String>,
    pub event_rings: BTreeMap<u32, String>,
}

impl Interner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_view_name(&self, id: ViewId) -> Option<&str> {
        self.views.get(&id.0).map(String::as_str)
    }
    pub fn get_mask_name(&self, id: MaskId) -> Option<&str> {
        self.masks.get(&id.0).map(String::as_str)
    }
    pub fn get_scoring_name(&self, id: ScoringId) -> Option<&str> {
        self.scorings.get(&id.0).map(String::as_str)
    }
    pub fn get_physics_rule_name(&self, id: PhysicsRuleId) -> Option<&str> {
        self.physics_rules.get(&id.0).map(String::as_str)
    }
    pub fn get_event_kind_name(&self, id: EventKindId) -> Option<&str> {
        self.event_kinds.get(&id.0).map(String::as_str)
    }
    pub fn get_action_name(&self, id: ActionId) -> Option<&str> {
        self.actions.get(&id.0).map(String::as_str)
    }
    pub fn get_config_const_name(&self, id: ConfigConstId) -> Option<&str> {
        self.config_consts.get(&id.0).map(String::as_str)
    }
    pub fn get_event_ring_name(&self, id: EventRingId) -> Option<&str> {
        self.event_rings.get(&id.0).map(String::as_str)
    }
}

/// Helper — render `id` as `name` if interned, else `#N`. Used by both
/// the program pretty-printer and `display_with_names`.
fn render_id<'a>(name: Option<&'a str>, id: u32) -> NamedId<'a> {
    NamedId { name, id }
}

struct NamedId<'a> {
    name: Option<&'a str>,
    id: u32,
}

impl fmt::Display for NamedId<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name {
            Some(n) => write!(f, "{}", n),
            None => write!(f, "#{}", self.id),
        }
    }
}

impl fmt::Display for Interner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("interner {\n")?;
        write_table(f, "views", &self.views)?;
        write_table(f, "masks", &self.masks)?;
        write_table(f, "scorings", &self.scorings)?;
        write_table(f, "physics_rules", &self.physics_rules)?;
        write_table(f, "event_kinds", &self.event_kinds)?;
        write_table(f, "actions", &self.actions)?;
        write_table(f, "config_consts", &self.config_consts)?;
        write_table(f, "event_rings", &self.event_rings)?;
        f.write_str("}")
    }
}

fn write_table(
    f: &mut fmt::Formatter<'_>,
    label: &str,
    table: &BTreeMap<u32, String>,
) -> fmt::Result {
    write!(f, "    {}: {{", label)?;
    if table.is_empty() {
        f.write_str("}\n")?;
        return Ok(());
    }
    let mut first = true;
    for (id, name) in table {
        if !first {
            f.write_str(",")?;
        }
        first = false;
        write!(f, " #{} -> \"{}\"", id, name)?;
    }
    f.write_str(" }\n")
}

// ---------------------------------------------------------------------------
// BuilderError
// ---------------------------------------------------------------------------

/// Typed errors returned by [`CgProgramBuilder`]'s `add_*` methods when
/// a node references an id not yet allocated in the relevant arena, or
/// when an argument list itself contains a dangling id.
///
/// Every variant names the offending id; no `String` reasons. The
/// builder's fail-at-insertion-time design means consumers can rely on
/// `program.exprs[id.0 as usize]` succeeding for every id referenced
/// inside `program`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuilderError {
    /// A `CgExprId` referenced in the new node is out of range.
    DanglingExprId { referenced: CgExprId, arena_len: u32 },
    /// A `CgStmtId` referenced in the new node is out of range.
    DanglingStmtId { referenced: CgStmtId, arena_len: u32 },
    /// A `CgStmtListId` referenced in the new node is out of range.
    DanglingStmtListId {
        referenced: CgStmtListId,
        arena_len: u32,
    },
}

impl fmt::Display for BuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuilderError::DanglingExprId {
                referenced,
                arena_len,
            } => write!(
                f,
                "dangling CgExprId(#{}) (expr arena holds {} entries)",
                referenced.0, arena_len
            ),
            BuilderError::DanglingStmtId {
                referenced,
                arena_len,
            } => write!(
                f,
                "dangling CgStmtId(#{}) (stmt arena holds {} entries)",
                referenced.0, arena_len
            ),
            BuilderError::DanglingStmtListId {
                referenced,
                arena_len,
            } => write!(
                f,
                "dangling CgStmtListId(#{}) (stmt-list arena holds {} entries)",
                referenced.0, arena_len
            ),
        }
    }
}

impl std::error::Error for BuilderError {}

// ---------------------------------------------------------------------------
// CgProgram
// ---------------------------------------------------------------------------

/// Top-level container for the Compute-Graph IR.
///
/// Owns every arena IR-layer passes index into. Construct with
/// [`CgProgramBuilder`]; the builder enforces the indexing invariants
/// every later pass relies on.
///
/// `serde::Deserialize` round-trips structural fields; spans on
/// embedded [`ComputeOp`]s are reset to [`Span::dummy`] (`#[serde(skip,
/// default = "Span::dummy")]`), and spans on [`CgDiagnostic`]s are
/// reset to `None`. Compare on the structural fields, not the spans.
///
/// # Mutation rules
///
/// - During construction, mutate only via [`CgProgramBuilder`].
/// - After construction, the only sanctioned mutations are
///   [`ComputeOp::record_read`] / [`ComputeOp::record_write`] on
///   `program.ops[op_id.0 as usize]`, used by lowering passes to
///   inject registry-resolved bindings the auto-walker can't
///   synthesize.
/// - Direct `program.exprs.push(...)` etc. is allowed only when the
///   caller has manually validated that no existing entry references
///   an id beyond the new arena length — no consumer of the IR
///   defends against arena truncation, so don't shrink the arenas.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CgProgram {
    /// Every compute op the DSL produced. Indexed by [`OpId`].
    pub ops: Vec<ComputeOp>,
    /// Every [`CgExpr`] — flattened arena indexed by [`CgExprId`].
    pub exprs: Vec<CgExpr>,
    /// Every [`CgStmt`] — flattened arena indexed by [`CgStmtId`].
    /// Not listed in the plan body but required for `CgStmtListId →
    /// CgStmtList → CgStmtId → CgStmt` resolution; see the module-
    /// level note on the editorial gap.
    pub stmts: Vec<CgStmt>,
    /// Every [`CgStmtList`] — flattened arena indexed by
    /// [`CgStmtListId`].
    pub stmt_lists: Vec<CgStmtList>,
    /// Stable interned ids → human-readable names (for diagnostics
    /// and pretty-printing).
    pub interner: Interner,
    /// IR-level diagnostics (warnings, lints, fusion suggestions).
    pub diagnostics: Vec<CgDiagnostic>,
}

impl CgProgram {
    /// Construct an empty program. Equivalent to `CgProgram::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Render `handle` consulting `self.interner` — `view[#3]` becomes
    /// `view[standing]` if the interner has a name for `ViewId(3)`,
    /// else stays `view[#3]`. Mirrors the [`fmt::Display`] impl on
    /// `DataHandle` shape-for-shape; the only difference is the named
    /// substitution.
    pub fn display_with_names(&self, handle: &DataHandle) -> String {
        match handle {
            DataHandle::AgentField { field, target } => {
                format!("agent.{}.{}", target, field)
            }
            DataHandle::ViewStorage { view, slot } => {
                let name = render_id(self.interner.get_view_name(*view), view.0);
                format!("view[{}].{}", name, slot)
            }
            DataHandle::EventRing { ring, kind } => {
                let name = render_id(self.interner.get_event_ring_name(*ring), ring.0);
                format!("event_ring[{}].{}", name, kind)
            }
            DataHandle::ConfigConst { id } => {
                let name = render_id(self.interner.get_config_const_name(*id), id.0);
                format!("config[{}]", name)
            }
            DataHandle::MaskBitmap { mask } => {
                let name = render_id(self.interner.get_mask_name(*mask), mask.0);
                format!("mask[{}].bitmap", name)
            }
            DataHandle::ScoringOutput => "scoring.output".to_string(),
            DataHandle::SpatialStorage { kind } => format!("spatial.{}", kind),
            DataHandle::Rng { purpose } => format!("rng({})", purpose),
        }
    }
}

// --- Arena trait impls --------------------------------------------------

impl ExprArena for CgProgram {
    fn get(&self, id: CgExprId) -> Option<&CgExpr> {
        ExprArena::get(&self.exprs, id)
    }
}

impl StmtArena for CgProgram {
    fn get(&self, id: CgStmtId) -> Option<&CgStmt> {
        StmtArena::get(&self.stmts, id)
    }
}

impl StmtListArena for CgProgram {
    fn get(&self, id: CgStmtListId) -> Option<&CgStmtList> {
        StmtListArena::get(&self.stmt_lists, id)
    }
}

// --- Display ------------------------------------------------------------

impl fmt::Display for CgProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("program {\n")?;

        // Diagnostics block
        f.write_str("    diagnostics: [")?;
        if self.diagnostics.is_empty() {
            f.write_str("],\n")?;
        } else {
            f.write_str("\n")?;
            for d in &self.diagnostics {
                writeln!(f, "        {},", d)?;
            }
            f.write_str("    ],\n")?;
        }

        // Interner block — already prints multi-line.
        // Indent each line by 4 spaces.
        let interner_text = format!("{}", self.interner);
        for line in interner_text.lines() {
            writeln!(f, "    {}", line)?;
        }

        // Ops block
        f.write_str("    ops: [")?;
        if self.ops.is_empty() {
            f.write_str("],\n")?;
        } else {
            f.write_str("\n")?;
            for op in &self.ops {
                writeln!(f, "        {},", op)?;
            }
            f.write_str("    ],\n")?;
        }

        f.write_str("}")
    }
}

// ---------------------------------------------------------------------------
// CgProgramBuilder
// ---------------------------------------------------------------------------

/// Constructs a [`CgProgram`] with reference invariants enforced at
/// insertion time.
///
/// Each `add_*` method returns the freshly-allocated id; ids are
/// contiguous starting from zero, in insertion order. Methods that take
/// a node containing references (`add_expr` for a `Binary`, `add_op`
/// for a `MaskPredicate`, …) validate every referenced id against the
/// current arena length before pushing — a dangling id surfaces as a
/// typed [`BuilderError`].
///
/// Lowering passes are the canonical caller. Tests can also build
/// programs directly when exercising consumer code.
pub struct CgProgramBuilder {
    inner: CgProgram,
}

impl Default for CgProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CgProgramBuilder {
    /// Start a new builder with empty arenas, an empty interner, and
    /// no diagnostics.
    pub fn new() -> Self {
        Self {
            inner: CgProgram::new(),
        }
    }

    // --- Interner intern_* methods --------------------------------------

    /// Record a human-readable name for `id` — used by
    /// [`CgProgram::display_with_names`] and by diagnostics.
    pub fn intern_view_name(&mut self, id: ViewId, name: impl Into<String>) {
        self.inner.interner.views.insert(id.0, name.into());
    }
    pub fn intern_mask_name(&mut self, id: MaskId, name: impl Into<String>) {
        self.inner.interner.masks.insert(id.0, name.into());
    }
    pub fn intern_scoring_name(&mut self, id: ScoringId, name: impl Into<String>) {
        self.inner.interner.scorings.insert(id.0, name.into());
    }
    pub fn intern_physics_rule_name(&mut self, id: PhysicsRuleId, name: impl Into<String>) {
        self.inner.interner.physics_rules.insert(id.0, name.into());
    }
    pub fn intern_event_kind_name(&mut self, id: EventKindId, name: impl Into<String>) {
        self.inner.interner.event_kinds.insert(id.0, name.into());
    }
    pub fn intern_action_name(&mut self, id: ActionId, name: impl Into<String>) {
        self.inner.interner.actions.insert(id.0, name.into());
    }
    pub fn intern_config_const_name(&mut self, id: ConfigConstId, name: impl Into<String>) {
        self.inner.interner.config_consts.insert(id.0, name.into());
    }
    pub fn intern_event_ring_name(&mut self, id: EventRingId, name: impl Into<String>) {
        self.inner.interner.event_rings.insert(id.0, name.into());
    }

    // --- Diagnostic push ------------------------------------------------

    /// Append a diagnostic. Accepted unconditionally — the IR layer
    /// does not validate diagnostic structure (severity + kind are
    /// already typed at the call site).
    pub fn add_diagnostic(&mut self, diag: CgDiagnostic) {
        self.inner.diagnostics.push(diag);
    }

    // --- Arena length helpers (used by validators) ---------------------

    fn expr_len(&self) -> u32 {
        self.inner.exprs.len() as u32
    }
    fn stmt_len(&self) -> u32 {
        self.inner.stmts.len() as u32
    }
    fn list_len(&self) -> u32 {
        self.inner.stmt_lists.len() as u32
    }

    fn check_expr_id(&self, id: CgExprId) -> Result<(), BuilderError> {
        if id.0 < self.expr_len() {
            Ok(())
        } else {
            Err(BuilderError::DanglingExprId {
                referenced: id,
                arena_len: self.expr_len(),
            })
        }
    }

    fn check_stmt_id(&self, id: CgStmtId) -> Result<(), BuilderError> {
        if id.0 < self.stmt_len() {
            Ok(())
        } else {
            Err(BuilderError::DanglingStmtId {
                referenced: id,
                arena_len: self.stmt_len(),
            })
        }
    }

    fn check_list_id(&self, id: CgStmtListId) -> Result<(), BuilderError> {
        if id.0 < self.list_len() {
            Ok(())
        } else {
            Err(BuilderError::DanglingStmtListId {
                referenced: id,
                arena_len: self.list_len(),
            })
        }
    }

    // --- Expr validation -----------------------------------------------

    /// Validate every `CgExprId` reference inside `expr`. Returns the
    /// first dangling id encountered, in left-to-right operand order.
    fn validate_expr_refs(&self, expr: &CgExpr) -> Result<(), BuilderError> {
        match expr {
            CgExpr::Read(_) | CgExpr::Lit(_) | CgExpr::Rng { .. } => Ok(()),
            CgExpr::Binary { lhs, rhs, .. } => {
                self.check_expr_id(*lhs)?;
                self.check_expr_id(*rhs)?;
                Ok(())
            }
            CgExpr::Unary { arg, .. } => self.check_expr_id(*arg),
            CgExpr::Builtin { args, .. } => {
                for a in args {
                    self.check_expr_id(*a)?;
                }
                Ok(())
            }
            CgExpr::Select {
                cond, then, else_, ..
            } => {
                self.check_expr_id(*cond)?;
                self.check_expr_id(*then)?;
                self.check_expr_id(*else_)?;
                Ok(())
            }
        }
    }

    // --- Stmt validation -----------------------------------------------

    /// Validate every id reference inside `stmt`. `Assign` walks its
    /// value expression; `Emit` walks each field expression; `If`
    /// walks the condition expression and the then/else list ids.
    fn validate_stmt_refs(&self, stmt: &CgStmt) -> Result<(), BuilderError> {
        match stmt {
            CgStmt::Assign { value, .. } => self.check_expr_id(*value),
            CgStmt::Emit { fields, .. } => {
                for (_, expr_id) in fields {
                    self.check_expr_id(*expr_id)?;
                }
                Ok(())
            }
            CgStmt::If { cond, then, else_ } => {
                self.check_expr_id(*cond)?;
                self.check_list_id(*then)?;
                if let Some(else_id) = else_ {
                    self.check_list_id(*else_id)?;
                }
                Ok(())
            }
        }
    }

    // --- Add_* entry points --------------------------------------------

    /// Push a `CgExpr` and return its freshly-allocated [`CgExprId`].
    /// Validates every child id reference against the current expr
    /// arena length before pushing.
    pub fn add_expr(&mut self, expr: CgExpr) -> Result<CgExprId, BuilderError> {
        self.validate_expr_refs(&expr)?;
        let id = CgExprId(self.expr_len());
        self.inner.exprs.push(expr);
        Ok(id)
    }

    /// Push a `CgStmt` and return its freshly-allocated [`CgStmtId`].
    /// Validates every referenced expr / list id.
    pub fn add_stmt(&mut self, stmt: CgStmt) -> Result<CgStmtId, BuilderError> {
        self.validate_stmt_refs(&stmt)?;
        let id = CgStmtId(self.stmt_len());
        self.inner.stmts.push(stmt);
        Ok(id)
    }

    /// Push a `CgStmtList` and return its freshly-allocated
    /// [`CgStmtListId`]. Validates that every contained `CgStmtId`
    /// resolves.
    pub fn add_stmt_list(&mut self, list: CgStmtList) -> Result<CgStmtListId, BuilderError> {
        for stmt_id in &list.stmts {
            self.check_stmt_id(*stmt_id)?;
        }
        let id = CgStmtListId(self.list_len());
        self.inner.stmt_lists.push(list);
        Ok(id)
    }

    /// Add an op. The builder assigns the [`OpId`], validates every
    /// expr / list id referenced by `kind`, then constructs the
    /// [`ComputeOp`] via [`ComputeOp::new`] (which auto-derives reads
    /// + writes from the current arenas).
    pub fn add_op(
        &mut self,
        kind: ComputeOpKind,
        shape: DispatchShape,
        span: Span,
    ) -> Result<OpId, BuilderError> {
        self.validate_op_kind_refs(&kind)?;
        let id = OpId(self.inner.ops.len() as u32);
        let op = ComputeOp::new(
            id,
            kind,
            shape,
            span,
            &self.inner.exprs,
            &self.inner.stmts,
            &self.inner.stmt_lists,
        );
        self.inner.ops.push(op);
        Ok(id)
    }

    /// Validate every id reference inside `kind`. Each variant has
    /// distinct reference shapes — mask predicates carry one expr id,
    /// scoring rows carry two each, physics/view-fold carry one stmt
    /// list id, spatial query / plumbing carry none.
    fn validate_op_kind_refs(&self, kind: &ComputeOpKind) -> Result<(), BuilderError> {
        match kind {
            ComputeOpKind::MaskPredicate { predicate, .. } => self.check_expr_id(*predicate),
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                for row in rows {
                    self.check_expr_id(row.utility)?;
                    self.check_expr_id(row.target)?;
                }
                Ok(())
            }
            ComputeOpKind::PhysicsRule { body, .. } => self.check_list_id(*body),
            ComputeOpKind::ViewFold { body, .. } => self.check_list_id(*body),
            ComputeOpKind::SpatialQuery { .. } => Ok(()),
            ComputeOpKind::Plumbing { kind } => {
                // Uninhabited until Task 2.7 — the empty match
                // exhausts every reachable variant.
                match *kind {}
            }
        }
    }

    /// Finalize the builder and return the constructed program. After
    /// `finish`, post-construction mutations (record_read /
    /// record_write on individual ops) are still permitted; arena
    /// growth via the builder is not.
    pub fn finish(self) -> CgProgram {
        self.inner
    }

    // --- Read-only accessors (used by lowering callers) ----------------

    /// Borrow the program-in-progress for read-only inspection — used
    /// by lowering passes that want to consult the current expr arena
    /// while building (e.g. to share a sub-expression they already
    /// added).
    pub fn program(&self) -> &CgProgram {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{
        AgentFieldId, AgentRef, EventRingAccess, RngPurpose, SpatialStorageKind, ViewStorageSlot,
    };
    use crate::cg::dispatch::{DispatchShape, PerPairSource};
    use crate::cg::expr::{BinaryOp, CgTy, LitValue};
    use crate::cg::op::{ActionId, EventKindId, PhysicsRuleId, ScoringRowOp, SpatialQueryKind};
    use crate::cg::stmt::EventField;

    // --- helpers -------------------------------------------------------

    fn read_self_hp() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        })
    }

    fn lit_f32(v: f32) -> CgExpr {
        CgExpr::Lit(LitValue::F32(v))
    }

    // --- Severity / CgDiagnostic --------------------------------------

    #[test]
    fn severity_display_distinct_per_variant() {
        assert_eq!(format!("{}", Severity::Note), "note");
        assert_eq!(format!("{}", Severity::Warning), "warning");
        assert_eq!(format!("{}", Severity::Lint), "lint");
    }

    #[test]
    fn cg_diagnostic_kind_display_orphaned_emit() {
        let d = CgDiagnosticKind::OrphanedEmit {
            event: EventKindId(7),
        };
        assert_eq!(format!("{}", d), "orphaned_emit(event=#7)");
    }

    #[test]
    fn cg_diagnostic_kind_display_unread_view() {
        let d = CgDiagnosticKind::UnreadView { view: ViewId(3) };
        assert_eq!(format!("{}", d), "unread_view(view=#3)");
    }

    #[test]
    fn cg_diagnostic_kind_display_unused_mask() {
        let d = CgDiagnosticKind::UnusedMask { mask: MaskId(5) };
        assert_eq!(format!("{}", d), "unused_mask(mask=#5)");
    }

    #[test]
    fn cg_diagnostic_kind_display_write_conflict() {
        let d = CgDiagnosticKind::WriteConflict {
            handle: DataHandle::MaskBitmap { mask: MaskId(2) },
            ops: vec![OpId(0), OpId(3)],
        };
        assert_eq!(
            format!("{}", d),
            "write_conflict(handle=mask[#2].bitmap, ops=[op#0, op#3])"
        );
    }

    #[test]
    fn cg_diagnostic_constructors_set_severity_and_span() {
        let d1 = CgDiagnostic::new(
            Severity::Warning,
            CgDiagnosticKind::UnusedMask { mask: MaskId(1) },
        );
        assert_eq!(d1.severity, Severity::Warning);
        assert_eq!(d1.span, None);

        let d2 = CgDiagnostic::with_span(
            Severity::Note,
            Span::new(10, 20),
            CgDiagnosticKind::UnreadView { view: ViewId(2) },
        );
        assert_eq!(d2.severity, Severity::Note);
        assert_eq!(d2.span, Some(Span::new(10, 20)));
    }

    #[test]
    fn cg_diagnostic_display_combines_severity_and_kind() {
        let d = CgDiagnostic::new(
            Severity::Warning,
            CgDiagnosticKind::OrphanedEmit {
                event: EventKindId(4),
            },
        );
        assert_eq!(format!("{}", d), "warning: orphaned_emit(event=#4)");
    }

    #[test]
    fn cg_diagnostic_serde_round_trip_modulo_span() {
        let d = CgDiagnostic::with_span(
            Severity::Warning,
            Span::new(10, 20),
            CgDiagnosticKind::UnusedMask { mask: MaskId(7) },
        );
        let json = serde_json::to_string(&d).expect("serialize");
        let back: CgDiagnostic = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.severity, d.severity);
        assert_eq!(back.kind, d.kind);
        // Span is not serialized — round-trip resets to None.
        assert_eq!(back.span, None);
    }

    // --- Interner ------------------------------------------------------

    #[test]
    fn interner_lookup_returns_inserted_name() {
        let mut interner = Interner::new();
        interner.views.insert(0, "standing".to_string());
        interner.masks.insert(3, "low_hp".to_string());
        assert_eq!(interner.get_view_name(ViewId(0)), Some("standing"));
        assert_eq!(interner.get_mask_name(MaskId(3)), Some("low_hp"));
        assert_eq!(interner.get_view_name(ViewId(99)), None);
    }

    #[test]
    fn interner_display_sorted_by_id() {
        let mut interner = Interner::new();
        interner.views.insert(2, "engaged".to_string());
        interner.views.insert(0, "standing".to_string());
        interner.views.insert(1, "fleeing".to_string());
        let s = format!("{}", interner);
        // Each table prints sorted (BTreeMap iteration order).
        assert!(s.contains("#0 -> \"standing\""));
        assert!(s.contains("#1 -> \"fleeing\""));
        assert!(s.contains("#2 -> \"engaged\""));
        // Ordering: standing before fleeing before engaged.
        let i_standing = s.find("standing").unwrap();
        let i_fleeing = s.find("fleeing").unwrap();
        let i_engaged = s.find("engaged").unwrap();
        assert!(i_standing < i_fleeing && i_fleeing < i_engaged);
    }

    #[test]
    fn interner_default_is_empty() {
        let interner = Interner::default();
        assert!(interner.views.is_empty());
        assert!(interner.masks.is_empty());
        assert!(interner.scorings.is_empty());
        assert!(interner.physics_rules.is_empty());
        assert!(interner.event_kinds.is_empty());
        assert!(interner.actions.is_empty());
        assert!(interner.config_consts.is_empty());
        assert!(interner.event_rings.is_empty());
    }

    // --- Builder add_expr ---------------------------------------------

    #[test]
    fn builder_add_expr_assigns_contiguous_ids() {
        let mut b = CgProgramBuilder::new();
        let a = b.add_expr(read_self_hp()).unwrap();
        let b1 = b.add_expr(lit_f32(1.0)).unwrap();
        assert_eq!(a, CgExprId(0));
        assert_eq!(b1, CgExprId(1));
    }

    #[test]
    fn builder_add_expr_validates_binary_ids() {
        let mut b = CgProgramBuilder::new();
        let lhs = b.add_expr(read_self_hp()).unwrap();
        // rhs id is dangling — only one expr in the arena so far.
        let err = b
            .add_expr(CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs,
                rhs: CgExprId(99),
                ty: CgTy::F32,
            })
            .expect_err("dangling rhs");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(99),
                arena_len: 1,
            }
        );
    }

    #[test]
    fn builder_add_expr_validates_select_ids() {
        let mut b = CgProgramBuilder::new();
        let cond = b.add_expr(CgExpr::Lit(LitValue::Bool(true))).unwrap();
        let err = b
            .add_expr(CgExpr::Select {
                cond,
                then: CgExprId(7),
                else_: CgExprId(0),
                ty: CgTy::F32,
            })
            .expect_err("dangling then");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(7),
                arena_len: 1,
            }
        );
    }

    #[test]
    fn builder_add_expr_accepts_lit_and_rng_with_no_refs() {
        let mut b = CgProgramBuilder::new();
        b.add_expr(lit_f32(2.5)).unwrap();
        b.add_expr(CgExpr::Rng {
            purpose: RngPurpose::Action,
            ty: CgTy::U32,
        })
        .unwrap();
        assert_eq!(b.program().exprs.len(), 2);
    }

    // --- Builder add_stmt + add_stmt_list -----------------------------

    #[test]
    fn builder_add_stmt_validates_assign_value() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(0),
            })
            .expect_err("dangling value");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_stmt_emit_walks_field_exprs() {
        let mut b = CgProgramBuilder::new();
        let _ = b.add_expr(read_self_hp()).unwrap();
        // Field expr id #99 dangles.
        let err = b
            .add_stmt(CgStmt::Emit {
                event: EventKindId(0),
                fields: vec![(
                    EventField {
                        event: EventKindId(0),
                        index: 0,
                    },
                    CgExprId(99),
                )],
            })
            .expect_err("dangling field expr");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(99),
                arena_len: 1,
            }
        );
    }

    #[test]
    fn builder_add_stmt_if_validates_then_else_lists() {
        let mut b = CgProgramBuilder::new();
        let cond = b.add_expr(CgExpr::Lit(LitValue::Bool(true))).unwrap();
        // No lists yet — `then` dangles.
        let err = b
            .add_stmt(CgStmt::If {
                cond,
                then: CgStmtListId(0),
                else_: None,
            })
            .expect_err("dangling then list");
        assert_eq!(
            err,
            BuilderError::DanglingStmtListId {
                referenced: CgStmtListId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_stmt_list_validates_contained_stmt_ids() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_stmt_list(CgStmtList::new(vec![CgStmtId(7)]))
            .expect_err("dangling stmt id");
        assert_eq!(
            err,
            BuilderError::DanglingStmtId {
                referenced: CgStmtId(7),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_stmt_list_accepts_empty() {
        let mut b = CgProgramBuilder::new();
        let id = b.add_stmt_list(CgStmtList::new(vec![])).unwrap();
        assert_eq!(id, CgStmtListId(0));
    }

    // --- Builder add_op ------------------------------------------------

    #[test]
    fn builder_add_op_mask_predicate_validates_predicate_id() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_op(
                ComputeOpKind::MaskPredicate {
                    mask: MaskId(0),
                    predicate: CgExprId(0),
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .expect_err("dangling predicate");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_op_mask_predicate_succeeds_when_expr_exists() {
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
        let op_id = b
            .add_op(
                ComputeOpKind::MaskPredicate {
                    mask: MaskId(0),
                    predicate: pred,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap();
        assert_eq!(op_id, OpId(0));

        let prog = b.finish();
        // reads/writes auto-populated by ComputeOp::new walking the arenas.
        assert_eq!(
            prog.ops[0].reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(
            prog.ops[0].writes,
            vec![DataHandle::MaskBitmap { mask: MaskId(0) }]
        );
    }

    #[test]
    fn builder_add_op_scoring_validates_each_row() {
        let mut b = CgProgramBuilder::new();
        let utility = b.add_expr(lit_f32(1.0)).unwrap();
        // Target id is dangling — only one expr in arena.
        let err = b
            .add_op(
                ComputeOpKind::ScoringArgmax {
                    scoring: ScoringId(0),
                    rows: vec![ScoringRowOp {
                        action: ActionId(0),
                        utility,
                        target: CgExprId(99),
                    }],
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .expect_err("dangling row target");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(99),
                arena_len: 1,
            }
        );
    }

    #[test]
    fn builder_add_op_physics_rule_validates_body_id() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule: PhysicsRuleId(0),
                    on_event: EventKindId(0),
                    body: CgStmtListId(0),
                },
                DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
                Span::dummy(),
            )
            .expect_err("dangling body list");
        assert_eq!(
            err,
            BuilderError::DanglingStmtListId {
                referenced: CgStmtListId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_op_view_fold_validates_body_id() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_op(
                ComputeOpKind::ViewFold {
                    view: ViewId(0),
                    on_event: EventKindId(0),
                    body: CgStmtListId(2),
                },
                DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
                Span::dummy(),
            )
            .expect_err("dangling body list");
        assert_eq!(
            err,
            BuilderError::DanglingStmtListId {
                referenced: CgStmtListId(2),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_op_spatial_query_no_id_refs() {
        let mut b = CgProgramBuilder::new();
        let id = b
            .add_op(
                ComputeOpKind::SpatialQuery {
                    kind: SpatialQueryKind::BuildHash,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap();
        assert_eq!(id, OpId(0));
    }

    #[test]
    fn builder_add_op_assigns_contiguous_op_ids() {
        let mut b = CgProgramBuilder::new();
        let a = b
            .add_op(
                ComputeOpKind::SpatialQuery {
                    kind: SpatialQueryKind::BuildHash,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap();
        let c = b
            .add_op(
                ComputeOpKind::SpatialQuery {
                    kind: SpatialQueryKind::KinQuery,
                },
                DispatchShape::PerPair {
                    source: PerPairSource::SpatialQuery(SpatialQueryKind::KinQuery),
                },
                Span::dummy(),
            )
            .unwrap();
        assert_eq!(a, OpId(0));
        assert_eq!(c, OpId(1));
    }

    // --- Diagnostics + interner intern_* ------------------------------

    #[test]
    fn builder_add_diagnostic_pushes_into_arena() {
        let mut b = CgProgramBuilder::new();
        b.add_diagnostic(CgDiagnostic::new(
            Severity::Warning,
            CgDiagnosticKind::UnusedMask { mask: MaskId(0) },
        ));
        let prog = b.finish();
        assert_eq!(prog.diagnostics.len(), 1);
        assert_eq!(prog.diagnostics[0].severity, Severity::Warning);
    }

    #[test]
    fn builder_intern_methods_populate_interner_tables() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing");
        b.intern_mask_name(MaskId(1), "low_hp");
        b.intern_scoring_name(ScoringId(2), "combat");
        b.intern_physics_rule_name(PhysicsRuleId(3), "on_attack");
        b.intern_event_kind_name(EventKindId(4), "AttackHit");
        b.intern_action_name(ActionId(5), "MoveToward");
        b.intern_config_const_name(ConfigConstId(6), "attack_range");
        b.intern_event_ring_name(EventRingId(7), "apply_ring");
        let prog = b.finish();
        assert_eq!(
            prog.interner.get_view_name(ViewId(0)),
            Some("standing")
        );
        assert_eq!(
            prog.interner.get_mask_name(MaskId(1)),
            Some("low_hp")
        );
        assert_eq!(
            prog.interner.get_scoring_name(ScoringId(2)),
            Some("combat")
        );
        assert_eq!(
            prog.interner.get_physics_rule_name(PhysicsRuleId(3)),
            Some("on_attack")
        );
        assert_eq!(
            prog.interner.get_event_kind_name(EventKindId(4)),
            Some("AttackHit")
        );
        assert_eq!(
            prog.interner.get_action_name(ActionId(5)),
            Some("MoveToward")
        );
        assert_eq!(
            prog.interner.get_config_const_name(ConfigConstId(6)),
            Some("attack_range")
        );
        assert_eq!(
            prog.interner.get_event_ring_name(EventRingId(7)),
            Some("apply_ring")
        );
    }

    // --- CgProgram arena traits ---------------------------------------

    #[test]
    fn program_implements_expr_arena() {
        let mut b = CgProgramBuilder::new();
        let id = b.add_expr(read_self_hp()).unwrap();
        let prog = b.finish();
        let node: &CgExpr = ExprArena::get(&prog, id).expect("expr id resolves");
        match node {
            CgExpr::Read(DataHandle::AgentField { field, .. }) => {
                assert_eq!(*field, AgentFieldId::Hp);
            }
            other => panic!("expected hp read, got {other:?}"),
        }
    }

    #[test]
    fn program_implements_stmt_arena() {
        let mut b = CgProgramBuilder::new();
        let val = b.add_expr(lit_f32(0.0)).unwrap();
        let stmt_id = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: val,
            })
            .unwrap();
        let prog = b.finish();
        let node: &CgStmt = StmtArena::get(&prog, stmt_id).expect("stmt id resolves");
        match node {
            CgStmt::Assign { value, .. } => assert_eq!(*value, val),
            other => panic!("expected Assign, got {other:?}"),
        }
    }

    #[test]
    fn program_implements_stmt_list_arena() {
        let mut b = CgProgramBuilder::new();
        let val = b.add_expr(lit_f32(0.0)).unwrap();
        let stmt_id = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: val,
            })
            .unwrap();
        let list_id = b
            .add_stmt_list(CgStmtList::new(vec![stmt_id]))
            .unwrap();
        let prog = b.finish();
        let node: &CgStmtList = StmtListArena::get(&prog, list_id).expect("list id resolves");
        assert_eq!(node.stmts, vec![stmt_id]);
    }

    // --- display_with_names -------------------------------------------

    #[test]
    fn display_with_names_substitutes_view_name() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(3), "engaged");
        let prog = b.finish();
        let h = DataHandle::ViewStorage {
            view: ViewId(3),
            slot: ViewStorageSlot::Primary,
        };
        assert_eq!(prog.display_with_names(&h), "view[engaged].primary");
    }

    #[test]
    fn display_with_names_falls_back_to_id_form_without_interner_entry() {
        let prog = CgProgram::new();
        let h = DataHandle::ViewStorage {
            view: ViewId(3),
            slot: ViewStorageSlot::Primary,
        };
        assert_eq!(prog.display_with_names(&h), "view[#3].primary");
    }

    #[test]
    fn display_with_names_does_not_modify_data_handle_display() {
        // The opaque-id Display on DataHandle is preserved (Task 1.1
        // contract). display_with_names is purely additive.
        let h = DataHandle::ViewStorage {
            view: ViewId(3),
            slot: ViewStorageSlot::Primary,
        };
        assert_eq!(format!("{}", h), "view[#3].primary");
    }

    #[test]
    fn display_with_names_for_each_handle_variant() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing");
        b.intern_mask_name(MaskId(1), "low_hp");
        b.intern_event_ring_name(EventRingId(2), "apply");
        b.intern_config_const_name(ConfigConstId(3), "attack_range");
        let prog = b.finish();

        // AgentField — no interner involvement (field name is fixed).
        assert_eq!(
            prog.display_with_names(&DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
            "agent.self.hp"
        );
        // ViewStorage — named.
        assert_eq!(
            prog.display_with_names(&DataHandle::ViewStorage {
                view: ViewId(0),
                slot: ViewStorageSlot::Primary,
            }),
            "view[standing].primary"
        );
        // EventRing — named.
        assert_eq!(
            prog.display_with_names(&DataHandle::EventRing {
                ring: EventRingId(2),
                kind: EventRingAccess::Append,
            }),
            "event_ring[apply].append"
        );
        // ConfigConst — named.
        assert_eq!(
            prog.display_with_names(&DataHandle::ConfigConst {
                id: ConfigConstId(3),
            }),
            "config[attack_range]"
        );
        // MaskBitmap — named.
        assert_eq!(
            prog.display_with_names(&DataHandle::MaskBitmap { mask: MaskId(1) }),
            "mask[low_hp].bitmap"
        );
        // ScoringOutput — atomic.
        assert_eq!(
            prog.display_with_names(&DataHandle::ScoringOutput),
            "scoring.output"
        );
        // SpatialStorage — atomic.
        assert_eq!(
            prog.display_with_names(&DataHandle::SpatialStorage {
                kind: SpatialStorageKind::GridCells,
            }),
            "spatial.grid_cells"
        );
        // Rng — atomic.
        assert_eq!(
            prog.display_with_names(&DataHandle::Rng {
                purpose: RngPurpose::Action,
            }),
            "rng(action)"
        );
    }

    // --- CgProgram serde round-trip -----------------------------------

    #[test]
    fn program_serde_round_trip_preserves_structure_modulo_spans() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing");
        b.intern_mask_name(MaskId(1), "low_hp");
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
                mask: MaskId(1),
                predicate: pred,
            },
            DispatchShape::PerAgent,
            Span::new(10, 20),
        )
        .unwrap();
        b.add_diagnostic(CgDiagnostic::with_span(
            Severity::Warning,
            Span::new(30, 40),
            CgDiagnosticKind::UnusedMask { mask: MaskId(7) },
        ));
        let prog = b.finish();

        let json = serde_json::to_string(&prog).expect("serialize");
        let back: CgProgram = serde_json::from_str(&json).expect("deserialize");

        // Structural fields round-trip.
        assert_eq!(back.exprs, prog.exprs);
        assert_eq!(back.stmts, prog.stmts);
        assert_eq!(back.stmt_lists, prog.stmt_lists);
        assert_eq!(back.interner, prog.interner);
        // Op compares modulo span (Span has no Deserialize, so it
        // resets to dummy on round-trip). The other op fields match.
        assert_eq!(back.ops.len(), 1);
        assert_eq!(back.ops[0].id, prog.ops[0].id);
        assert_eq!(back.ops[0].kind, prog.ops[0].kind);
        assert_eq!(back.ops[0].reads, prog.ops[0].reads);
        assert_eq!(back.ops[0].writes, prog.ops[0].writes);
        assert_eq!(back.ops[0].shape, prog.ops[0].shape);
        assert_eq!(back.ops[0].span, Span::dummy());
        // Diagnostic span is None on round-trip; severity + kind preserved.
        assert_eq!(back.diagnostics.len(), 1);
        assert_eq!(back.diagnostics[0].severity, Severity::Warning);
        assert_eq!(back.diagnostics[0].kind, prog.diagnostics[0].kind);
        assert_eq!(back.diagnostics[0].span, None);
    }

    // --- Pretty-printer fixture ---------------------------------------

    /// Build a small but representative program — one mask, one
    /// scoring, one physics rule, one view fold, one spatial query —
    /// and pin the full pretty-printed output.
    fn build_fixture() -> CgProgram {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing");
        b.intern_mask_name(MaskId(0), "low_hp");

        // Expression arena layout (deterministic, in insertion order).
        let hp = b.add_expr(read_self_hp()).unwrap(); // #0
        let half = b.add_expr(lit_f32(0.5)).unwrap(); // #1
        let pred = b
            .add_expr(CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: half,
                ty: CgTy::Bool,
            })
            .unwrap(); // #2
        let utility = b.add_expr(lit_f32(1.0)).unwrap(); // #3
        let target = b.add_expr(CgExpr::Lit(LitValue::AgentId(0))).unwrap(); // #4
        let zero = b.add_expr(lit_f32(0.0)).unwrap(); // #5

        // Statement arena layout.
        let assign_hp = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: zero,
            })
            .unwrap(); // stmt#0
        let assign_view = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::ViewStorage {
                    view: ViewId(0),
                    slot: ViewStorageSlot::Primary,
                },
                value: hp,
            })
            .unwrap(); // stmt#1

        // Statement list arena layout.
        let physics_body = b
            .add_stmt_list(CgStmtList::new(vec![assign_hp]))
            .unwrap(); // list#0
        let view_body = b
            .add_stmt_list(CgStmtList::new(vec![assign_view]))
            .unwrap(); // list#1

        // Ops.
        b.add_op(
            ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: pred,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap(); // op#0
        b.add_op(
            ComputeOpKind::ScoringArgmax {
                scoring: ScoringId(0),
                rows: vec![ScoringRowOp {
                    action: ActionId(0),
                    utility,
                    target,
                }],
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap(); // op#1
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body: physics_body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap(); // op#2
        b.add_op(
            ComputeOpKind::ViewFold {
                view: ViewId(0),
                on_event: EventKindId(0),
                body: view_body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap(); // op#3
        b.add_op(
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::BuildHash,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap(); // op#4

        // Diagnostic.
        b.add_diagnostic(CgDiagnostic::new(
            Severity::Warning,
            CgDiagnosticKind::UnusedMask { mask: MaskId(0) },
        ));

        b.finish()
    }

    #[test]
    fn program_pretty_print_matches_fixture() {
        let prog = build_fixture();
        let printed = format!("{}", prog);
        let expected = "\
program {
    diagnostics: [
        warning: unused_mask(mask=#0),
    ],
    interner {
        views: { #0 -> \"standing\" }
        masks: { #0 -> \"low_hp\" }
        scorings: {}
        physics_rules: {}
        event_kinds: {}
        actions: {}
        config_consts: {}
        event_rings: {}
    }
    ops: [
        op#0 kind=mask_predicate(mask=#0) shape=per_agent reads=[agent.self.hp] writes=[mask[#0].bitmap],
        op#1 kind=scoring_argmax(scoring=#0, rows=1) shape=per_agent reads=[] writes=[scoring.output],
        op#2 kind=physics_rule(rule=#0, on_event=#0) shape=per_event(ring=#0) reads=[] writes=[agent.self.hp],
        op#3 kind=view_fold(view=#0, on_event=#0) shape=per_event(ring=#0) reads=[agent.self.hp] writes=[view[#0].primary],
        op#4 kind=spatial_query(build_hash) shape=per_agent reads=[] writes=[spatial.grid_cells, spatial.grid_offsets],
    ],
}";
        assert_eq!(printed, expected, "actual output:\n{}", printed);
    }

    #[test]
    fn program_pretty_print_empty_program() {
        let prog = CgProgram::new();
        let printed = format!("{}", prog);
        let expected = "\
program {
    diagnostics: [],
    interner {
        views: {}
        masks: {}
        scorings: {}
        physics_rules: {}
        event_kinds: {}
        actions: {}
        config_consts: {}
        event_rings: {}
    }
    ops: [],
}";
        assert_eq!(printed, expected, "actual output:\n{}", printed);
    }

    // --- record_read / record_write seam ------------------------------

    #[test]
    fn record_write_through_program_ops_appends_to_writes() {
        // Lowering pattern: build the op, then reach into
        // `prog.ops[op_id]` and call record_write to inject the
        // registry-resolved binding.
        let mut b = CgProgramBuilder::new();
        let body = b.add_stmt_list(CgStmtList::new(vec![])).unwrap();
        let op_id = b
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule: PhysicsRuleId(0),
                    on_event: EventKindId(7),
                    body,
                },
                DispatchShape::PerEvent {
                    source_ring: EventRingId(7),
                },
                Span::dummy(),
            )
            .unwrap();
        let mut prog = b.finish();

        let dest = DataHandle::EventRing {
            ring: EventRingId(9),
            kind: EventRingAccess::Append,
        };
        prog.ops[op_id.0 as usize].record_write(dest.clone());
        assert!(prog.ops[op_id.0 as usize].writes.contains(&dest));
    }

    // --- BuilderError display -----------------------------------------

    #[test]
    fn builder_error_display_dangling_expr() {
        let e = BuilderError::DanglingExprId {
            referenced: CgExprId(7),
            arena_len: 3,
        };
        assert_eq!(
            format!("{}", e),
            "dangling CgExprId(#7) (expr arena holds 3 entries)"
        );
    }

    #[test]
    fn builder_error_display_dangling_stmt() {
        let e = BuilderError::DanglingStmtId {
            referenced: CgStmtId(2),
            arena_len: 0,
        };
        assert_eq!(
            format!("{}", e),
            "dangling CgStmtId(#2) (stmt arena holds 0 entries)"
        );
    }

    #[test]
    fn builder_error_display_dangling_stmt_list() {
        let e = BuilderError::DanglingStmtListId {
            referenced: CgStmtListId(1),
            arena_len: 0,
        };
        assert_eq!(
            format!("{}", e),
            "dangling CgStmtListId(#1) (stmt-list arena holds 0 entries)"
        );
    }
}
