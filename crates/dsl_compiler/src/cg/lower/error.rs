//! Unified typed-error surface for every CG lowering pass.
//!
//! Phase 2 of the Compute-Graph IR pipeline (see
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`) splits
//! AST → CG lowering across one task per construct (expression, mask,
//! view, physics, scoring, spatial, plumbing, driver). Each task can
//! contribute new defect variants without inventing a sibling enum:
//! every pass returns [`LoweringError`].
//!
//! # Naming convention
//!
//! Variants that name a defect specific to one construct are prefixed
//! with the construct name to keep the enum readable as it grows:
//!
//! - `Mask*` for mask-lowering defects (`MaskPredicateNotBool`,
//!   `UnsupportedMaskFromClause`, `UnsupportedMaskHeadShape`, …).
//! - Future tasks follow the same shape — `View*`, `Scoring*`,
//!   `Physics*`, `Spatial*` — so a Task 2.4 author adding
//!   `ViewFoldHandlerNotBool` doesn't have to disambiguate against a
//!   bare `HandlerNotBool` shared with mask predicates.
//!
//! Variants that name a defect shared across passes (e.g.,
//! [`LoweringError::BuilderRejected`], [`LoweringError::TypeCheckFailure`])
//! stay unprefixed.
//!
//! # Why one enum
//!
//! Phase 2's tasks compound — the view pass calls into the expression
//! pass, the scoring pass into the view pass, and so on. A wrapper-per-pass
//! design (`MaskLoweringError::Predicate(LoweringError)`) builds linearly
//! deeper match nests as the call graph stacks up. A single enum keeps
//! `?`-propagation flat and `Display` chains readable.

use std::fmt;

use dsl_ast::ast::{BinOp, Span};
use dsl_ast::ir::{Builtin, NamespaceId, ViewRef as AstViewRef};

use crate::cg::data_handle::{MaskId, ViewId, ViewStorageSlot};
use crate::cg::expr::{CgTy, TypeError};
use crate::cg::op::SpatialQueryKind;
use crate::cg::program::BuilderError;

/// Typed defect surfaced by any CG lowering pass.
///
/// Every variant pins the offending location (a `Span` from the AST
/// node) plus the structural reason. No `String` reasons except for the
/// genuinely-string-keyed surfaces (free-form field / namespace / view
/// names that don't appear as a typed AST enum).
///
/// See module docs for the naming convention used by per-pass variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    // -- Expression pass (Task 2.1) --------------------------------------

    /// AST variant the expression lowering does not yet handle. Used as
    /// a typed escape hatch for the staged work in Phase 2 — quantifiers,
    /// folds, struct literals, etc. land in their own tasks; until then
    /// they surface here.
    UnsupportedAstNode {
        /// Stable label for the AST variant — closed set of `&'static
        /// str` tags pulled from `IrExpr` discriminants.
        ast_label: &'static str,
        span: Span,
    },

    /// `agent.alive < 5` and friends: an operator was applied to operand
    /// types that don't share a usable structural form. The closest
    /// CG-IR analogue would be an ill-typed `CgExpr::Binary`; we
    /// refuse to construct it.
    IllTypedExpression {
        expected: CgTy,
        got: CgTy,
        span: Span,
    },

    /// Two-operand ops (binary, equality, comparison) whose operand
    /// types disagree with each other. Distinct from
    /// [`Self::IllTypedExpression`] (which carries an *expected* shape):
    /// here we report both sides as a typed mismatch and let the caller
    /// surface the specific operator that produced it.
    BinaryOperandTyMismatch {
        op: BinOp,
        lhs_ty: CgTy,
        rhs_ty: CgTy,
        span: Span,
    },

    /// An integer literal whose value does not fit into the chosen
    /// 32-bit narrowing target. The DSL surface uses `i64` literals; the
    /// CG IR's numeric literals are 32-bit. Out-of-range narrowings
    /// surface as this typed defect instead of silent truncation.
    LiteralOutOfRange {
        value: i64,
        target: CgTy,
        span: Span,
    },

    /// `if cond then a else b` whose `then` and `else` arms produce
    /// different types — a `CgExpr::Select` requires both arms to
    /// share the result type. Distinct from
    /// [`Self::BinaryOperandTyMismatch`] because Select is not a binary
    /// op; surfacing the typed variant keeps diagnostic readers from
    /// chasing a non-existent operator.
    SelectArmMismatch {
        then_ty: CgTy,
        else_ty: CgTy,
        span: Span,
    },

    /// A multi-operand builtin (`Min`, `Max`, `Clamp`, `SaturatingAdd`)
    /// whose operands disagree on the underlying numeric type. The
    /// builtin itself is well-formed; the operands are not. Distinct
    /// from [`Self::BinaryOperandTyMismatch`] because these builtins are
    /// not binary ops in the AST surface.
    BuiltinOperandMismatch {
        builtin: Builtin,
        lhs_ty: CgTy,
        rhs_ty: CgTy,
        span: Span,
    },

    /// A namespace call (`rng.<method>(<args>)`, etc.) with the wrong
    /// number of arguments for its expression-level lowering. Distinct
    /// from [`Self::BuiltinArityMismatch`] because namespace calls are
    /// not [`Builtin`]s — the DSL's namespace identifier is the typed
    /// surface, the method is a free-form identifier on the namespace.
    NamespaceCallArityMismatch {
        ns: NamespaceId,
        method: String,
        expected: usize,
        got: usize,
        span: Span,
    },

    /// A constructed `CgExpr` failed `type_check` — the lowering
    /// produced an internally-inconsistent node. Shouldn't normally fire
    /// (every arm constructs a `ty` derived from the operator), but the
    /// pass surfaces the typed error rather than panicking, so an
    /// emitter bug shows up as a typed report.
    TypeCheckFailure {
        error: TypeError,
        span: Span,
    },

    /// A `Field { base, field_name, .. }` access whose `field_name`
    /// doesn't map to any `AgentFieldId`. The base must already have
    /// been recognised as `self`; the field name is the offending part,
    /// so it appears here as a string (DSL field names are free-form
    /// identifiers — there's no closed enum for them in the AST).
    UnknownAgentField {
        field_name: String,
        span: Span,
    },

    /// A `Field { base, field_name, .. }` access whose `base` is not a
    /// shape the expression lowering recognises today (only `self.<f>`
    /// is wired through Task 2.1). Other bases — locals other than
    /// `self`, namespace fields, builder receivers — surface as this
    /// typed deferral so a follow-up task can light them up without a
    /// matching `_ =>` fallthrough.
    UnsupportedFieldBase {
        field_name: String,
        span: Span,
    },

    /// A binary operator the CG IR doesn't surface (`Mod` is the only
    /// one in the v1 surface). Carried as a typed variant so the
    /// rejection is structured rather than a string.
    UnsupportedBinaryOp {
        op: BinOp,
        span: Span,
    },

    /// A builtin call has the wrong number of arguments for its
    /// CG-side signature. Mirrors `TypeError::ArityMismatch` but
    /// surfaces *before* the CG node is constructed.
    BuiltinArityMismatch {
        builtin: Builtin,
        expected: u8,
        got: u8,
        span: Span,
    },

    /// A builtin maps to no `BuiltinId` yet — quantifier-style
    /// builtins (`Count`, `Sum`, `Forall`, `Exists`, plus the
    /// fold-shape `Min`/`Max` variants) lower to op-level constructs,
    /// not to `CgExpr::Builtin`. Surfacing them here as a typed deferral
    /// avoids silently dropping them.
    UnsupportedBuiltin {
        builtin: Builtin,
        span: Span,
    },

    /// Numeric builtins (`Min`, `Max`, `Clamp`, `SaturatingAdd`) are
    /// typed in the CG IR (`Min(NumericTy::F32)` vs `Min(NumericTy::U32)`),
    /// but the AST `Builtin` enum is untyped. The lowering picks the
    /// `NumericTy` from the operand type — if the operand isn't one of
    /// `{F32, U32, I32}`, this typed error fires.
    NumericBuiltinNonNumericOperand {
        builtin: Builtin,
        operand_index: u8,
        got: CgTy,
        span: Span,
    },

    /// A `Local` reference whose name isn't `self` (the only local the
    /// expression lowering binds to a CG concept today). Other locals —
    /// `target`, event-pattern bindings, fold binders — light up as the
    /// surrounding op-lowering tasks bind them.
    UnsupportedLocalBinding {
        name: String,
        span: Span,
    },

    /// A `ViewRef` referenced by an `IrExpr::ViewCall` has no entry in
    /// the lowering context's view map. Until Task 2.3 wires the global
    /// view table, expression-level tests inject the map directly; an
    /// unknown ref surfaces here.
    UnknownView {
        ast_ref: AstViewRef,
        span: Span,
    },

    /// `IrExpr::NamespaceCall` for a namespace / method pair that has
    /// no expression-level lowering today. Most namespace calls are
    /// op-level (spatial queries, RNG draws that produce typed ops);
    /// the few that fold into a single `CgExpr` (e.g., `rng.uniform`)
    /// are wired here as they're needed.
    UnsupportedNamespaceCall {
        ns: NamespaceId,
        method: String,
        span: Span,
    },

    /// `IrExpr::NamespaceField` for a namespace / field pair that has
    /// no expression-level CG lowering. `world.tick` and friends will
    /// arrive in later tasks once a concrete consumer needs them.
    UnsupportedNamespaceField {
        ns: NamespaceId,
        field: String,
        span: Span,
    },

    /// The builder rejected an `add_expr` / `add_op` call. Because the
    /// lowering only ever pushes children before parents, this should
    /// not fire under normal operation; surfacing it as a typed wrap
    /// makes any regression (a builder invariant tightening)
    /// immediately debuggable. Shared between expression-level pushes
    /// (Task 2.1) and op-level pushes (Tasks 2.2+).
    BuilderRejected {
        error: BuilderError,
        span: Span,
    },

    // -- Mask pass (Task 2.2) --------------------------------------------

    /// The mask's predicate body type-checked to a non-`Bool` type.
    /// Mask predicates write into a per-agent bitmap; the predicate
    /// value must be `Bool` so each tick's value is a single bit.
    /// Distinct from [`Self::IllTypedExpression`] because the constraint
    /// here is mask-level (the bitmap shape), not operator-level.
    MaskPredicateNotBool {
        mask: MaskId,
        got: CgTy,
        span: Span,
    },

    /// A type-check of the mask predicate node itself (after lowering)
    /// failed. Surfaces the underlying [`TypeError`] without panicking;
    /// should not normally fire because expression lowering already
    /// type-checks every node it constructs.
    MaskPredicateTypeCheckFailure {
        mask: MaskId,
        error: TypeError,
        span: Span,
    },

    /// The mask's `from <expr>` clause is a shape mask lowering does
    /// not recognise. v1 supports only
    /// `query.nearby_agents(<pos>, <radius>)`; any other shape (a bare
    /// view call, a different namespace method, a literal) surfaces
    /// here. Span points at the `from`-clause expression.
    UnsupportedMaskFromClause {
        mask: MaskId,
        span: Span,
    },

    /// The mask carries a parametric head (`mask Cast(ability:
    /// AbilityId)`, `mask MoveToward(target)`) without an explicit
    /// `from` clause. Today's [`crate::cg::dispatch::DispatchShape`]
    /// surface routes such heads to [`crate::cg::dispatch::DispatchShape::PerAgent`]
    /// by default, but the per-pair semantics implied by the head
    /// (e.g., `(agent × ability)` for the `Cast` example in
    /// `assets/sim/masks.sim`) cannot be represented until Task 2.6
    /// adds the matching [`crate::cg::dispatch::PerPairSource`] variant
    /// (e.g., `AbilityCatalog`). Until then the lowering refuses
    /// rather than silently miscompiling.
    ///
    /// `head_label` is a closed-set tag (`"positional"` | `"named"`)
    /// — it's a `&'static str` so the variant carries no free-form
    /// payload.
    UnsupportedMaskHeadShape {
        mask: MaskId,
        head_label: &'static str,
        span: Span,
    },

    /// The mask has a `from` clause but the caller supplied no
    /// [`SpatialQueryKind`]. The driver (Task 2.6 / 2.8) is responsible
    /// for resolving each from-bearing mask to a kin / engagement /
    /// future kind; a missing resolution is a driver defect, surfaced
    /// here as a typed error rather than a panic.
    MissingSpatialQueryKind {
        mask: MaskId,
        span: Span,
    },

    /// The caller supplied a [`SpatialQueryKind`] but the mask has no
    /// `from` clause. Catches the inverse bug (driver pre-allocated a
    /// kind for a self-only mask).
    UnexpectedSpatialQueryKind {
        mask: MaskId,
        kind: SpatialQueryKind,
        span: Span,
    },

    // -- View pass (Task 2.3) --------------------------------------------

    /// A statement form inside a view's fold-handler body that the CG
    /// statement language does not represent. Reasonable cases (a `Let`
    /// binding, a bare `Expr` statement, etc.) lower fine in later
    /// tasks; until then they surface here as a typed deferral so the
    /// caller knows precisely which AST node was rejected and where.
    ///
    /// `ast_label` is a closed-set tag drawn from `IrStmt`'s
    /// discriminants (`"Let"`, `"Expr"`, `"For"`, `"Match"`,
    /// `"BeliefObserve"`, `"Emit"`); kept as `&'static str` so the
    /// payload stays allocation-free.
    UnsupportedViewFoldStmt {
        view: ViewId,
        ast_label: &'static str,
        span: Span,
    },

    /// The driver supplied a per-handler resolution list whose length
    /// does not match the view's fold-handler count. Catches a driver
    /// invariant drift — every fold handler must have its
    /// `(EventKindId, EventRingId)` resolved before lowering can run.
    ViewHandlerResolutionLengthMismatch {
        view: ViewId,
        expected: usize,
        got: usize,
        span: Span,
    },

    /// A fold handler's `Assign` (lowered from a `self += <expr>`-style
    /// statement) writes to a [`ViewStorageSlot`] the view's storage
    /// hint does not expose. The mapping `StorageHint → valid slots` is
    /// documented on [`ViewStorageSlot`]; an out-of-bounds slot is a
    /// lowering defect, surfaced typed rather than silently producing a
    /// broken op.
    InvalidViewStorageSlot {
        view: ViewId,
        hint_label: &'static str,
        requested_slot: ViewStorageSlot,
        span: Span,
    },

    /// A `@lazy` view declared with at least one `@materialized`-only
    /// trait (storage hint, decay annotation) — or vice versa. The
    /// resolver normally rejects this; lowering surfaces the typed
    /// variant so a driver-supplied `IrView` that bypassed resolve
    /// (tests, snapshot fixtures) still lights up structurally.
    ///
    /// `kind_label` ("lazy" | "materialized") names which view-kind
    /// surface was claimed; `body_label` ("expr" | "fold") names what
    /// body shape was found.
    ViewKindBodyMismatch {
        view: ViewId,
        kind_label: &'static str,
        body_label: &'static str,
        span: Span,
    },
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // -- Expression pass --------------------------------------
            LoweringError::UnsupportedAstNode { ast_label, span } => write!(
                f,
                "lowering: AST variant `{}` at {}..{} is not yet supported",
                ast_label, span.start, span.end
            ),
            LoweringError::IllTypedExpression { expected, got, span } => write!(
                f,
                "lowering: expression at {}..{} is ill-typed — expected {}, got {}",
                span.start, span.end, expected, got
            ),
            LoweringError::BinaryOperandTyMismatch {
                op,
                lhs_ty,
                rhs_ty,
                span,
            } => write!(
                f,
                "lowering: binary `{:?}` at {}..{} has mismatched operands — lhs is {}, rhs is {}",
                op, span.start, span.end, lhs_ty, rhs_ty
            ),
            LoweringError::LiteralOutOfRange { value, target, span } => write!(
                f,
                "lowering: literal at {}..{} out of range — value {} does not fit in {}",
                span.start, span.end, value, target
            ),
            LoweringError::SelectArmMismatch {
                then_ty,
                else_ty,
                span,
            } => write!(
                f,
                "lowering: select at {}..{} has mismatched arms — then is {}, else is {}",
                span.start, span.end, then_ty, else_ty
            ),
            LoweringError::BuiltinOperandMismatch {
                builtin,
                lhs_ty,
                rhs_ty,
                span,
            } => write!(
                f,
                "lowering: builtin `{}` at {}..{} has mismatched operands — lhs is {}, rhs is {}",
                builtin.name(),
                span.start,
                span.end,
                lhs_ty,
                rhs_ty
            ),
            LoweringError::NamespaceCallArityMismatch {
                ns,
                method,
                expected,
                got,
                span,
            } => write!(
                f,
                "lowering: namespace call `{}.{}()` at {}..{} expected {} argument(s), got {}",
                ns.name(),
                method,
                span.start,
                span.end,
                expected,
                got
            ),
            LoweringError::TypeCheckFailure { error, span } => write!(
                f,
                "lowering: constructed CgExpr at {}..{} failed type-check — {}",
                span.start, span.end, error
            ),
            LoweringError::UnknownAgentField { field_name, span } => write!(
                f,
                "lowering: `self.{}` at {}..{} does not name an agent field",
                field_name, span.start, span.end
            ),
            LoweringError::UnsupportedFieldBase { field_name, span } => write!(
                f,
                "lowering: field access `.{}` at {}..{} has an unsupported base shape",
                field_name, span.start, span.end
            ),
            LoweringError::UnsupportedBinaryOp { op, span } => write!(
                f,
                "lowering: binary operator `{:?}` at {}..{} has no CG-IR equivalent",
                op, span.start, span.end
            ),
            LoweringError::BuiltinArityMismatch {
                builtin,
                expected,
                got,
                span,
            } => write!(
                f,
                "lowering: builtin `{}` at {}..{} expected {} argument(s), got {}",
                builtin.name(),
                span.start,
                span.end,
                expected,
                got
            ),
            LoweringError::UnsupportedBuiltin { builtin, span } => write!(
                f,
                "lowering: builtin `{}` at {}..{} has no expression-level CG equivalent",
                builtin.name(),
                span.start,
                span.end
            ),
            LoweringError::NumericBuiltinNonNumericOperand {
                builtin,
                operand_index,
                got,
                span,
            } => write!(
                f,
                "lowering: builtin `{}` at {}..{} operand[{}] expected numeric type, got {}",
                builtin.name(),
                span.start,
                span.end,
                operand_index,
                got
            ),
            LoweringError::UnsupportedLocalBinding { name, span } => write!(
                f,
                "lowering: local binding `{}` at {}..{} is not bound to a CG concept",
                name, span.start, span.end
            ),
            LoweringError::UnknownView { ast_ref, span } => write!(
                f,
                "lowering: ViewRef({}) at {}..{} not present in lowering context",
                ast_ref.0, span.start, span.end
            ),
            LoweringError::UnsupportedNamespaceCall { ns, method, span } => write!(
                f,
                "lowering: namespace call `{}.{}()` at {}..{} has no expression-level lowering",
                ns.name(),
                method,
                span.start,
                span.end
            ),
            LoweringError::UnsupportedNamespaceField { ns, field, span } => write!(
                f,
                "lowering: namespace field `{}.{}` at {}..{} has no expression-level lowering",
                ns.name(),
                field,
                span.start,
                span.end
            ),
            LoweringError::BuilderRejected { error, span } => write!(
                f,
                "lowering: builder rejected node at {}..{} — {}",
                span.start, span.end, error
            ),

            // -- Mask pass --------------------------------------------
            LoweringError::MaskPredicateNotBool { mask, got, span } => write!(
                f,
                "mask#{} predicate at {}..{} produced {} — must be Bool",
                mask.0, span.start, span.end, got
            ),
            LoweringError::MaskPredicateTypeCheckFailure { mask, error, span } => write!(
                f,
                "mask#{} predicate at {}..{} failed type-check — {}",
                mask.0, span.start, span.end, error
            ),
            LoweringError::UnsupportedMaskFromClause { mask, span } => write!(
                f,
                "mask#{} `from` clause at {}..{} has an unsupported shape — only `query.nearby_agents(<pos>, <radius>)` is recognised",
                mask.0, span.start, span.end
            ),
            LoweringError::UnsupportedMaskHeadShape {
                mask,
                head_label,
                span,
            } => write!(
                f,
                "mask#{} head shape `{}` at {}..{} requires a `from` clause — parametric heads without an explicit dispatch source are not yet routable (Task 2.6)",
                mask.0, head_label, span.start, span.end
            ),
            LoweringError::MissingSpatialQueryKind { mask, span } => write!(
                f,
                "mask#{} at {}..{} has a `from` clause but no SpatialQueryKind was supplied by the driver",
                mask.0, span.start, span.end
            ),
            LoweringError::UnexpectedSpatialQueryKind { mask, kind, span } => write!(
                f,
                "mask#{} at {}..{} has no `from` clause but the driver supplied SpatialQueryKind::{}",
                mask.0, span.start, span.end, kind
            ),

            // -- View pass --------------------------------------------
            LoweringError::UnsupportedViewFoldStmt {
                view,
                ast_label,
                span,
            } => write!(
                f,
                "view#{} fold body at {}..{} contains AST statement `{}` which has no CG-statement equivalent yet",
                view.0, span.start, span.end, ast_label
            ),
            LoweringError::ViewHandlerResolutionLengthMismatch {
                view,
                expected,
                got,
                span,
            } => write!(
                f,
                "view#{} at {}..{} has {} fold handler(s) but the driver supplied {} (EventKindId, EventRingId) entries",
                view.0, span.start, span.end, expected, got
            ),
            LoweringError::InvalidViewStorageSlot {
                view,
                hint_label,
                requested_slot,
                span,
            } => write!(
                f,
                "view#{} fold body at {}..{} writes to {} slot which is not exposed by the `{}` storage hint",
                view.0, span.start, span.end, requested_slot, hint_label
            ),
            LoweringError::ViewKindBodyMismatch {
                view,
                kind_label,
                body_label,
                span,
            } => write!(
                f,
                "view#{} at {}..{} declared as `@{}` but has a `{}` body — `@lazy` views require an expression body and `@materialized` views require a fold body",
                view.0, span.start, span.end, kind_label, body_label
            ),
        }
    }
}

impl std::error::Error for LoweringError {}
