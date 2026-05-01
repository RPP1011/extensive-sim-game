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

use crate::cg::data_handle::{ConfigConstId, MaskId, ViewId, ViewStorageSlot};
use crate::cg::expr::{CgTy, TypeError};
use crate::cg::op::{ActionId, PhysicsRuleId, ScoringId, SpatialQueryKind};
use crate::cg::program::BuilderError;
use crate::cg::stmt::VariantId;
use crate::cg::well_formed::CgError;

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

    /// A bare `IrExpr::Local(local_ref, name)` resolved through
    /// [`super::expr::LoweringCtx::local_ids`] to a typed
    /// [`crate::cg::stmt::LocalId`], but the matching
    /// `LocalId → CgTy` entry in
    /// [`super::expr::LoweringCtx::local_tys`] was missing. The
    /// driver populates `local_tys` as part of `IrStmt::Let` lowering
    /// (`record_local_ty`); a missing entry is either a stale
    /// registry or a hand-built AST whose let-binding was lowered
    /// without recording the type. Distinct from
    /// [`Self::UnsupportedLocalBinding`] — that variant fires when
    /// the *name* is unknown; this one fires when the *type* of a
    /// bound name is unknown. Task 5.5d.
    UnknownLocalType {
        local: crate::cg::stmt::LocalId,
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

    /// `IrExpr::NamespaceField { ns: Config, field, .. }` whose
    /// `(ns, field)` pair has no [`ConfigConstId`] in
    /// [`super::expr::LoweringCtx::config_const_ids`]. Either the
    /// driver's `populate_config_consts` walk missed an entry (driver
    /// defect) or the source references a field declared in no
    /// `config <Block> { ... }` block (resolver should have caught it,
    /// but this surface stays defensive). Distinct from
    /// [`Self::UnsupportedNamespaceField`], which now only fires for
    /// non-`Config` namespace fields.
    UnknownConfigField {
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

    // -- Physics pass (Task 2.4) -----------------------------------------

    /// The driver supplied a per-handler resolution list whose length
    /// does not match the physics rule's handler count. Mirrors the
    /// view-pass [`Self::ViewHandlerResolutionLengthMismatch`] —
    /// every physics handler must have its `(EventKindId,
    /// EventRingId)` resolved before lowering can run.
    PhysicsHandlerResolutionLengthMismatch {
        rule: PhysicsRuleId,
        expected: usize,
        got: usize,
        span: Span,
    },

    /// A statement form inside a physics rule's handler body that the
    /// CG statement language does not represent yet. Reasonable cases
    /// — `Let` (local-binding desugaring still on the driver),
    /// `For` (iteration over `abilities.effects(ab)` in the `cast`
    /// rule, deferred), `BeliefObserve` (decomposition into
    /// BeliefState SoA assigns), bare `Expr` (namespace setter calls
    /// like `agents.set_hp(t, x)` that need namespace lowering),
    /// `SelfUpdate` (forbidden in physics — only valid inside view
    /// fold bodies) — surface here as a typed deferral with a closed-
    /// set tag so the caller knows precisely which AST node was
    /// rejected.
    ///
    /// `ast_label` is a closed-set `&'static str` tag drawn from
    /// `IrStmt`'s discriminants (`"Let"`, `"For"`, `"Match"`,
    /// `"BeliefObserve"`, `"Expr"`, `"SelfUpdate"`).
    UnsupportedPhysicsStmt {
        rule: PhysicsRuleId,
        ast_label: &'static str,
        span: Span,
    },

    /// A `Match` arm's `IrPattern::Struct { name, .. }` references a
    /// variant name that the lowering context's `variant_ids`
    /// registry does not know. Driver populates the registry from the
    /// stdlib enum surface (today `EffectOp`); a missing entry is
    /// either a stale registry or a hand-built AST referencing a
    /// variant outside the stdlib enums. Surfaced with the
    /// source-level name so the diagnostic can name the offending
    /// variant.
    UnknownMatchVariant {
        rule: PhysicsRuleId,
        variant_name: String,
        span: Span,
    },

    /// A `Match` arm carries an `IrPattern` shape the physics
    /// lowering does not recognise as a match arm. Today only
    /// `IrPattern::Struct { name, bindings }` is wired (the shape
    /// stdlib `EffectOp` matches uses); other shapes (`Bind`,
    /// `Ctor`, `Expr`, `Wildcard`) lower as this typed deferral
    /// rather than silently routing through the wrong arm.
    ///
    /// `pattern_label` is a closed-set `&'static str` tag drawn from
    /// `IrPattern`'s discriminants.
    UnsupportedMatchPattern {
        rule: PhysicsRuleId,
        pattern_label: &'static str,
        span: Span,
    },

    /// A `Match` arm's pattern binding (e.g., `Damage { amount }`'s
    /// `amount`) carries an inner `IrPattern` shape the physics
    /// lowering does not recognise. Today only the shorthand bind
    /// shape (`IrPattern::Bind { name, local }`) is wired — that
    /// matches the canonical `Damage { amount }` form where the
    /// field name and binder name are the same identifier. Aliased
    /// binds (`Damage { amount: a }`) parse as a nested `Bind` too;
    /// the resolver flattens both into the same shape. Other shapes
    /// (literal patterns, nested ctors, wildcards inside a struct
    /// binding) surface here.
    UnsupportedMatchBindingShape {
        rule: PhysicsRuleId,
        field_name: String,
        pattern_label: &'static str,
        span: Span,
    },

    /// A pattern binder references an AST `LocalRef` that the
    /// lowering context's `local_ids` registry does not know. The
    /// driver populates the map per-handler from the resolver's
    /// scope tracker; a missing entry is either a stale registry or
    /// a hand-built AST referencing a synthetic local. The error
    /// names the source-level binder identifier so the diagnostic
    /// can pinpoint the offending name.
    UnknownLocalRef {
        rule: PhysicsRuleId,
        binder_name: String,
        span: Span,
    },

    /// A `Match` carries no arms — physics lowering refuses to
    /// produce a `CgStmt::Match` with an empty arm list because the
    /// resulting op would have no defined behaviour at runtime
    /// (every dispatch on the scrutinee falls through unmatched).
    /// The resolver normally rejects empty-arm matches at parse
    /// time; defense-in-depth here catches synthetic IR.
    EmptyMatchArms {
        rule: PhysicsRuleId,
        span: Span,
    },

    /// A physics `Emit { event_name, fields }` carries a field name
    /// the driver-supplied event-field schema (LoweringCtx
    /// `event_field_indices`) does not know. The driver populates
    /// the schema from each event variant's declared field list
    /// (Task 5.7); a missing entry surfaces here with the resolved
    /// [`crate::cg::op::EventKindId`] and the source-level field
    /// name.
    UnknownEventField {
        event: crate::cg::op::EventKindId,
        field_name: String,
        span: Span,
    },

    // -- Scoring pass (Task 2.5) -----------------------------------------

    /// A scoring row's `utility` expression (the score) type-checked to
    /// a non-`F32` type. Scoring utilities are scalar floats —
    /// `engine::scoring` accumulates them with `+` and picks the
    /// argmax — so an integer / agent-id / bool utility is rejected.
    /// Distinct from [`Self::IllTypedExpression`] because the
    /// constraint here is scoring-row-level (the argmax shape requires
    /// F32), not operator-level.
    ScoringUtilityNotF32 {
        scoring: ScoringId,
        action: ActionId,
        got: CgTy,
        span: Span,
    },

    /// A scoring row's `target` expression (per-ability rows only)
    /// type-checked to a non-[`CgTy::AgentId`] type. Per-ability rows
    /// require the target expression to resolve to an agent id
    /// because the engine applies the action to that specific agent
    /// when the row wins the argmax.
    ScoringTargetNotAgentId {
        scoring: ScoringId,
        action: ActionId,
        got: CgTy,
        span: Span,
    },

    /// A scoring row's `guard` expression (per-ability rows only)
    /// type-checked to a non-`Bool` type. Guards are boolean
    /// predicates — `None` parses as `true`; an explicit guard must
    /// evaluate to `Bool` so the kernel can short-circuit the row
    /// when the predicate fails.
    ScoringGuardNotBool {
        scoring: ScoringId,
        action: ActionId,
        got: CgTy,
        span: Span,
    },

    /// A scoring row references a source-level action name that the
    /// lowering context's `action_ids` registry does not know.
    /// Mirrors the physics pass's
    /// [`Self::UnknownMatchVariant`] precedent — both surface a
    /// missing-name resolution as a dedicated typed variant rather
    /// than overloading a sibling type-check failure.
    ///
    /// The driver (Task 2.7 / 2.8) populates `action_ids` from the
    /// action surface; tests register names directly via
    /// [`super::expr::LoweringCtx::register_action`]. A missing entry
    /// is either a stale registry or a hand-built AST referencing a
    /// synthetic action name. The error names the source-level
    /// identifier so the diagnostic can pinpoint the offending name.
    UnknownScoringAction {
        scoring: ScoringId,
        name: String,
        span: Span,
    },

    /// A scoring row carries a parametric head (`Attack(target)`,
    /// `Cast(ability: AbilityId)`) that the scoring lowering does not
    /// route. Today's scoring rows in `assets/sim/scoring.sim`
    /// exclusively use `IrActionHeadShape::None` (bare action names —
    /// the per-action argmax kernel resolves the target implicitly
    /// from the action kind). Mirrors the mask pass's
    /// [`Self::UnsupportedMaskHeadShape`] precedent — surface the
    /// gate so a future scoring row with a parametric head fails
    /// loudly rather than silently dropping the binders.
    ///
    /// `action` is `Option` because the head-shape gate fires before
    /// action-id resolution succeeds — a parametric head may never
    /// resolve to an `ActionId` if the registry only knows the bare
    /// name.
    ///
    /// `head_label` is a closed-set tag (`"positional"` | `"named"`)
    /// — `&'static str` so the variant carries no free-form payload,
    /// matching mask's pattern.
    UnsupportedScoringHeadShape {
        scoring: ScoringId,
        action: Option<ActionId>,
        head_label: &'static str,
        span: Span,
    },

    /// A type-check of a scoring-row's expression node itself (after
    /// lowering) failed. Surfaces the underlying [`TypeError`] without
    /// panicking; should not normally fire because expression
    /// lowering already type-checks every node it constructs.
    /// `subject` names which sub-expression of the row tripped the
    /// check ([`ScoringRowSubject::Utility`] |
    /// [`ScoringRowSubject::Target`] | [`ScoringRowSubject::Guard`]).
    ScoringRowTypeCheckFailure {
        scoring: ScoringId,
        action: ActionId,
        subject: ScoringRowSubject,
        error: TypeError,
        span: Span,
    },

    // -- (continued: pre-existing variants) -----------------------------

    /// A fold-handler `IrStmt::SelfUpdate` carries an operator the CG
    /// IR's `ComputeOpKind::ViewFold` wrapper does not yet thread.
    /// Today only `+=` is lowered — the merge semantics for `=`,
    /// `-=`, `*=`, `/=` would silently lower to identical CG IR as
    /// `+=` because the operator is not represented in the op kind
    /// (see the module-level "Statement-body coverage" docs on
    /// `view.rs`). Defense-in-depth gate: the resolver enforces the
    /// 5-element vocabulary at parse time, but the AST holds the
    /// operator as a free-form `String`, so this error also catches
    /// resolver-permitted-but-unsupported variants and any unknown
    /// string a synthetic AST might smuggle through.
    ///
    /// `op_label` is a closed-set tag drawn from the canonical
    /// operator strings (`"="` | `"+="` | `"-="` | `"*="` | `"/="`),
    /// or `"unknown"` for unrecognised strings. When
    /// `ComputeOpKind::ViewFold` gains an explicit operator field
    /// (Task 2.8 / driver-IR shape change), this gate goes away.
    UnsupportedFoldOperator {
        view: ViewId,
        op_label: &'static str,
        span: Span,
    },

    // -- Driver pass (Task 2.8) ------------------------------------------

    /// A view fold-handler or physics-rule kind-pattern names an
    /// event the driver's event-kind registry could not resolve.
    /// Carries the source-level event name and the offending
    /// pattern's span so the diagnostic can pinpoint the
    /// reference.
    ///
    /// Surfaces in two cases today: (a) the resolver populated
    /// `pattern.event` but the [`dsl_ast::ir::EventRef`] points
    /// past the registry's table (driver/registry drift); (b) the
    /// resolver left `pattern.event` as `None` (parse-time
    /// failure that didn't abort resolution). Both are
    /// driver-level defects, surfaced typed rather than panicking.
    UnresolvedEventPattern {
        event_name: String,
        span: Span,
    },

    /// The driver's `check_well_formed` gate flagged a
    /// user-op-only program defect. Wraps the typed [`CgError`]
    /// produced by [`crate::cg::well_formed::check_well_formed`]
    /// without losing the structural payload.
    ///
    /// Carries no span — `CgError` variants pin their own
    /// op/expr/list ids; the driver passes them through unchanged.
    WellFormed {
        error: CgError,
    },

    /// The driver's variant-registry population observed two enum
    /// variants with the same source-level name (e.g. `Damage`
    /// declared in two distinct enums). The registry is a flat
    /// name → id map; the second occurrence overwrites the first,
    /// which would silently mis-route physics `Match` arms naming
    /// the colliding name. The driver flags it but does not abort —
    /// physics matches today only reference stdlib `EffectOp`
    /// variants, so a real-world collision is unlikely.
    DuplicateVariantInRegistry {
        /// The source-level variant identifier that collided.
        name: String,
        /// The id the registry already held for `name` before the
        /// collision.
        prior_id: VariantId,
        /// The id the driver tried to register for `name` (the
        /// collision write — last-write-wins, so this is the id the
        /// registry holds after the call).
        new_id: VariantId,
    },

    /// The driver's view-registry population observed two views
    /// resolving to the same typed [`ViewId`] — a driver-side defect
    /// since [`ViewId`]s are allocated in source order. The driver
    /// flags it but does not abort.
    DuplicateViewInRegistry {
        /// The AST-level view ref the driver attempted to register.
        ast_ref: AstViewRef,
        /// The view id that was already registered for the same AST
        /// ref before this attempt.
        prior_id: ViewId,
        /// The id the driver tried to register (the collision write —
        /// last-write-wins, so this is the id the registry holds
        /// after the call).
        new_id: ViewId,
    },

    /// Two `(block, field)` pairs in `Compilation::configs` resolved
    /// to the same registry key. Driver-side defect; last-write-wins
    /// semantics in place. Mirrors
    /// [`Self::DuplicateViewInRegistry`].
    DuplicateConfigConstInRegistry {
        key: String,
        prior_id: ConfigConstId,
        new_id: ConfigConstId,
    },

    /// A `@lazy` view body was registered twice for the same
    /// [`ViewId`] — driver-side defect. Last-write-wins semantics in
    /// place; surfacing the variant lets tests assert exclusive
    /// allocation. Task 5.5c.
    DuplicateLazyViewBodyRegistration {
        view: ViewId,
        span: Span,
    },

    /// A `@lazy` view call's argument count does not match the
    /// registered body's parameter count. Surfaces during inlining
    /// when [`super::expr::lower_view_call`] consults
    /// [`super::expr::LoweringCtx::lazy_view_bodies`]. Task 5.5c.
    ViewCallArityMismatch {
        view: ViewId,
        expected: usize,
        got: usize,
        span: Span,
    },
}

/// Closed-set discriminant for which sub-expression of a scoring row
/// tripped a type-check failure. Replaces the previous `&'static str`
/// payload on [`LoweringError::ScoringRowTypeCheckFailure`] so the
/// vocabulary is enforced by the type system rather than by string
/// convention. Mirrors the typed-enum precedent established by prior
/// Phase 2 tasks for closed-set discriminants.
///
/// `Display` renders the canonical lowercase tag (`"utility"`,
/// `"target"`, `"guard"`) — preserves the exact format produced by the
/// previous `&'static str` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoringRowSubject {
    /// The row's `utility` (score) expression — must type-check to
    /// `CgTy::F32`.
    Utility,
    /// A per-ability row's `target` expression — must type-check to
    /// `CgTy::AgentId`.
    Target,
    /// A per-ability row's `guard` expression — must type-check to
    /// `CgTy::Bool`.
    Guard,
}

impl fmt::Display for ScoringRowSubject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScoringRowSubject::Utility => f.write_str("utility"),
            ScoringRowSubject::Target => f.write_str("target"),
            ScoringRowSubject::Guard => f.write_str("guard"),
        }
    }
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
            LoweringError::UnknownLocalType { local, span } => write!(
                f,
                "lowering: bare local {} at {}..{} resolved to a binding with no recorded CgTy",
                local, span.start, span.end
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
            LoweringError::UnknownConfigField { ns, field, span } => write!(
                f,
                "lowering: config field `{}.{}` at {}..{} has no ConfigConstId — driver registry missed it",
                ns.name(),
                field,
                span.start,
                span.end,
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
            LoweringError::UnsupportedFoldOperator {
                view,
                op_label,
                span: _,
            } => write!(
                f,
                "view #{} self-update operator {} not supported by CG IR; only += is lowered today",
                view.0, op_label
            ),

            // -- Physics pass -----------------------------------------
            LoweringError::PhysicsHandlerResolutionLengthMismatch {
                rule,
                expected,
                got,
                span,
            } => write!(
                f,
                "physics#{} at {}..{} has {} handler(s) but the driver supplied {} (EventKindId, EventRingId) entries",
                rule.0, span.start, span.end, expected, got
            ),
            LoweringError::UnsupportedPhysicsStmt {
                rule,
                ast_label,
                span,
            } => write!(
                f,
                "physics#{} body at {}..{} contains AST statement `{}` which has no CG-statement equivalent yet",
                rule.0, span.start, span.end, ast_label
            ),
            LoweringError::UnknownMatchVariant {
                rule,
                variant_name,
                span,
            } => write!(
                f,
                "physics#{} match arm at {}..{} references variant `{}` which is not registered in the lowering context",
                rule.0, span.start, span.end, variant_name
            ),
            LoweringError::UnsupportedMatchPattern {
                rule,
                pattern_label,
                span,
            } => write!(
                f,
                "physics#{} match arm at {}..{} uses pattern shape `{}` — only `Struct {{ Name {{ binders }} }}` is recognised today",
                rule.0, span.start, span.end, pattern_label
            ),
            LoweringError::UnsupportedMatchBindingShape {
                rule,
                field_name,
                pattern_label,
                span,
            } => write!(
                f,
                "physics#{} match arm at {}..{} binding for field `{}` uses pattern shape `{}` — only shorthand `Bind` is recognised today",
                rule.0, span.start, span.end, field_name, pattern_label
            ),
            LoweringError::UnknownLocalRef {
                rule,
                binder_name,
                span,
            } => write!(
                f,
                "physics#{} match-arm binder `{}` at {}..{} resolves to a LocalRef not registered in the lowering context",
                rule.0, binder_name, span.start, span.end
            ),
            LoweringError::EmptyMatchArms { rule, span } => write!(
                f,
                "physics#{} match at {}..{} has no arms — empty matches have no defined runtime behaviour",
                rule.0, span.start, span.end
            ),
            LoweringError::UnknownEventField {
                event,
                field_name,
                span,
            } => write!(
                f,
                "lowering: emit field `{}` at {}..{} is not registered for event#{} in the event-field schema",
                field_name, span.start, span.end, event.0
            ),

            // -- Scoring pass -----------------------------------------
            LoweringError::ScoringUtilityNotF32 {
                scoring,
                action,
                got,
                span,
            } => write!(
                f,
                "scoring#{} row(action=#{}) utility at {}..{} produced {} — must be F32",
                scoring.0, action.0, span.start, span.end, got
            ),
            LoweringError::ScoringTargetNotAgentId {
                scoring,
                action,
                got,
                span,
            } => write!(
                f,
                "scoring#{} row(action=#{}) target at {}..{} produced {} — must be AgentId",
                scoring.0, action.0, span.start, span.end, got
            ),
            LoweringError::ScoringGuardNotBool {
                scoring,
                action,
                got,
                span,
            } => write!(
                f,
                "scoring#{} row(action=#{}) guard at {}..{} produced {} — must be Bool",
                scoring.0, action.0, span.start, span.end, got
            ),
            LoweringError::UnknownScoringAction {
                scoring,
                name,
                span,
            } => write!(
                f,
                "scoring #{} references unknown action `{}` at {}..{}",
                scoring.0, name, span.start, span.end
            ),
            LoweringError::UnsupportedScoringHeadShape {
                scoring,
                action,
                head_label,
                span,
            } => {
                let action_label = match action {
                    Some(a) => format!("#{}", a.0),
                    None => String::from("<unresolved>"),
                };
                write!(
                    f,
                    "scoring#{} action `{}` head shape `{}` at {}..{} not supported — only bare action names are routable today",
                    scoring.0, action_label, head_label, span.start, span.end
                )
            }
            LoweringError::ScoringRowTypeCheckFailure {
                scoring,
                action,
                subject,
                error,
                span,
            } => write!(
                f,
                "scoring#{} row(action=#{}) {} at {}..{} failed type-check — {}",
                scoring.0, action.0, subject, span.start, span.end, error
            ),

            // -- Driver pass ------------------------------------------
            LoweringError::UnresolvedEventPattern { event_name, span } => write!(
                f,
                "lowering: event pattern `{}` at {}..{} could not be resolved against the event-kind registry",
                event_name, span.start, span.end
            ),
            LoweringError::WellFormed { error } => write!(
                f,
                "lowering: user-op-only well-formedness gate flagged: {}",
                error
            ),
            LoweringError::DuplicateVariantInRegistry {
                name,
                prior_id,
                new_id,
            } => write!(
                f,
                "lowering: duplicate variant `{}` in registry — prior id #{}, new id #{} (last-write-wins)",
                name, prior_id.0, new_id.0
            ),
            LoweringError::DuplicateViewInRegistry {
                ast_ref,
                prior_id,
                new_id,
            } => write!(
                f,
                "lowering: duplicate view AstViewRef(#{}) in registry — prior ViewId(#{}), new ViewId(#{}) (last-write-wins)",
                ast_ref.0, prior_id.0, new_id.0
            ),
            LoweringError::DuplicateConfigConstInRegistry {
                key,
                prior_id,
                new_id,
            } => write!(
                f,
                "lowering: duplicate config const `{}` in registry — prior ConfigConstId(#{}), new ConfigConstId(#{}) (last-write-wins)",
                key, prior_id.0, new_id.0
            ),
            LoweringError::DuplicateLazyViewBodyRegistration { view, span } => write!(
                f,
                "lowering: lazy view body for ViewId(#{}) registered twice at {}..{} (driver defect)",
                view.0, span.start, span.end
            ),
            LoweringError::ViewCallArityMismatch {
                view,
                expected,
                got,
                span,
            } => write!(
                f,
                "lowering: view#{} call at {}..{} expected {} argument(s), got {}",
                view.0, span.start, span.end, expected, got
            ),
        }
    }
}

impl std::error::Error for LoweringError {}
