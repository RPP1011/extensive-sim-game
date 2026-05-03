//! Expression lowering — `IrExprNode → CgExprId`.
//!
//! Walks resolved DSL IR (`dsl_ast::ir::IrExprNode`) and pushes nodes
//! into a [`CgProgramBuilder`]. Every constructed [`CgExpr`] is
//! type-checked via [`crate::cg::expr::type_check`] before its id is
//! returned, so a successful lowering produces a node whose claimed
//! [`CgTy`] matches its operand types.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 2.1, for the design rationale and step list.
//!
//! # Diagnostics vs hard errors
//!
//! `lower_expr` operates on a *single* `IrExprNode` — the expression
//! tree. If lowering fails (anywhere in the tree), the caller gets back
//! a [`LoweringError`] and no node is pushed. This is the unit at this
//! layer; the next-layer-up driver (mask / scoring / fold lowering, in
//! later tasks) decides whether to accumulate per-rule diagnostics or
//! short-circuit. Diagnostic accumulation lives on
//! [`LoweringCtx::diagnostics`] for that future use; this pass does not
//! push to it directly.

use std::collections::HashMap;

use dsl_ast::ast::{BinOp, Span, UnOp};
use dsl_ast::ir::{
    Builtin, IrCallArg, IrExpr, IrExprNode, LocalRef, NamespaceId, ViewRef as AstViewRef,
};

use crate::cg::data_handle::{
    AgentFieldId, AgentFieldTy, AgentRef, CgExprId, ConfigConstId, DataHandle, RngPurpose, ViewId,
};
use crate::cg::expr::{
    data_handle_ty, type_check, BinaryOp, BuiltinId, CgExpr, CgTy, LitValue, NumericTy,
    TypeCheckCtx, TypeError, UnaryOp,
};
use crate::cg::op::{ActionId, EventKindId};
use crate::cg::program::CgProgramBuilder;
use crate::cg::stmt::{CgStmt, LocalId, VariantId};

pub use super::error::LoweringError;

// ---------------------------------------------------------------------------
// LoweringCtx
// ---------------------------------------------------------------------------

/// Context threaded through expression lowering.
///
/// Carries the in-flight [`CgProgramBuilder`] (the recipient of every
/// `add_expr` call) plus typed lookup tables that map AST resolved-ids
/// to CG newtype ids. The maps are populated by the surrounding
/// op-lowering driver (Task 2.7); for Task 2.1's tests, they're built
/// directly (typically empty).
pub struct LoweringCtx<'a> {
    /// Builder receiving every freshly-allocated [`CgExpr`].
    pub builder: &'a mut CgProgramBuilder,
    /// AST `ViewRef` → CG `ViewId` map. Empty at expression-pass tests
    /// that don't exercise `IrExpr::ViewCall`; populated by the driver.
    pub view_ids: HashMap<AstViewRef, ViewId>,
    /// Optional view signature resolver — passed through to
    /// [`type_check`]. `None` means `IrExpr::ViewCall` lowering itself
    /// pins the result type from `view_ids` but the type checker can't
    /// validate operand types; the builder's `validate_expr_refs`
    /// catches dangling ids and the lowering catches arity at AST level.
    pub view_signatures: HashMap<ViewId, (Vec<CgTy>, CgTy)>,
    /// Sum-type variant-name → typed [`VariantId`] resolver, keyed on
    /// the source-level variant identifier (`"Damage"`, `"Heal"`, …).
    /// Used by physics-pass `Match` lowering to resolve arm patterns
    /// to their typed variant id. The driver populates this from the
    /// stdlib enum registry (today only `EffectOp` is matched); tests
    /// populate the map directly.
    ///
    /// **Distinct from [`Self::event_kind_ids`].** Sum-type variants
    /// (matched in arms) and event kinds (named in `Emit` and fold
    /// handlers) inhabit independent id spaces. A driver populating
    /// both with the natural per-sequence allocation pattern (e.g.,
    /// `Damage → 0` and `AgentDied → 0`) must not let an emit-name
    /// resolution route through this map — that's exactly the silent
    /// mis-routing the split prevents.
    pub variant_ids: HashMap<String, VariantId>,
    /// Event-name → typed [`EventKindId`] resolver, keyed on the
    /// source-level event variant identifier (`"AgentDied"`,
    /// `"ChronicleEntry"`, …). Used by physics `Emit` lowering and
    /// view-fold handler resolution to map the event name in the
    /// source surface to the typed id the IR carries. The driver
    /// populates this from the event registry; tests populate it
    /// directly.
    ///
    /// **Distinct from [`Self::variant_ids`].** See its doc for the
    /// rationale.
    pub event_kind_ids: HashMap<String, EventKindId>,
    /// Per-event field-name → field-index resolver, keyed on the
    /// `(EventKindId, field_name)` pair. Populated by the driver
    /// (Task 5.7) from each event variant's declared field list;
    /// tests populate it directly via [`Self::register_event_field`].
    /// Used by physics `Emit` lowering to resolve each
    /// `IrFieldInit { name, value, .. }` to a typed
    /// [`crate::cg::stmt::EventField`] with `(event, index)`. A
    /// missing entry surfaces as
    /// [`LoweringError::UnknownEventField`].
    pub event_field_indices: HashMap<(EventKindId, String), u8>,
    /// AST [`LocalRef`] → typed [`LocalId`] resolver. Pattern binders
    /// resolved by the AST resolver carry a `LocalRef`; physics-pass
    /// `Match` lowering converts each binding's local through this
    /// map. The driver populates it; tests populate it directly.
    pub local_ids: HashMap<LocalRef, LocalId>,
    /// Action-name → typed [`ActionId`] resolver. Scoring-row heads
    /// (`Hold`, `MoveToward`, `Attack`, … and `row <name>
    /// per_ability` row names) resolve through this map. The driver
    /// populates this from the action surface (one allocation per
    /// distinct head name across the scoring decl); tests populate it
    /// directly via [`Self::register_action`].
    ///
    /// Standard scoring rows AND per-ability scoring rows share this
    /// id space — both are "actions" in the engine's apply layer (the
    /// engine maps the winning [`ActionId`] to a behaviour). The
    /// driver allocates each distinct name to a unique id; using the
    /// same map for both row shapes preserves the contract.
    pub action_ids: HashMap<String, ActionId>,
    /// Accumulator for per-rule diagnostics. The expression lowering
    /// itself returns `Err` on first defect; this vector exists so
    /// later op-lowering passes can collect non-fatal rule-level
    /// diagnostics in the same context.
    pub diagnostics: Vec<LoweringError>,
    /// Whether `target` is bound as the per-pair candidate in the
    /// current lowering context.
    ///
    /// Set by op-level driver passes that lower a pair-bound construct
    /// (`mask <Name>(target) from query.nearby_agents(...)` today;
    /// Task 5.5b/c will extend this to per-pair scoring rows and
    /// fold-body event binders). When `true`, `target.<field>` accesses
    /// in the predicate / body resolve to a `Read(AgentField {
    /// field, target: AgentRef::PerPairCandidate })` — see
    /// [`AgentRef::PerPairCandidate`]'s docstring for the resolution
    /// contract. When `false` (the default), any `target.<field>` access
    /// surfaces as [`LoweringError::UnsupportedFieldBase`] (the same
    /// shape as other unbound receivers) so the driver-side invariant is
    /// enforced at every layer.
    pub target_local: bool,
    /// `(NamespaceId::Config, "<block>.<field>")` → typed `ConfigConstId`
    /// resolver. Populated by the driver's
    /// [`super::driver::populate_config_consts`] walk over
    /// `Compilation::configs` (one id per block × field, allocated in
    /// source order). Used by `IrExpr::NamespaceField`'s expression
    /// lowering to map a `config.<block>.<field>` access to
    /// `Read(DataHandle::ConfigConst { id })`. An unknown
    /// `(ns, field)` pair surfaces as
    /// [`LoweringError::UnknownConfigField`]; the legacy
    /// [`LoweringError::UnsupportedNamespaceField`] now only fires for
    /// non-`Config` namespaces.
    pub config_const_ids: HashMap<(NamespaceId, String), ConfigConstId>,
    /// Captured `@lazy` view bodies for at-call-site inlining.
    /// `ViewId` → snapshot. Populated by
    /// [`super::view::lower_view`]'s lazy arm in Phase 2; consumed by
    /// [`lower_view_call`]. A view absent from this map is materialized
    /// — the call lowers through `BuiltinId::ViewCall { view }` as
    /// before, with the type checker resolving against
    /// `ctx.view_signatures`. Task 5.5c.
    pub lazy_view_bodies: HashMap<ViewId, LazyViewSnapshot>,
    /// Typed `LocalId → CgTy` map. Populated by `IrStmt::Let` lowering
    /// (Task 5.5b/d) at the moment each binding's CG type becomes
    /// known. Used by `IrExpr::Local` resolution (Task 5.5d) to
    /// reconstruct `CgExpr::ReadLocal { local, ty }` for bare-local
    /// reads.
    ///
    /// Distinct from [`Self::local_ids`]: that map carries
    /// `LocalRef → LocalId` (binder identity), this one carries
    /// `LocalId → CgTy` (binder type).
    pub local_tys: HashMap<LocalId, CgTy>,
    /// Per-event-kind payload layouts. Populated by the driver's
    /// `populate_event_kinds` walk over the event registry; consumed by
    /// physics + view-fold handler lowering when synthesizing
    /// `CgStmt::Let` for each event-pattern binding (the `actor: c,
    /// target: t, amount: a` shape introduces three locals whose
    /// values come from typed `CgExpr::EventField` reads keyed on this
    /// schema). At `finish()` time the lowering driver copies this
    /// table onto [`crate::cg::program::CgProgram::event_layouts`] so
    /// the WGSL emit can resolve the layout per-kind without a
    /// separate registry walk.
    pub event_layouts: HashMap<EventKindId, super::super::program::EventLayout>,
    /// Stdlib namespace registry — schema for `CgExpr::NamespaceCall`
    /// and `CgExpr::NamespaceField` lowering. Populated by the driver's
    /// `populate_namespace_registry`; consumed by `lower_namespace_call`
    /// and the `IrExpr::NamespaceField` arm of `lower_expr`. At
    /// `finish()` time the driver copies this onto
    /// [`crate::cg::program::CgProgram::namespace_registry`] so the
    /// WGSL emit can resolve return types + access forms without a
    /// separate registry walk.
    pub namespace_registry: super::super::program::NamespaceRegistry,
    /// Statements an in-flight expression lowering wants prepended to
    /// the surrounding statement list. The N²-fold lowering uses this:
    /// `IrExpr::Fold { Sum|Count, ... }` allocates a [`CgStmt::ForEachAgent`]
    /// for the accumulator loop and pushes its id here, then returns a
    /// [`crate::cg::expr::CgExpr::ReadLocal`] that reads the just-
    /// populated accumulator. The driver-level `lower_stmt_list` (in
    /// `physics.rs`) drains this buffer before each child stmt so the
    /// fold loop runs ahead of its consumer in source order.
    pub pending_pre_stmts: Vec<crate::cg::stmt::CgStmtId>,
    /// Source-level name of the binder bound by the innermost active
    /// fold (if any). Set by the fold lowering before lowering the
    /// projection expression and cleared after; consumed by
    /// [`lower_field`] / [`lower_bare_local`] so reads of `<binder>` /
    /// `<binder>.<field>` resolve to [`AgentRef::PerPairCandidate`] /
    /// [`crate::cg::expr::CgExpr::PerPairCandidateId`] just like the
    /// existing pair-bound surfaces. `None` outside a fold context.
    ///
    /// Single-slot rather than a stack because today's fixtures don't
    /// nest folds; a future fixture that does (`sum(other in agents :
    /// sum(third in agents : ...))`) would need this to grow into a
    /// stack and the fold lowering to save / restore around the
    /// nested call.
    pub fold_binder_name: Option<String>,
}

/// Captured form of a `@lazy` view's resolved AST: enough to
/// substitute its body at every call site without re-lowering the
/// view declaration itself. Populated by
/// [`super::view::lower_view`] on the lazy arm; consumed by
/// [`lower_view_call`] when it observes a call to a lazy view.
///
/// `param_locals` is the i-th positional parameter's `LocalRef`,
/// in declaration order. The substitution walk replaces every
/// `IrExpr::Local(LocalRef, _)` whose ref appears in this slice
/// with the matching positional argument expression.
#[derive(Debug, Clone)]
pub struct LazyViewSnapshot {
    pub param_locals: Vec<LocalRef>,
    pub body: IrExprNode,
}

impl<'a> LoweringCtx<'a> {
    /// Construct a context with empty maps and no diagnostics.
    ///
    /// `target_local` defaults to `false`: `target.<field>` accesses
    /// produce [`LoweringError::UnsupportedFieldBase`] until an
    /// op-level driver pass sets the flag (today the pair-bound mask
    /// driver in [`crate::cg::lower::mask`]; Task 5.5b/c will extend
    /// this to per-pair scoring + fold-body event binders).
    pub fn new(builder: &'a mut CgProgramBuilder) -> Self {
        Self {
            builder,
            view_ids: HashMap::new(),
            view_signatures: HashMap::new(),
            variant_ids: HashMap::new(),
            event_kind_ids: HashMap::new(),
            event_field_indices: HashMap::new(),
            local_ids: HashMap::new(),
            action_ids: HashMap::new(),
            diagnostics: Vec::new(),
            target_local: false,
            config_const_ids: HashMap::new(),
            lazy_view_bodies: HashMap::new(),
            local_tys: HashMap::new(),
            event_layouts: HashMap::new(),
            namespace_registry: super::super::program::NamespaceRegistry::default(),
            pending_pre_stmts: Vec::new(),
            fold_binder_name: None,
        }
    }

    /// Register a sum-type variant-name → typed id mapping. Returns
    /// the prior `VariantId` if one was registered for the same name
    /// (a duplicate registration is a driver-side defect — surfacing
    /// it lets tests assert exclusive allocation).
    ///
    /// Note: this populates [`Self::variant_ids`] only — event-kind
    /// names use the dedicated [`Self::register_event_kind`] helper.
    pub fn register_variant(&mut self, name: impl Into<String>, id: VariantId) -> Option<VariantId> {
        self.variant_ids.insert(name.into(), id)
    }

    /// Register an event-name → typed [`EventKindId`] mapping. Returns
    /// the prior `EventKindId` if one was registered for the same name
    /// (a duplicate registration is a driver-side defect — surfacing
    /// it lets tests assert exclusive allocation).
    ///
    /// Note: this populates [`Self::event_kind_ids`] only — sum-type
    /// variant names use the dedicated [`Self::register_variant`]
    /// helper. The two id spaces are distinct; see their field docs.
    pub fn register_event_kind(
        &mut self,
        name: impl Into<String>,
        id: EventKindId,
    ) -> Option<EventKindId> {
        self.event_kind_ids.insert(name.into(), id)
    }

    /// Register an `(EventKindId, field_name) → field_index` entry.
    /// Used by physics `Emit` lowering to resolve each
    /// `IrFieldInit { name, .. }` to a typed
    /// [`crate::cg::stmt::EventField`] with `(event, index)`.
    /// Driver populates the table from each event variant's
    /// declared field list (in declaration order); tests populate
    /// it directly. Returns the prior index if one was registered
    /// for the same `(event, field_name)` pair (driver-side
    /// duplicate).
    pub fn register_event_field(
        &mut self,
        event: EventKindId,
        field_name: impl Into<String>,
        index: u8,
    ) -> Option<u8> {
        self.event_field_indices.insert((event, field_name.into()), index)
    }

    /// Register the per-event-kind payload layout. Used by physics +
    /// view-fold handler lowering when synthesizing `CgStmt::Let` for
    /// each event-pattern binding (the binder's value comes from a
    /// typed `CgExpr::EventField` keyed on the layout's `field_offset`
    /// + `field_ty`). The driver populates this from the event registry
    /// via `populate_event_kinds`; tests populate it directly. Returns
    /// the prior layout if one was registered for the same kind
    /// (driver-side duplicate).
    pub fn register_event_layout(
        &mut self,
        event: EventKindId,
        layout: super::super::program::EventLayout,
    ) -> Option<super::super::program::EventLayout> {
        self.event_layouts.insert(event, layout)
    }

    /// Register an AST `LocalRef` → typed [`LocalId`] mapping. Returns
    /// the prior `LocalId` if one was registered for the same ref
    /// (driver-side duplicate).
    pub fn register_local(&mut self, ast_ref: LocalRef, id: LocalId) -> Option<LocalId> {
        self.local_ids.insert(ast_ref, id)
    }

    /// Allocate a fresh [`LocalId`] disjoint from every id already
    /// present in [`Self::local_ids`]. Used by physics `Let` lowering
    /// (Task 5.5b) to introduce a binding for `IrStmt::Let { local: ast_ref, .. }`
    /// when the driver has not pre-registered the mapping. The
    /// allocation strategy picks one past the maximum existing
    /// `LocalId`, so successive calls produce a strictly increasing
    /// sequence regardless of insertion order.
    pub fn allocate_local(&mut self, ast_ref: LocalRef) -> LocalId {
        // Pick max + 1 over both registries: AST-bound locals
        // (`local_ids`) AND typed-only accumulator locals
        // (`local_tys`-only entries are produced by the N²-fold
        // lowering, which allocates anonymous `LocalId`s for fold
        // accumulators without a corresponding `LocalRef`). Without
        // chaining `local_tys` keys here, a fold accumulator allocated
        // before a user `let` would share its id and the WGSL emit
        // would produce `let local_0: ... = local_0;` aliasing.
        let next = self
            .local_ids
            .values()
            .map(|id| id.0)
            .chain(self.local_tys.keys().map(|id| id.0))
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let id = LocalId(next);
        self.local_ids.insert(ast_ref, id);
        id
    }

    /// Register an action-name → typed [`ActionId`] mapping. Returns
    /// the prior `ActionId` if one was registered for the same name
    /// (driver-side duplicate). Used by scoring lowering to resolve
    /// row heads to stable typed action ids.
    pub fn register_action(&mut self, name: impl Into<String>, id: ActionId) -> Option<ActionId> {
        self.action_ids.insert(name.into(), id)
    }

    /// Register an AST view ref → CG view id mapping. Returns the prior
    /// `ViewId` if one was registered for the same ref (shouldn't
    /// happen in practice — surfacing it lets tests assert exclusive
    /// allocation).
    pub fn register_view(&mut self, ast_ref: AstViewRef, view_id: ViewId) -> Option<ViewId> {
        self.view_ids.insert(ast_ref, view_id)
    }

    /// Register the typed signature of `view_id`. Used by the recursive
    /// type-checker when it encounters a `CgExpr::Builtin { fn_id:
    /// ViewCall { view }, .. }`. Tests that don't exercise view calls
    /// can leave this empty.
    pub fn register_view_signature(
        &mut self,
        view_id: ViewId,
        args: Vec<CgTy>,
        result: CgTy,
    ) -> Option<(Vec<CgTy>, CgTy)> {
        self.view_signatures.insert(view_id, (args, result))
    }

    /// Register a `(NamespaceId, "<block>.<field>")` → typed
    /// [`ConfigConstId`] mapping. Returns the prior id if one was
    /// registered for the same key (a duplicate is a driver-side
    /// defect — surfacing it lets tests assert exclusive allocation).
    ///
    /// The driver populates this from
    /// `Compilation::configs` in source order; tests populate it
    /// directly. Used by `IrExpr::NamespaceField` lowering.
    pub fn register_config_const(
        &mut self,
        ns: NamespaceId,
        field: impl Into<String>,
        id: ConfigConstId,
    ) -> Option<ConfigConstId> {
        self.config_const_ids.insert((ns, field.into()), id)
    }

    /// Register the captured body of a `@lazy` view for at-call-site
    /// inlining. Returns the prior snapshot if one was registered for
    /// the same id (driver-side defect). Used by
    /// [`super::view::lower_view`]'s lazy arm.
    pub fn register_lazy_view_body(
        &mut self,
        view_id: ViewId,
        snapshot: LazyViewSnapshot,
    ) -> Option<LazyViewSnapshot> {
        self.lazy_view_bodies.insert(view_id, snapshot)
    }

    /// Record `local_id → ty` for later `IrExpr::Local` resolution.
    /// Called by physics-Let lowering after the bound expression's CG
    /// type is known. Returns the prior `CgTy` if one was registered
    /// for the same id (driver-side duplicate — surfacing it lets
    /// tests assert exclusive allocation).
    pub fn record_local_ty(&mut self, local_id: LocalId, ty: CgTy) -> Option<CgTy> {
        self.local_tys.insert(local_id, ty)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower a resolved DSL expression to a CG expression id.
///
/// On success returns the newly-allocated `CgExprId`; the corresponding
/// `CgExpr` is in `ctx.builder.program().exprs[id.0 as usize]`. On
/// failure returns a typed [`LoweringError`] naming the offending node
/// (via its `Span`) and the structural reason. The arena is not rolled
/// back on failure — see [`add`]'s "Orphan behavior" note for the full
/// story; in short, partial children (and possibly the just-pushed
/// parent itself) remain as orphans that downstream emit walks ignore.
///
/// Type-checking runs after every node is constructed: a successful
/// return means the produced `CgExpr` matches its operand types under
/// the operator's signature.
pub fn lower_expr(ast: &IrExprNode, ctx: &mut LoweringCtx<'_>) -> Result<CgExprId, LoweringError> {
    let span = ast.span;
    match &ast.kind {
        // ---- Literals ----
        IrExpr::LitBool(b) => add(ctx, CgExpr::Lit(LitValue::Bool(*b)), span),
        IrExpr::LitInt(v) => {
            // The DSL surface uses signed `i64` literals; the CG IR's
            // numeric literals are 32-bit. We pick `I32` for negative
            // values and `U32` for non-negative ones, narrowing the
            // i64. Out-of-range narrowings surface as typed
            // ill-typed errors so silent truncation can't sneak past.
            if *v < 0 {
                if *v < i32::MIN as i64 {
                    return Err(LoweringError::LiteralOutOfRange {
                        value: *v,
                        target: CgTy::I32,
                        span,
                    });
                }
                add(ctx, CgExpr::Lit(LitValue::I32(*v as i32)), span)
            } else {
                if *v > u32::MAX as i64 {
                    return Err(LoweringError::LiteralOutOfRange {
                        value: *v,
                        target: CgTy::U32,
                        span,
                    });
                }
                add(ctx, CgExpr::Lit(LitValue::U32(*v as u32)), span)
            }
        }
        IrExpr::LitFloat(v) => add(ctx, CgExpr::Lit(LitValue::F32(*v as f32)), span),

        // ---- Field access ----
        IrExpr::Field {
            base, field_name, ..
        } => lower_field(base, field_name, span, ctx),

        // ---- Local references ----
        //
        // Resolution order in `lower_bare_local`: let-bound locals
        // first (via `ctx.local_ids` → `ctx.local_tys`), then bare
        // `self` and pair-bound `target`, then a typed deferral.
        IrExpr::Local(local_ref, name) => lower_bare_local(*local_ref, name, span, ctx),

        // ---- Operators ----
        IrExpr::Binary(op, lhs, rhs) => lower_binary(*op, lhs, rhs, span, ctx),
        IrExpr::Unary(op, arg) => lower_unary(*op, arg, span, ctx),

        // ---- Conditional expression ----
        IrExpr::If {
            cond,
            then_expr,
            else_expr,
        } => match else_expr {
            Some(else_box) => lower_select(cond, then_expr, else_box, span, ctx),
            // `if … then …` without `else` has no value type; only the
            // statement form supports a None else-branch.
            None => Err(LoweringError::UnsupportedAstNode {
                ast_label: "If(without-else)",
                span,
            }),
        },

        // ---- Calls ----
        IrExpr::BuiltinCall(b, args) => lower_builtin_call(*b, args, span, ctx),
        IrExpr::ViewCall(view_ref, args) => lower_view_call(*view_ref, args, span, ctx),
        IrExpr::NamespaceCall { ns, method, args } => {
            lower_namespace_call(*ns, method.as_str(), args, span, ctx)
        }
        IrExpr::NamespaceField { ns, field, .. } => {
            // `config.<block>.<field>` (the only NamespaceField shape the
            // resolver produces today via the dedicated `Config`
            // namespace) lowers to a typed `Read(ConfigConst { id })`.
            // Other namespaces consult the schema-driven
            // `namespace_registry` (Task 4 of the CG lowering gap
            // closure plan) to lower `world.tick` and friends as
            // typed `CgExpr::NamespaceField` nodes.
            if *ns == NamespaceId::Config {
                match ctx.config_const_ids.get(&(*ns, field.clone())) {
                    Some(id) => add(ctx, CgExpr::Read(DataHandle::ConfigConst { id: *id }), span),
                    None => Err(LoweringError::UnknownConfigField {
                        ns: *ns,
                        field: field.clone(),
                        span,
                    }),
                }
            } else if let Some(def) = ctx
                .namespace_registry
                .namespaces
                .get(ns)
                .and_then(|nd| nd.fields.get(field))
            {
                let ty = def.ty;
                add(
                    ctx,
                    CgExpr::NamespaceField {
                        ns: *ns,
                        field: field.clone(),
                        ty,
                    },
                    span,
                )
            } else {
                Err(LoweringError::UnsupportedNamespaceField {
                    ns: *ns,
                    field: field.clone(),
                    span,
                })
            }
        }
        IrExpr::Namespace(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Namespace",
            span,
        }),
        IrExpr::Event(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Event",
            span,
        }),
        IrExpr::Entity(r) => {
            // Entity-name-as-value: lower to its declaration-order
            // discriminant. Used in `where (self.creature_type ==
            // <EntityName>)` per-handler filters; the per-creature
            // SoA AgentField::CreatureType (`AgentFieldTy::OptEnumU32`,
            // CgTy::U32 at the expression layer) is compared against
            // this constant so the where-guard's body only fires for
            // matching agents.
            //
            // Discriminant convention: EntityRef.0 (declaration order
            // index). The runtime is responsible for setting
            // `agent.creature_type = entity_ref_index` when spawning
            // an agent of that entity declaration. Since the
            // EntityRef index is stable across compiles for a given
            // .sim, the runtime can hard-code or look up the mapping
            // off `comp.entities` order.
            add(
                ctx,
                CgExpr::Lit(LitValue::U32(r.0 as u32)),
                span,
            )
        }
        IrExpr::View(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "View",
            span,
        }),
        IrExpr::Verb(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Verb",
            span,
        }),
        IrExpr::VerbCall(_, _) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "VerbCall",
            span,
        }),
        IrExpr::UnresolvedCall(_, _) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "UnresolvedCall",
            span,
        }),
        IrExpr::EnumVariant { .. } => Err(LoweringError::UnsupportedAstNode {
            ast_label: "EnumVariant",
            span,
        }),
        IrExpr::LitString(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "LitString",
            span,
        }),
        IrExpr::Index(_, _) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Index",
            span,
        }),
        IrExpr::In(_, _) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "In",
            span,
        }),
        IrExpr::Contains(_, _) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Contains",
            span,
        }),
        IrExpr::Quantifier { .. } => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Quantifier",
            span,
        }),
        // `count(binder in iter where pred)` / `sum(...)` / `max(...)` /
        // `min(...)`. The resolver shapes every aggregation comprehension
        // as `IrExpr::Fold { kind, binder, iter, body }`. Lowering today
        // recognises **only** `FoldKind::Count` over the `agents`
        // namespace iterator — the shape Boids' `neighbor_count` lazy
        // view uses (`assets/sim/boids.sim`).
        //
        // ## What this arm produces
        //
        // A typed-zero short-circuit at the expression position. The
        // Fold is **not** materialised as compute — neither as a CG IR
        // variant carrying the loop, nor as a real WGSL `for` walk.
        // Real fold emit requires either (a) a top-level WGSL helper-fn
        // prelude (out of scope: would need to edit
        // `cg/emit/program.rs::compose_wgsl_file`) or (b) statement
        // injection from the lowering layer (out of scope: would need a
        // `CgStmt::Fold` shape and changes to `physics.rs` /
        // `view.rs` / scoring lowering to splice the synthesised stmt
        // before its consumer). Both paths reach beyond the file scope
        // pinned by the Fold-lowering subagent task.
        //
        // ## Why a literal short-circuit is honest here
        //
        // The B1 conventions across this emit stack (the
        // `b1_default_for_field_ty` fallback in
        // `wgsl_body.rs::lower_cg_expr_to_wgsl`, the wildcard PerUnit
        // collapse a few arms below, the `MOVEMENT_BODY` /
        // `SPATIAL_BUILD_HASH_BODY` placeholders in
        // `cg/emit/kernel.rs`) all share the same posture: structural
        // scaffolding lands first, real semantics layers on once the
        // surrounding infrastructure exists. Boids' `neighbor_count`
        // is consumed only by the `MoveBoid` per-agent physics body,
        // which is itself a `MOVEMENT_BODY` placeholder today — so a
        // real fold value would have nothing to feed.
        //
        // ## What this unblocks
        //
        // The Boids fixture lowers cleanly through the lazy-view inline
        // path (`lower_view_call`'s `lazy_view_bodies` branch above),
        // which substitutes `neighbor_count`'s body at every call site
        // and walks it through `lower_expr` recursively. Without this
        // arm, every such call surfaces as `UnsupportedAstNode {
        // ast_label: "Fold" }` and the entire enclosing op fails. With
        // it, the call lowers to `0i` and the rest of the body
        // continues. The real fold semantic lands when:
        //   1. A kernel actually consumes a `count(...)` result (today
        //      no kernel does — the only consumer is `MOVEMENT_BODY`'s
        //      hand-written placeholder).
        //   2. The compose_wgsl_file pipeline grows a fold-helper-fn
        //      prelude OR `CgStmt::Fold` lands with statement-injection
        //      lowering wired through physics + view + scoring bodies.
        //
        // ## Sum / Min / Max
        //
        // Out of scope for the Boids unblock — Boids only uses Count.
        // Future fixtures wanting `sum_vec3(...)` etc. surface as their
        // own `UnsupportedAstNode` deferrals here; extend the match
        // when a real consumer arrives.
        // N²-fold over `agents` — Sum-projection or Count-predicate
        // shape, lowered as a `CgStmt::ForEachAgent` injected via
        // `pending_pre_stmts` plus a `CgExpr::ReadLocal` reading the
        // populated accumulator. See `lower_fold_over_agents` for the
        // shape contract. Min/Max remain UnsupportedAstNode until a
        // fixture asks for them — they require a different
        // accumulator init (NEG_INFINITY / INFINITY) and per-iteration
        // op (max / min instead of `+`), so the same scaffolding
        // generalises easily but isn't useful today.
        IrExpr::Fold { kind, binder_name, iter, body, .. } => {
            use dsl_ast::ast::FoldKind;
            match kind {
                FoldKind::Count | FoldKind::Sum => {
                    lower_fold_over_agents(
                        *kind,
                        binder_name.as_deref(),
                        iter.as_deref(),
                        body,
                        span,
                        ctx,
                    )
                }
                FoldKind::Min | FoldKind::Max => {
                    Err(LoweringError::UnsupportedAstNode {
                        ast_label: "Fold",
                        span,
                    })
                }
            }
        }
        IrExpr::List(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "List",
            span,
        }),
        IrExpr::Tuple(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Tuple",
            span,
        }),
        IrExpr::StructLit { .. } => Err(LoweringError::UnsupportedAstNode {
            ast_label: "StructLit",
            span,
        }),
        IrExpr::Ctor { .. } => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Ctor",
            span,
        }),
        IrExpr::Match { .. } => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Match",
            span,
        }),
        // PerUnit semantic simplification (Phase 6 Task 1, refined Task 2.5):
        // `<expr> per_unit <delta>` lowers as `expr * delta` per the AST
        // docstring. The "iterate over each unit in the result" semantic
        // that pertains inside scoring contexts is deferred — for view
        // storage that's empty (smoke fixture, idle agents) the result
        // is identical (0 * delta = 0). Closing the gap for richer
        // fixtures requires per-unit-fold-over-view-storage IR primitive;
        // tracked as future work.
        //
        // Wildcard short-circuit: when `expr` contains a wildcard `_`
        // (e.g. `view::threat_level(self, _) per_unit 0.01`), the inner
        // view call has typed-mismatch issues because `_` substitutes as
        // a u32 placeholder while the view signature expects an AgentId.
        // The wildcard semantically means "iterate over all candidates
        // and sum" — under the B1 simplification the per-unit fold is
        // unperformed, so the contribution is 0 regardless. Short-circuit
        // the entire PerUnit to a literal `0.0_f32` to avoid the
        // type-mismatch path. Same semantic as the non-wildcard
        // simplification (modifier contributes 0 for empty storage).
        IrExpr::PerUnit { expr, delta } => {
            if expr_contains_wildcard(expr) {
                add(ctx, CgExpr::Lit(LitValue::F32(0.0)), span)
            } else {
                lower_binary(BinOp::Mul, expr, delta, span, ctx)
            }
        }
        IrExpr::AbilityTag { .. } => Err(LoweringError::UnsupportedAstNode {
            ast_label: "AbilityTag",
            span,
        }),
        IrExpr::AbilityHint => Err(LoweringError::UnsupportedAstNode {
            ast_label: "AbilityHint",
            span,
        }),
        IrExpr::AbilityHintLit(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "AbilityHintLit",
            span,
        }),
        IrExpr::AbilityRange => Err(LoweringError::UnsupportedAstNode {
            ast_label: "AbilityRange",
            span,
        }),
        IrExpr::AbilityOnCooldown(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "AbilityOnCooldown",
            span,
        }),
        IrExpr::Raw(_) => Err(LoweringError::UnsupportedAstNode {
            ast_label: "Raw",
            span,
        }),
        IrExpr::BeliefsAccessor { .. } => Err(LoweringError::UnsupportedAstNode {
            ast_label: "BeliefsAccessor",
            span,
        }),
        IrExpr::BeliefsConfidence { .. } => Err(LoweringError::UnsupportedAstNode {
            ast_label: "BeliefsConfidence",
            span,
        }),
        IrExpr::BeliefsView { .. } => Err(LoweringError::UnsupportedAstNode {
            ast_label: "BeliefsView",
            span,
        }),
    }
}

// ---------------------------------------------------------------------------
// Per-shape helpers
// ---------------------------------------------------------------------------

/// Push `expr` into the builder, type-check it, and return its id.
///
/// Wraps the builder error and the type-check error in
/// [`LoweringError`] so every push goes through one funnel.
///
/// **Orphan behavior:** if `type_check` fails, the just-pushed parent
/// expression remains in the arena as an orphan — the caller gets
/// `Err`, but the arena is not rolled back. Children pushed before the
/// failing type-check also remain. Orphans are harmless: downstream
/// emit walks only ids reachable from `ComputeOpKind`, so orphan exprs
/// are dead-stripped at emit time. The well-formed pass treats them as
/// non-errors (an orphan expr in the arena that no op references is
/// not a P10 / structural concern).
fn add(
    ctx: &mut LoweringCtx<'_>,
    expr: CgExpr,
    span: Span,
) -> Result<CgExprId, LoweringError> {
    let id = ctx
        .builder
        .add_expr(expr)
        .map_err(|e| LoweringError::BuilderRejected { error: e, span })?;
    typecheck_node(ctx, id, span)?;
    Ok(id)
}

/// Run [`type_check`] on the node at `id`, surfacing any failure as a
/// typed [`LoweringError::TypeCheckFailure`]. Defers view-signature
/// lookup to the in-context map (`ctx.view_signatures`).
pub(super) fn typecheck_node(
    ctx: &LoweringCtx<'_>,
    id: CgExprId,
    span: Span,
) -> Result<CgTy, LoweringError> {
    let prog = ctx.builder.program();
    let node = prog
        .exprs
        .get(id.0 as usize)
        .ok_or(LoweringError::TypeCheckFailure {
            error: TypeError::DanglingExprId {
                node: id,
                referenced: id,
            },
            span,
        })?;
    let resolver: &dyn Fn(ViewId) -> Option<(Vec<CgTy>, CgTy)> = &|view_id| {
        ctx.view_signatures
            .get(&view_id)
            .map(|(args, result)| (args.clone(), *result))
    };
    let tc_ctx = TypeCheckCtx::with_view_signature(prog, resolver);
    type_check(node, id, &tc_ctx).map_err(|e| LoweringError::TypeCheckFailure { error: e, span })
}

/// Lower `<base>.<field_name>`. Today the wired bases are `self` (any
/// dispatch shape) and `target` (only inside a pair-bound op — gated by
/// [`LoweringCtx::target_local`]).
///
/// # Limitations
///
/// - `target.<field>` only resolves when [`LoweringCtx::target_local`]
///   is `true`. The driver pass for pair-bound masks (today
///   [`crate::cg::lower::mask::lower_mask`] when the dispatch shape is
///   [`crate::cg::dispatch::DispatchShape::PerPair`]) sets the flag
///   before lowering the predicate and restores it after; outside
///   pair-bound contexts the same access surfaces as the typed
///   [`LoweringError::UnsupportedFieldBase`] deferral so a stray
///   `target` reference can't accidentally route through the per-pair
///   candidate buffer.
/// - Other bases (locals other than `self` / `target`, namespace
///   fields, builder-receiver chains) surface as the same
///   [`LoweringError::UnsupportedFieldBase`] typed deferral.
fn lower_field(
    base: &IrExprNode,
    field_name: &str,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    // Resolve the agent reference implied by the base expression. Today
    // wired bases are `self` (every dispatch shape) and `target` inside
    // a pair-bound op. Anything else falls through as the typed
    // `UnsupportedFieldBase` deferral so a stray base reference can't
    // accidentally route through the per-pair candidate buffer.
    let target = match &base.kind {
        IrExpr::Local(_, local_name) if local_name == "self" => AgentRef::Self_,
        IrExpr::Local(_, local_name)
            if (local_name == "target" || local_name == "candidate") && ctx.target_local =>
        {
            // Pair-bound mask predicates (and, in 5.5b/c, scoring rows /
            // fold bodies) bind `target` to the per-pair candidate. The
            // emit layer (Task 4.x) resolves `AgentRef::PerPairCandidate`
            // to the candidate buffer + per-thread offset implied by the
            // dispatch shape's `PerPair { source }`; the IR layer just
            // tags the read.
            //
            // Phase 7 Task 5: spatial_query bodies bind their per-pair
            // neighbour as `candidate` (the v1 convention for the new
            // `spatial_query <name>(self, candidate, ...) = <filter>`
            // surface). When such a body is lowered via
            // `lower_filter_for_mask` (which sets `target_local = true`),
            // a `candidate.<field>` access must also resolve to
            // `PerPairCandidate`. Both names route here so existing
            // wolf-sim source (`target.<field>`) and new spatial_query
            // source (`candidate.<field>`) coexist without renaming
            // user-visible identifiers.
            AgentRef::PerPairCandidate
        }
        IrExpr::Local(_, local_name)
            if ctx.fold_binder_name.as_deref() == Some(local_name.as_str()) =>
        {
            // N²-fold body: the user-named binder (e.g. `other` in
            // `sum(other in agents where ... : other.pos)`) resolves
            // to the per-iteration loop variable, which the
            // `CgStmt::ForEachAgent` WGSL emit declares as
            // `per_pair_candidate`. Sharing AgentRef::PerPairCandidate
            // means `binder.<field>` reads route through the same
            // `agent_<field>[per_pair_candidate]` access shape the
            // pair-bound contexts already use.
            AgentRef::PerPairCandidate
        }
        _ => {
            return Err(LoweringError::UnsupportedFieldBase {
                field_name: field_name.to_string(),
                span,
            });
        }
    };

    // Virtual fields: names that don't map to an `AgentFieldId` but
    // synthesize a CG expression from real primitives. Today: `hp_pct`
    // ⇒ `hp / max_hp`. Future entries (mana_pct, cooldown_progress,
    // …) extend the dispatch below.
    if let Some(synth) = lookup_virtual_field(field_name) {
        return synth(target, span, ctx);
    }

    let field = AgentFieldId::from_snake(field_name).ok_or_else(|| {
        LoweringError::UnknownAgentField {
            field_name: field_name.to_string(),
            span,
        }
    })?;
    add(
        ctx,
        CgExpr::Read(DataHandle::AgentField { field, target }),
        span,
    )
}

/// Type alias for a virtual-field synthesizer. Each entry takes the
/// resolved [`AgentRef`] (whatever `self`-or-`target` the original
/// `IrExpr::Field` carried), the source [`Span`], and the lowering
/// context, and produces the synthesized [`CgExprId`].
type VirtualFieldSynth =
    fn(AgentRef, Span, &mut LoweringCtx<'_>) -> Result<CgExprId, LoweringError>;

/// Virtual fields synthesized from real `AgentField` primitives. Today
/// the only entry is `hp_pct = hp / max_hp`; new virtuals
/// (`mana_pct`, `cooldown_progress`, …) extend this table without
/// touching `lower_field`'s control flow.
const VIRTUAL_FIELDS: &[(&str, VirtualFieldSynth)] = &[("hp_pct", lower_hp_pct)];

/// Lookup helper for [`VIRTUAL_FIELDS`]. Returns the synthesizer for a
/// virtual field name, or `None` for real `AgentFieldId` names (which
/// fall through to `AgentFieldId::from_snake` in [`lower_field`]).
fn lookup_virtual_field(field_name: &str) -> Option<VirtualFieldSynth> {
    VIRTUAL_FIELDS
        .iter()
        .find_map(|(name, synth)| (*name == field_name).then_some(*synth))
}

/// Synthesize `<target>.hp / <target>.max_hp` for `<target>.hp_pct`.
/// Both reads carry the caller-supplied `target` so a per-pair
/// `target.hp_pct` lowers to per-pair-candidate reads, and `self.hp_pct`
/// lowers to self-reads.
fn lower_hp_pct(
    target: AgentRef,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let lhs = add(
        ctx,
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: target.clone(),
        }),
        span,
    )?;
    let rhs = add(
        ctx,
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::MaxHp,
            target,
        }),
        span,
    )?;
    add(
        ctx,
        CgExpr::Binary {
            op: BinaryOp::DivF32,
            lhs,
            rhs,
            ty: CgTy::F32,
        },
        span,
    )
}

/// Lower a bare `IrExpr::Local(local_ref, name)` (no `.field` access)
/// to a CG expression.
///
/// Resolution order:
///
/// 1. **Let-bound local.** If `ctx.local_ids` has an entry for
///    `local_ref`, the local was introduced by an enclosing
///    `IrStmt::Let` (lowered to `CgStmt::Let { local, value, ty }`).
///    The expression resolves to `CgExpr::ReadLocal { local, ty }`
///    where `ty` comes from `ctx.local_tys`. A missing
///    `local_tys` entry surfaces as
///    [`LoweringError::UnknownLocalType`].
/// 2. **Bare `self`.** Resolves to [`CgExpr::AgentSelfId`] (typed
///    `AgentId`). Used in surface DSL like `agents.alive(self)` and
///    `target != self`.
/// 3. **Bare `target` in a pair-bound context.** Resolves to
///    [`CgExpr::PerPairCandidateId`] (typed `AgentId`) when
///    `ctx.target_local` is `true`. Outside pair-bound contexts, falls
///    through to the default error.
/// 4. **Anything else** — surfaces as
///    [`LoweringError::UnsupportedLocalBinding`].
fn lower_bare_local(
    local_ref: LocalRef,
    name: &str,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    // Step 1: let-bound local.
    if let Some(&local_id) = ctx.local_ids.get(&local_ref) {
        let ty = ctx.local_tys.get(&local_id).copied().ok_or(
            LoweringError::UnknownLocalType {
                local: local_id,
                span,
            },
        )?;
        return add(ctx, CgExpr::ReadLocal { local: local_id, ty }, span);
    }

    // Step 2-3: structural locals.
    match name {
        "self" => add(ctx, CgExpr::AgentSelfId, span),
        // Phase 7 Task 5: accept both `target` (the action-head binder
        // name used by wolf-sim masks like `mask MoveToward(target) ...`)
        // and `candidate` (the spatial_query body binder name from the
        // new `spatial_query <name>(self, candidate, ...) = <filter>`
        // surface). Both resolve to the per-pair candidate id when the
        // pair-bound context is active. See the matching arm in
        // `lower_field` for the field-access path.
        "target" | "candidate" if ctx.target_local => {
            add(ctx, CgExpr::PerPairCandidateId, span)
        }
        // N²-fold body bare-binder read (`other != self` etc.). The
        // user-named binder resolves to the per-iteration loop
        // variable, mirroring the field-access path in `lower_field`.
        n if ctx.fold_binder_name.as_deref() == Some(n) => {
            add(ctx, CgExpr::PerPairCandidateId, span)
        }
        // Wildcard `_` is short-circuited at the PerUnit lowering level
        // (see `IrExpr::PerUnit` arm in `lower_expr`). It should never
        // reach `lower_bare_local` directly — if it does, that's a
        // genuinely-unsupported context (e.g., bare wildcard outside a
        // PerUnit-modified view call) which surfaces as the standard
        // UnsupportedLocalBinding diagnostic. Future per-unit-fold
        // semantic resolves wildcards in the fold-iteration variable
        // binding instead.
        _ => Err(LoweringError::UnsupportedLocalBinding {
            name: name.to_string(),
            span,
        }),
    }
}

/// True if `node` is a bare wildcard `_` (an `IrExpr::Local` with
/// name `"_"`). Used by `lower_view_call` to short-circuit view calls
/// with wildcard args to a typed zero literal. Sibling helper to
/// [`expr_contains_wildcard`].
fn is_wildcard_local(node: &IrExprNode) -> bool {
    use dsl_ast::ir::IrExpr;
    matches!(&node.kind, IrExpr::Local(_, name) if name == "_")
}

/// True if `node` (or any sub-expression) is an `IrExpr::Local` with
/// name `"_"`. Used by the PerUnit lowering arm to short-circuit
/// wildcard-bearing expressions to a literal 0.0 rather than
/// attempting the (currently-broken) view-call-with-wildcard-arg path.
/// See the `IrExpr::PerUnit` arm in `lower_expr` for the rationale.
///
/// Cheap recursive walk: the wildcard appears at most a few sites per
/// PerUnit expression in practice, and the match is exhaustive over
/// `IrExpr` so adding a new variant forces an explicit decision.
fn expr_contains_wildcard(node: &IrExprNode) -> bool {
    use dsl_ast::ir::IrExpr;
    match &node.kind {
        IrExpr::Local(_, name) => name == "_",
        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::LitString(_)
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::View(_)
        | IrExpr::Verb(_)
        | IrExpr::Namespace(_)
        | IrExpr::NamespaceField { .. }
        | IrExpr::EnumVariant { .. }
        | IrExpr::AbilityTag { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit { .. } => false,
        IrExpr::Unary(_, e) => expr_contains_wildcard(e),
        IrExpr::Binary(_, lhs, rhs)
        | IrExpr::In(lhs, rhs)
        | IrExpr::Contains(lhs, rhs)
        | IrExpr::Index(lhs, rhs) => {
            expr_contains_wildcard(lhs) || expr_contains_wildcard(rhs)
        }
        IrExpr::PerUnit { expr, delta } => {
            expr_contains_wildcard(expr) || expr_contains_wildcard(delta)
        }
        IrExpr::Field { base, .. } => expr_contains_wildcard(base),
        IrExpr::ViewCall(_, args)
        | IrExpr::VerbCall(_, args)
        | IrExpr::BuiltinCall(_, args)
        | IrExpr::UnresolvedCall(_, args)
        | IrExpr::NamespaceCall { args, .. } => {
            args.iter().any(|a| expr_contains_wildcard(&a.value))
        }
        IrExpr::If { cond, then_expr, else_expr } => {
            expr_contains_wildcard(cond)
                || expr_contains_wildcard(then_expr)
                || else_expr.as_ref().map_or(false, |e| expr_contains_wildcard(e))
        }
        IrExpr::Quantifier { iter, body, .. } => {
            expr_contains_wildcard(iter) || expr_contains_wildcard(body)
        }
        IrExpr::Fold { iter, body, .. } => {
            iter.as_ref().map_or(false, |i| expr_contains_wildcard(i))
                || expr_contains_wildcard(body)
        }
        IrExpr::Match { scrutinee, arms } => {
            expr_contains_wildcard(scrutinee)
                || arms.iter().any(|arm| expr_contains_wildcard(&arm.body))
        }
        IrExpr::List(items) | IrExpr::Tuple(items) => {
            items.iter().any(expr_contains_wildcard)
        }
        IrExpr::StructLit { fields, .. } => {
            fields.iter().any(|f| expr_contains_wildcard(&f.value))
        }
        IrExpr::Ctor { args, .. } => args.iter().any(expr_contains_wildcard),
        // Catch-all: any IrExpr variant not enumerated above is a leaf
        // shape with no IrExprNode sub-expressions (AbilityRange, Raw,
        // AbilityOnCooldown, etc.). They cannot contain a wildcard
        // syntactically. If a future variant adds children, the match
        // arm needs an explicit case — exhaustive checking will force
        // the update.
        _ => false,
    }
}

/// Lower a binary operator. Picks the typed [`BinaryOp`] variant based
/// on operand types — the AST's untyped `BinOp::Lt` becomes
/// `BinaryOp::LtF32` when both operands are `F32`, etc.
fn lower_binary(
    op: BinOp,
    lhs: &IrExprNode,
    rhs: &IrExprNode,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let lhs_id = lower_expr(lhs, ctx)?;
    let rhs_id = lower_expr(rhs, ctx)?;
    let lhs_ty = typecheck_node(ctx, lhs_id, lhs.span)?;
    let rhs_ty = typecheck_node(ctx, rhs_id, rhs.span)?;

    // Signed/unsigned integer literal coercion. A non-negative DSL
    // integer literal defaults to `LitValue::U32` at lowering time
    // (see `IrExpr::LitInt`). When the other operand is an `I32`-
    // typed non-literal (e.g., a signed event-field read like
    // `delta: i32`), the resulting `i32 != 0u32` shape would be
    // rejected as `BinaryOperandTyMismatch`. We coerce one side: if
    // exactly ONE operand is a `U32` literal and the OTHER operand
    // is `I32`-typed and non-literal, re-emit the literal as `I32`.
    // The asymmetric requirement (literal vs. non-literal) keeps a
    // genuine `i32_a != u32_b` mismatch (both non-literal) reported
    // as a real typing bug rather than silently coerced away.
    let (lhs_id, lhs_ty, rhs_id, rhs_ty) =
        coerce_int_literal_to_signed(ctx, lhs_id, lhs_ty, rhs_id, rhs_ty, span)?;

    // Asymmetric Vec3-by-scalar arithmetic: `vec3 * f32` /
    // `f32 * vec3` / `vec3 / f32`. WGSL handles these natively; we
    // pick the typed Mul/DivVec3ByF32 variant with the vec3 always
    // on the lhs (commute scalar*vec to vec*scalar at lowering
    // time). Falls through to the symmetric path below if neither
    // operand pair matches.
    if let Some(id) = try_lower_vec3_scalar(op, lhs_id, lhs_ty, rhs_id, rhs_ty, span, ctx)? {
        return Ok(id);
    }

    if lhs_ty != rhs_ty {
        return Err(LoweringError::BinaryOperandTyMismatch {
            op,
            lhs_ty,
            rhs_ty,
            span,
        });
    }

    let cg_op = pick_binary_op(op, lhs_ty, span)?;
    let result_ty = cg_op.result_ty();
    add(
        ctx,
        CgExpr::Binary {
            op: cg_op,
            lhs: lhs_id,
            rhs: rhs_id,
            ty: result_ty,
        },
        span,
    )
}

/// Try to lower a `vec3 * f32` / `f32 * vec3` / `vec3 / f32` binary
/// expression to its typed asymmetric variant
/// (`MulVec3ByF32` / `DivVec3ByF32`). Returns `Ok(Some(id))` when
/// the pattern matched and the typed binary node is now in the
/// arena; `Ok(None)` otherwise (the caller falls through to the
/// symmetric path). The `f32 * vec3` form commutes the operands so
/// vec3 is always on the lhs of the emitted `MulVec3ByF32`.
fn try_lower_vec3_scalar(
    op: BinOp,
    lhs_id: CgExprId,
    lhs_ty: CgTy,
    rhs_id: CgExprId,
    rhs_ty: CgTy,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<Option<CgExprId>, LoweringError> {
    let cg_op = match (op, lhs_ty, rhs_ty) {
        (BinOp::Mul, CgTy::Vec3F32, CgTy::F32) | (BinOp::Mul, CgTy::F32, CgTy::Vec3F32) => {
            BinaryOp::MulVec3ByF32
        }
        (BinOp::Div, CgTy::Vec3F32, CgTy::F32) => BinaryOp::DivVec3ByF32,
        // Div the other way (`f32 / vec3`) is component-divide-into-
        // scalar, semantically distinct and not needed by the boids
        // fixture; skip until a real consumer arrives.
        _ => return Ok(None),
    };
    let (vec_id, scalar_id) = if lhs_ty == CgTy::Vec3F32 {
        (lhs_id, rhs_id)
    } else {
        (rhs_id, lhs_id)
    };
    let id = add(
        ctx,
        CgExpr::Binary {
            op: cg_op,
            lhs: vec_id,
            rhs: scalar_id,
            ty: CgTy::Vec3F32,
        },
        span,
    )?;
    Ok(Some(id))
}

/// Coerce a default-`U32` integer literal operand to `I32` when the
/// peer operand is a non-literal `I32`. Returns the (possibly updated)
/// `(lhs_id, lhs_ty, rhs_id, rhs_ty)` tuple.
///
/// Rationale: the DSL surface uses signed `i64` literal values; the
/// CG layer narrows to `U32` for non-negative literals (see
/// `IrExpr::LitInt`'s lowering). Patterns like `delta != 0` —
/// where `delta: i32` reads as a signed event-field — need the `0`
/// to lower as `I32` for the binary type-check to succeed. The
/// coercion is intentionally narrow:
///
/// - Only `U32` → `I32`, not the symmetric direction (no operand we
///   produce today defaults a literal to `I32` at lowering time).
/// - Only when EXACTLY ONE operand is a `Lit` and the OTHER is
///   non-`Lit`. Two non-literal `i32`/`u32` operands is a genuine
///   typing bug and stays an error.
/// - Only the literal value `0..=i32::MAX` survives the cast
///   without truncation. Higher-magnitude literals would need
///   explicit user-side typing; we leave them as `U32` so the
///   downstream mismatch error stays visible.
fn coerce_int_literal_to_signed(
    ctx: &mut LoweringCtx<'_>,
    lhs_id: CgExprId,
    lhs_ty: CgTy,
    rhs_id: CgExprId,
    rhs_ty: CgTy,
    span: Span,
) -> Result<(CgExprId, CgTy, CgExprId, CgTy), LoweringError> {
    // Only act on (I32, U32) or (U32, I32) operand-type pairs.
    let (lhs_lit, rhs_lit) = {
        let prog = ctx.builder.program();
        let lhs_lit = prog
            .exprs
            .get(lhs_id.0 as usize)
            .and_then(|e| match e {
                CgExpr::Lit(LitValue::U32(v)) => Some(*v),
                _ => None,
            });
        let rhs_lit = prog
            .exprs
            .get(rhs_id.0 as usize)
            .and_then(|e| match e {
                CgExpr::Lit(LitValue::U32(v)) => Some(*v),
                _ => None,
            });
        (lhs_lit, rhs_lit)
    };

    // Case A: lhs is non-literal I32, rhs is U32 literal — coerce rhs.
    if lhs_ty == CgTy::I32 && rhs_ty == CgTy::U32 && rhs_lit.is_some() && lhs_lit.is_none() {
        let v = rhs_lit.unwrap();
        if v <= i32::MAX as u32 {
            let new_rhs = add(ctx, CgExpr::Lit(LitValue::I32(v as i32)), span)?;
            return Ok((lhs_id, lhs_ty, new_rhs, CgTy::I32));
        }
    }
    // Case B: rhs is non-literal I32, lhs is U32 literal — coerce lhs.
    if rhs_ty == CgTy::I32 && lhs_ty == CgTy::U32 && lhs_lit.is_some() && rhs_lit.is_none() {
        let v = lhs_lit.unwrap();
        if v <= i32::MAX as u32 {
            let new_lhs = add(ctx, CgExpr::Lit(LitValue::I32(v as i32)), span)?;
            return Ok((new_lhs, CgTy::I32, rhs_id, rhs_ty));
        }
    }
    Ok((lhs_id, lhs_ty, rhs_id, rhs_ty))
}

/// Pick the typed [`BinaryOp`] variant for a given AST op + operand
/// type. An unsupported combination (e.g., `agent.alive < 5`'s
/// `Lt<Bool>`) becomes [`LoweringError::IllTypedExpression`].
fn pick_binary_op(op: BinOp, ty: CgTy, span: Span) -> Result<BinaryOp, LoweringError> {
    match (op, ty) {
        // Logical — Bool only.
        (BinOp::And, CgTy::Bool) => Ok(BinaryOp::And),
        (BinOp::Or, CgTy::Bool) => Ok(BinaryOp::Or),
        (BinOp::And | BinOp::Or, _) => Err(LoweringError::IllTypedExpression {
            expected: CgTy::Bool,
            got: ty,
            span,
        }),

        // Equality — Bool, U32, I32, F32, AgentId. Tick comparisons go
        // through U32 (BinaryOp doc states this).
        (BinOp::Eq, CgTy::Bool) => Ok(BinaryOp::EqBool),
        (BinOp::Eq, CgTy::U32) | (BinOp::Eq, CgTy::Tick) => Ok(BinaryOp::EqU32),
        (BinOp::Eq, CgTy::I32) => Ok(BinaryOp::EqI32),
        (BinOp::Eq, CgTy::F32) => Ok(BinaryOp::EqF32),
        (BinOp::Eq, CgTy::AgentId) => Ok(BinaryOp::EqAgentId),
        (BinOp::Eq, CgTy::Vec3F32) | (BinOp::Eq, CgTy::ViewKey { .. }) => {
            Err(LoweringError::IllTypedExpression {
                expected: CgTy::F32,
                got: ty,
                span,
            })
        }
        (BinOp::NotEq, CgTy::Bool) => Ok(BinaryOp::NeBool),
        (BinOp::NotEq, CgTy::U32) | (BinOp::NotEq, CgTy::Tick) => Ok(BinaryOp::NeU32),
        (BinOp::NotEq, CgTy::I32) => Ok(BinaryOp::NeI32),
        (BinOp::NotEq, CgTy::F32) => Ok(BinaryOp::NeF32),
        (BinOp::NotEq, CgTy::AgentId) => Ok(BinaryOp::NeAgentId),
        (BinOp::NotEq, CgTy::Vec3F32) | (BinOp::NotEq, CgTy::ViewKey { .. }) => {
            Err(LoweringError::IllTypedExpression {
                expected: CgTy::F32,
                got: ty,
                span,
            })
        }

        // Ordered comparisons — F32, U32 (incl. Tick), I32 only.
        (BinOp::Lt, CgTy::F32) => Ok(BinaryOp::LtF32),
        (BinOp::Lt, CgTy::U32) | (BinOp::Lt, CgTy::Tick) => Ok(BinaryOp::LtU32),
        (BinOp::Lt, CgTy::I32) => Ok(BinaryOp::LtI32),
        (BinOp::LtEq, CgTy::F32) => Ok(BinaryOp::LeF32),
        (BinOp::LtEq, CgTy::U32) | (BinOp::LtEq, CgTy::Tick) => Ok(BinaryOp::LeU32),
        (BinOp::LtEq, CgTy::I32) => Ok(BinaryOp::LeI32),
        (BinOp::Gt, CgTy::F32) => Ok(BinaryOp::GtF32),
        (BinOp::Gt, CgTy::U32) | (BinOp::Gt, CgTy::Tick) => Ok(BinaryOp::GtU32),
        (BinOp::Gt, CgTy::I32) => Ok(BinaryOp::GtI32),
        (BinOp::GtEq, CgTy::F32) => Ok(BinaryOp::GeF32),
        (BinOp::GtEq, CgTy::U32) | (BinOp::GtEq, CgTy::Tick) => Ok(BinaryOp::GeU32),
        (BinOp::GtEq, CgTy::I32) => Ok(BinaryOp::GeI32),
        (BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq, _) => {
            Err(LoweringError::IllTypedExpression {
                expected: CgTy::F32,
                got: ty,
                span,
            })
        }

        // Arithmetic — F32, U32, I32, plus Vec3 (componentwise +/-).
        (BinOp::Add, CgTy::F32) => Ok(BinaryOp::AddF32),
        (BinOp::Add, CgTy::U32) => Ok(BinaryOp::AddU32),
        (BinOp::Add, CgTy::I32) => Ok(BinaryOp::AddI32),
        (BinOp::Add, CgTy::Vec3F32) => Ok(BinaryOp::AddVec3),
        (BinOp::Sub, CgTy::F32) => Ok(BinaryOp::SubF32),
        (BinOp::Sub, CgTy::U32) => Ok(BinaryOp::SubU32),
        (BinOp::Sub, CgTy::I32) => Ok(BinaryOp::SubI32),
        (BinOp::Sub, CgTy::Vec3F32) => Ok(BinaryOp::SubVec3),
        (BinOp::Mul, CgTy::F32) => Ok(BinaryOp::MulF32),
        (BinOp::Mul, CgTy::U32) => Ok(BinaryOp::MulU32),
        (BinOp::Mul, CgTy::I32) => Ok(BinaryOp::MulI32),
        (BinOp::Div, CgTy::F32) => Ok(BinaryOp::DivF32),
        (BinOp::Div, CgTy::U32) => Ok(BinaryOp::DivU32),
        (BinOp::Div, CgTy::I32) => Ok(BinaryOp::DivI32),
        // Vec3 mul/div not yet supported — boids steering uses only +/-
        // today. When weighted-sum forms (`alignment * weight + ...`)
        // arrive, add Vec3-by-scalar variants here.
        (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, _) => {
            Err(LoweringError::IllTypedExpression {
                expected: CgTy::F32,
                got: ty,
                span,
            })
        }

        // Mod — no CG variant in v1.
        (BinOp::Mod, _) => Err(LoweringError::UnsupportedBinaryOp { op, span }),
    }
}

/// Lower a unary operator. The CG-side variant is picked from operand
/// type (`Neg`/`Abs`/`Sqrt` go through `F32`/`I32`; `Not` is `Bool`-only).
fn lower_unary(
    op: UnOp,
    arg: &IrExprNode,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let arg_id = lower_expr(arg, ctx)?;
    let arg_ty = typecheck_node(ctx, arg_id, arg.span)?;
    let cg_op = pick_unary_op(op, arg_ty, span)?;
    let result_ty = cg_op.result_ty();
    add(
        ctx,
        CgExpr::Unary {
            op: cg_op,
            arg: arg_id,
            ty: result_ty,
        },
        span,
    )
}

fn pick_unary_op(op: UnOp, ty: CgTy, span: Span) -> Result<UnaryOp, LoweringError> {
    match (op, ty) {
        (UnOp::Not, CgTy::Bool) => Ok(UnaryOp::NotBool),
        (UnOp::Not, _) => Err(LoweringError::IllTypedExpression {
            expected: CgTy::Bool,
            got: ty,
            span,
        }),
        (UnOp::Neg, CgTy::F32) => Ok(UnaryOp::NegF32),
        (UnOp::Neg, CgTy::I32) => Ok(UnaryOp::NegI32),
        (UnOp::Neg, _) => Err(LoweringError::IllTypedExpression {
            expected: CgTy::F32,
            got: ty,
            span,
        }),
    }
}

/// Lower an `if cond then a else b` AST node into a [`CgExpr::Select`].
fn lower_select(
    cond: &IrExprNode,
    then_expr: &IrExprNode,
    else_expr: &IrExprNode,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let cond_id = lower_expr(cond, ctx)?;
    let then_id = lower_expr(then_expr, ctx)?;
    let else_id = lower_expr(else_expr, ctx)?;
    let cond_ty = typecheck_node(ctx, cond_id, cond.span)?;
    if cond_ty != CgTy::Bool {
        return Err(LoweringError::IllTypedExpression {
            expected: CgTy::Bool,
            got: cond_ty,
            span,
        });
    }
    let then_ty = typecheck_node(ctx, then_id, then_expr.span)?;
    let else_ty = typecheck_node(ctx, else_id, else_expr.span)?;
    if then_ty != else_ty {
        return Err(LoweringError::SelectArmMismatch {
            then_ty,
            else_ty,
            span,
        });
    }
    add(
        ctx,
        CgExpr::Select {
            cond: cond_id,
            then: then_id,
            else_: else_id,
            ty: then_ty,
        },
        span,
    )
}

/// Lower a [`Builtin`] call to a [`CgExpr::Builtin`]. The CG-side
/// `BuiltinId` variant is picked from the AST `Builtin` enum + (where
/// applicable) operand types.
fn lower_builtin_call(
    builtin: Builtin,
    args: &[IrCallArg],
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    // Aggregations / quantifiers are AST-level dedicated nodes (Fold,
    // Quantifier) — they don't appear here as `BuiltinCall(Min, _)` in
    // their fold-shape, but the parser does produce `BuiltinCall(Min,
    // [a, b])` for the pairwise shape. Differentiate by arity.
    match builtin {
        Builtin::Forall | Builtin::Exists | Builtin::Count | Builtin::Sum => {
            return Err(LoweringError::UnsupportedBuiltin { builtin, span });
        }
        _ => {}
    }

    // Lower every argument first; then dispatch on the typed shape.
    let mut arg_ids = Vec::with_capacity(args.len());
    let mut arg_tys = Vec::with_capacity(args.len());
    for a in args {
        let id = lower_expr(&a.value, ctx)?;
        let ty = typecheck_node(ctx, id, a.value.span)?;
        arg_ids.push(id);
        arg_tys.push(ty);
    }

    match builtin {
        Builtin::Distance | Builtin::PlanarDistance | Builtin::ZSeparation => {
            expect_arity(builtin, 2, args.len(), span)?;
            let fn_id = match builtin {
                Builtin::Distance => BuiltinId::Distance,
                Builtin::PlanarDistance => BuiltinId::PlanarDistance,
                Builtin::ZSeparation => BuiltinId::ZSeparation,
                _ => unreachable!("outer match restricts to 3 distance variants"),
            };
            // Operand types must both be Vec3F32. Type checker enforces
            // it; we run that on the parent below.
            let result_ty = CgTy::F32;
            add(
                ctx,
                CgExpr::Builtin {
                    fn_id,
                    args: arg_ids,
                    ty: result_ty,
                },
                span,
            )
        }
        Builtin::Entity => {
            expect_arity(builtin, 1, args.len(), span)?;
            let result_ty = CgTy::AgentId;
            add(
                ctx,
                CgExpr::Builtin {
                    fn_id: BuiltinId::Entity,
                    args: arg_ids,
                    ty: result_ty,
                },
                span,
            )
        }
        Builtin::Floor | Builtin::Ceil | Builtin::Round
        | Builtin::Ln | Builtin::Log2 | Builtin::Log10 => {
            expect_arity(builtin, 1, args.len(), span)?;
            let fn_id = match builtin {
                Builtin::Floor => BuiltinId::Floor,
                Builtin::Ceil => BuiltinId::Ceil,
                Builtin::Round => BuiltinId::Round,
                Builtin::Ln => BuiltinId::Ln,
                Builtin::Log2 => BuiltinId::Log2,
                Builtin::Log10 => BuiltinId::Log10,
                _ => unreachable!("outer match restricts to 6 unary-f32 builtins"),
            };
            add(
                ctx,
                CgExpr::Builtin {
                    fn_id,
                    args: arg_ids,
                    ty: CgTy::F32,
                },
                span,
            )
        }
        Builtin::Sqrt => {
            // `sqrt` lowers to a `UnaryOp` (CG IR represents shape-pure
            // scalar functions there). Surface this rewrite explicitly.
            expect_arity(builtin, 1, args.len(), span)?;
            // Re-use the already-pushed arg id; build a Unary node.
            let arg_id = arg_ids[0];
            let arg_ty = arg_tys[0];
            if arg_ty != CgTy::F32 {
                return Err(LoweringError::IllTypedExpression {
                    expected: CgTy::F32,
                    got: arg_ty,
                    span,
                });
            }
            add(
                ctx,
                CgExpr::Unary {
                    op: UnaryOp::SqrtF32,
                    arg: arg_id,
                    ty: CgTy::F32,
                },
                span,
            )
        }
        Builtin::Abs => {
            // Same UnaryOp rewrite as `sqrt`, but typed `F32` or `I32`.
            expect_arity(builtin, 1, args.len(), span)?;
            let arg_id = arg_ids[0];
            let arg_ty = arg_tys[0];
            let unary = match arg_ty {
                CgTy::F32 => UnaryOp::AbsF32,
                CgTy::I32 => UnaryOp::AbsI32,
                _ => {
                    return Err(LoweringError::NumericBuiltinNonNumericOperand {
                        builtin,
                        operand_index: 0,
                        got: arg_ty,
                        span,
                    });
                }
            };
            add(
                ctx,
                CgExpr::Unary {
                    op: unary,
                    arg: arg_id,
                    ty: arg_ty,
                },
                span,
            )
        }
        Builtin::Min => lower_pairwise_numeric(
            builtin,
            BuiltinIdCtor::Min,
            &arg_ids,
            &arg_tys,
            args,
            span,
            ctx,
        ),
        Builtin::Max => lower_pairwise_numeric(
            builtin,
            BuiltinIdCtor::Max,
            &arg_ids,
            &arg_tys,
            args,
            span,
            ctx,
        ),
        Builtin::Clamp => {
            expect_arity(builtin, 3, args.len(), span)?;
            let nty = numeric_ty_from(builtin, arg_tys[0], 0, span)?;
            // Validate the other two are the same numeric type.
            for (idx, &t) in arg_tys.iter().enumerate().skip(1) {
                let other = numeric_ty_from(builtin, t, idx as u8, span)?;
                if other != nty {
                    return Err(LoweringError::BuiltinOperandMismatch {
                        builtin,
                        lhs_ty: nty.cg_ty(),
                        rhs_ty: other.cg_ty(),
                        span,
                    });
                }
            }
            add(
                ctx,
                CgExpr::Builtin {
                    fn_id: BuiltinId::Clamp(nty),
                    args: arg_ids,
                    ty: nty.cg_ty(),
                },
                span,
            )
        }
        Builtin::SaturatingAdd => lower_pairwise_numeric(
            builtin,
            BuiltinIdCtor::SaturatingAdd,
            &arg_ids,
            &arg_tys,
            args,
            span,
            ctx,
        ),
        Builtin::Vec3 => {
            // `vec3(x, y, z)` — three F32 operands → Vec3F32 result.
            // BuiltinId::Vec3Ctor's signature() enforces the operand
            // types at type-check time; the lowering just records the
            // CgExpr::Builtin shape with the three arg ids.
            expect_arity(builtin, 3, args.len(), span)?;
            for arg_ty in &arg_tys {
                if *arg_ty != CgTy::F32 {
                    return Err(LoweringError::IllTypedExpression {
                        expected: CgTy::F32,
                        got: *arg_ty,
                        span,
                    });
                }
            }
            add(
                ctx,
                CgExpr::Builtin {
                    fn_id: BuiltinId::Vec3Ctor,
                    args: arg_ids,
                    ty: CgTy::Vec3F32,
                },
                span,
            )
        }
        // Already filtered above.
        Builtin::Forall | Builtin::Exists | Builtin::Count | Builtin::Sum => {
            unreachable!("filtered earlier in lower_builtin_call")
        }
    }
}

/// Tag distinguishing the three pairwise-numeric AST builtins that
/// share the same lowering shape. Only used inside
/// `lower_pairwise_numeric`.
enum BuiltinIdCtor {
    Min,
    Max,
    SaturatingAdd,
}

impl BuiltinIdCtor {
    fn build(&self, t: NumericTy) -> BuiltinId {
        match self {
            BuiltinIdCtor::Min => BuiltinId::Min(t),
            BuiltinIdCtor::Max => BuiltinId::Max(t),
            BuiltinIdCtor::SaturatingAdd => BuiltinId::SaturatingAdd(t),
        }
    }
}

fn lower_pairwise_numeric(
    builtin: Builtin,
    ctor: BuiltinIdCtor,
    arg_ids: &[CgExprId],
    arg_tys: &[CgTy],
    args: &[IrCallArg],
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    expect_arity(builtin, 2, args.len(), span)?;
    let nty_lhs = numeric_ty_from(builtin, arg_tys[0], 0, span)?;
    let nty_rhs = numeric_ty_from(builtin, arg_tys[1], 1, span)?;
    if nty_lhs != nty_rhs {
        return Err(LoweringError::BuiltinOperandMismatch {
            builtin,
            lhs_ty: nty_lhs.cg_ty(),
            rhs_ty: nty_rhs.cg_ty(),
            span,
        });
    }
    let fn_id = ctor.build(nty_lhs);
    add(
        ctx,
        CgExpr::Builtin {
            fn_id,
            args: arg_ids.to_vec(),
            ty: nty_lhs.cg_ty(),
        },
        span,
    )
}

fn numeric_ty_from(
    builtin: Builtin,
    ty: CgTy,
    operand_index: u8,
    span: Span,
) -> Result<NumericTy, LoweringError> {
    match ty {
        CgTy::F32 => Ok(NumericTy::F32),
        CgTy::U32 => Ok(NumericTy::U32),
        CgTy::I32 => Ok(NumericTy::I32),
        _ => Err(LoweringError::NumericBuiltinNonNumericOperand {
            builtin,
            operand_index,
            got: ty,
            span,
        }),
    }
}

fn expect_arity(
    builtin: Builtin,
    expected: u8,
    got: usize,
    span: Span,
) -> Result<(), LoweringError> {
    if got as u8 == expected {
        Ok(())
    } else {
        Err(LoweringError::BuiltinArityMismatch {
            builtin,
            expected,
            got: got as u8,
            span,
        })
    }
}

/// Lower a `view::<name>(args)` call into a `CgExpr::Builtin { fn_id:
/// ViewCall { view }, .. }`. Result type is fetched from
/// `ctx.view_signatures` if registered; otherwise falls back to
/// `ViewKey { view }` (Phase 1's chosen phantom) and the type checker
/// surfaces an unresolved-signature error if a downstream consumer
/// requires the concrete shape.
fn lower_view_call(
    ast_ref: AstViewRef,
    args: &[IrCallArg],
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let view_id = *ctx
        .view_ids
        .get(&ast_ref)
        .ok_or(LoweringError::UnknownView { ast_ref, span })?;

    // Lazy-view inlining (Task 5.5c). When the driver registered a
    // body snapshot for this view_id, substitute the body directly
    // at the call site instead of emitting a `BuiltinId::ViewCall`.
    // This sidesteps the `BuiltinSignature::ViewCall` type-check
    // path for lazy views entirely — `view_signatures` only needs
    // entries for materialized views (which still lower as
    // `BuiltinId::ViewCall`).
    if let Some(snapshot) = ctx.lazy_view_bodies.get(&view_id).cloned() {
        if snapshot.param_locals.len() != args.len() {
            return Err(LoweringError::ViewCallArityMismatch {
                view: view_id,
                expected: snapshot.param_locals.len(),
                got: args.len(),
                span,
            });
        }
        // Build the binder map: i-th param's LocalRef → i-th arg's IrExprNode.
        let mut binder_map: HashMap<LocalRef, IrExprNode> = HashMap::new();
        for (param_local, call_arg) in snapshot.param_locals.iter().zip(args.iter()) {
            binder_map.insert(*param_local, call_arg.value.clone());
        }
        // Substitute and lower the result.
        let substituted = substitute_locals(&snapshot.body, &binder_map);
        return lower_expr(&substituted, ctx);
    }

    // Wildcard short-circuit (Phase 6 Task 2.5): if any arg is a bare
    // wildcard `_`, the call is a per-unit-fold-over-all-candidates
    // shape that the current IR doesn't represent. Rather than thread
    // the wildcard through (which the type-checker rejects because the
    // wildcard's typed-default substitution doesn't match the view's
    // declared signature), short-circuit the whole view-call to a
    // type-appropriate zero literal. Same B1 semantic as the
    // PerUnit-with-wildcard short-circuit: empty view storage produces
    // 0; the call's result is 0.
    if args.iter().any(|a| is_wildcard_local(&a.value)) {
        let result_ty = ctx
            .view_signatures
            .get(&view_id)
            .map(|(_, r)| *r)
            .unwrap_or(CgTy::F32);
        let zero = match result_ty {
            CgTy::F32 => CgExpr::Lit(LitValue::F32(0.0)),
            CgTy::U32 => CgExpr::Lit(LitValue::U32(0)),
            CgTy::AgentId => CgExpr::Lit(LitValue::AgentId(0)),
            // Other typed defaults — fall through to F32(0) since views
            // historically return numeric scalars. ViewKey shouldn't
            // appear here (the registered-signature path returns a
            // concrete type).
            _ => CgExpr::Lit(LitValue::F32(0.0)),
        };
        return add(ctx, zero, span);
    }

    // Materialized-view path: lower as a typed `BuiltinId::ViewCall`.
    let mut arg_ids = Vec::with_capacity(args.len());
    for a in args {
        let id = lower_expr(&a.value, ctx)?;
        arg_ids.push(id);
    }
    // Result type — pulled from the context's signature registry, or
    // defaulted to `ViewKey { view }` when unregistered (matches the
    // Phase 1 phantom shape).
    let result_ty = ctx
        .view_signatures
        .get(&view_id)
        .map(|(_, r)| *r)
        .unwrap_or(CgTy::ViewKey { view: view_id });
    add(
        ctx,
        CgExpr::Builtin {
            fn_id: BuiltinId::ViewCall { view: view_id },
            args: arg_ids,
            ty: result_ty,
        },
        span,
    )
}

/// Walk `expr` and return a new `IrExprNode` where every
/// `IrExpr::Local(local_ref, _)` whose ref appears in `binders`
/// is replaced by `binders[&local_ref]`. Other shapes are walked
/// recursively (children re-built with their substituted forms).
/// Span is preserved from the original node at every level.
///
/// Used only by lazy-view inlining (Task 5.5c). The walk is
/// exhaustive over `IrExpr`; literal / tag / ability / belief
/// shapes that have no binder children are returned via clone
/// without descent.
fn substitute_locals(
    expr: &IrExprNode,
    binders: &HashMap<LocalRef, IrExprNode>,
) -> IrExprNode {
    let span = expr.span;
    let kind = match &expr.kind {
        IrExpr::Local(local_ref, _name) if binders.contains_key(local_ref) => {
            // Substituted node carries the *callsite arg's* span,
            // not the param-binder's span — that's deliberate: the
            // diagnostic span for "operand is wrong type" should
            // point at the call site's argument, not the view
            // parameter declaration.
            return binders[local_ref].clone();
        }
        IrExpr::Local(_, _) => expr.kind.clone(), // unbound local — pass through
        IrExpr::Field { base, field_name, field } => IrExpr::Field {
            base: Box::new(substitute_locals(base, binders)),
            field_name: field_name.clone(),
            field: *field,
        },
        IrExpr::Binary(op, l, r) => IrExpr::Binary(
            *op,
            Box::new(substitute_locals(l, binders)),
            Box::new(substitute_locals(r, binders)),
        ),
        IrExpr::Unary(op, a) => IrExpr::Unary(*op, Box::new(substitute_locals(a, binders))),
        IrExpr::If { cond, then_expr, else_expr } => IrExpr::If {
            cond: Box::new(substitute_locals(cond, binders)),
            then_expr: Box::new(substitute_locals(then_expr, binders)),
            else_expr: else_expr
                .as_ref()
                .map(|e| Box::new(substitute_locals(e, binders))),
        },
        IrExpr::BuiltinCall(b, args) => IrExpr::BuiltinCall(
            *b,
            args.iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: substitute_locals(&a.value, binders),
                    span: a.span,
                })
                .collect(),
        ),
        IrExpr::ViewCall(vr, args) => IrExpr::ViewCall(
            *vr,
            args.iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: substitute_locals(&a.value, binders),
                    span: a.span,
                })
                .collect(),
        ),
        IrExpr::NamespaceCall { ns, method, args } => IrExpr::NamespaceCall {
            ns: *ns,
            method: method.clone(),
            args: args
                .iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: substitute_locals(&a.value, binders),
                    span: a.span,
                })
                .collect(),
        },
        // Pass-through for shapes that carry no `IrExprNode` children
        // we need to descend through.
        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::LitString(_)
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::View(_)
        | IrExpr::Verb(_)
        | IrExpr::Namespace(_)
        | IrExpr::NamespaceField { .. }
        | IrExpr::EnumVariant { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit(_)
        | IrExpr::AbilityRange
        | IrExpr::AbilityTag { .. }
        | IrExpr::Raw(_) => expr.kind.clone(),
        IrExpr::AbilityOnCooldown(inner) => {
            IrExpr::AbilityOnCooldown(Box::new(substitute_locals(inner, binders)))
        }
        IrExpr::BeliefsAccessor { observer, target, field } => IrExpr::BeliefsAccessor {
            observer: Box::new(substitute_locals(observer, binders)),
            target: Box::new(substitute_locals(target, binders)),
            field: field.clone(),
        },
        IrExpr::BeliefsConfidence { observer, target } => IrExpr::BeliefsConfidence {
            observer: Box::new(substitute_locals(observer, binders)),
            target: Box::new(substitute_locals(target, binders)),
        },
        IrExpr::BeliefsView { observer, view_name } => IrExpr::BeliefsView {
            observer: Box::new(substitute_locals(observer, binders)),
            view_name: view_name.clone(),
        },
        // Forms that *could* carry locals; descend into children.
        IrExpr::Index(base, idx) => IrExpr::Index(
            Box::new(substitute_locals(base, binders)),
            Box::new(substitute_locals(idx, binders)),
        ),
        IrExpr::VerbCall(vr, args) => IrExpr::VerbCall(
            *vr,
            args.iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: substitute_locals(&a.value, binders),
                    span: a.span,
                })
                .collect(),
        ),
        IrExpr::UnresolvedCall(name, args) => IrExpr::UnresolvedCall(
            name.clone(),
            args.iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: substitute_locals(&a.value, binders),
                    span: a.span,
                })
                .collect(),
        ),
        IrExpr::In(l, r) => IrExpr::In(
            Box::new(substitute_locals(l, binders)),
            Box::new(substitute_locals(r, binders)),
        ),
        IrExpr::Contains(l, r) => IrExpr::Contains(
            Box::new(substitute_locals(l, binders)),
            Box::new(substitute_locals(r, binders)),
        ),
        IrExpr::List(items) => IrExpr::List(
            items.iter().map(|i| substitute_locals(i, binders)).collect(),
        ),
        IrExpr::Tuple(items) => IrExpr::Tuple(
            items.iter().map(|i| substitute_locals(i, binders)).collect(),
        ),
        // Quantifier / Fold / Match / StructLit / Ctor / PerUnit:
        // these introduce new binders that may shadow our map.
        // Real lazy view bodies don't exercise these today (the
        // canonical lazy views — is_hostile, is_stunned,
        // slow_factor — use only Field, BuiltinCall,
        // NamespaceCall, Binary, If, Lit). If a future lazy view
        // does, the substituter must extend `binders` with a
        // shadow-aware walk; until then, return the unchanged
        // form so a stray reference to an outer local is
        // visible to the failure path rather than silently
        // miscompiled. Documented as a known limitation.
        IrExpr::Quantifier { .. }
        | IrExpr::Fold { .. }
        | IrExpr::Match { .. }
        | IrExpr::StructLit { .. }
        | IrExpr::Ctor { .. }
        | IrExpr::PerUnit { .. } => expr.kind.clone(),
    };
    IrExprNode { kind, span }
}

/// Map a `dsl_ast::ir::IrType` to its `CgTy` representation. Used
/// by view-signature population (Task 5.5c). Falls back to
/// `CgTy::U32` for shapes the current CG IR doesn't surface; if
/// such a view's signature is consulted, the type checker will
/// surface a mismatch downstream rather than the registration
/// itself panicking.
pub(super) fn ir_type_to_cg_ty(ty: &dsl_ast::ir::IrType) -> CgTy {
    use dsl_ast::ir::IrType as T;
    match ty {
        T::Bool => CgTy::Bool,
        T::U8 | T::U16 | T::U32 => CgTy::U32,
        T::I8 | T::I16 | T::I32 => CgTy::I32,
        T::F32 => CgTy::F32,
        T::Vec3 => CgTy::Vec3F32,
        T::AgentId => CgTy::AgentId,
        // Entity references in the DSL surface as `Agent`, `Item`, etc.
        // — they're typed AgentId-equivalents at the IR layer (the
        // resolver maps the entity reference to its primary id type).
        // Without this arm, view signatures like `(a: Agent, b: Agent)`
        // would register as `(u32, u32)` via the fallthrough, then the
        // type-checker rejects calls like `view::X(self, target)`
        // because `self`/`target` lower to typed `AgentId`. (Phase 6
        // Task 2.5: surfaced when wildcard short-circuit unblocked the
        // type-check from running on real view-call sites.)
        T::EntityRef(_) => CgTy::AgentId,
        // Tick-typed fields go through CgTy::Tick at the read
        // layer; views don't return Tick today, but reserve the
        // mapping for symmetry.
        T::U64 | T::I64 | T::F64 => CgTy::U32, // narrowed (DSL surface is 32-bit)
        // Falls through for unsupported shapes — the type checker
        // will surface a mismatch when the registered signature
        // is consulted; the registration itself shouldn't panic.
        _ => CgTy::U32,
    }
}

/// Lower an `IrExpr::NamespaceCall`. Most stdlib namespace calls don't
/// produce a single `CgExpr` — they lower to op-level constructs
/// (`SpatialQuery`, `EventRing`, etc.). The two cases that do are:
///
/// * `rng.<purpose>()` — pure expression, becomes `CgExpr::Rng`.
/// * `agents.<field>(<expr>)` — read of an agent field whose target is
///   given by a sub-expression. The sub-expression must already lower
///   to an `AgentId`-typed `CgExpr`.
///
/// Lower a `count(<binder> in agents where <pred>)` or
/// `sum(<binder> in agents where <projection>)` fold to a
/// [`CgStmt::ForEachAgent`] (pushed onto `pending_pre_stmts`) plus a
/// [`CgExpr::ReadLocal`] reading the populated accumulator.
///
/// # Why this is N²
///
/// Today the loop walks every agent slot (`for i in 0..agent_cap`).
/// No spatial index, no early-out — each fold over `agents` runs in
/// O(N) per fold-evaluating thread, so a per-agent rule that contains
/// k folds runs in O(k · N) per agent and O(k · N²) per tick. Fine
/// for the boids fixture at thousand-scale agent counts; the
/// declared `spatial_query nearby_other` in `boids.sim` is the
/// future surface that will let this same DSL form lower to a
/// bounded walk over a spatial hash instead.
///
/// # Type rules
///
/// - **Sum**: the `body` is the projection. Its computed CG type is
///   the accumulator type; init is the type's zero literal. Today
///   I32 / F32 / Vec3F32 are supported (matching the `+` operator
///   coverage in `lower_binary`).
/// - **Count**: the `body` is the predicate (Bool-typed). The
///   accumulator is I32; the per-iteration projection is
///   `select(0i, 1i, body)` so the loop sums 1 for each true case.
///
/// # Source-level binder
///
/// `binder_name` is captured from `IrExpr::Fold::binder_name` (the
/// surface identifier) and pushed onto `ctx.fold_binder_name` for the
/// duration of the body lowering, so reads of `<binder>.<field>`
/// inside the body resolve via [`AgentRef::PerPairCandidate`].
/// Restored on return.
fn lower_fold_over_agents(
    kind: dsl_ast::ast::FoldKind,
    binder_name: Option<&str>,
    iter: Option<&IrExprNode>,
    body: &IrExprNode,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    use dsl_ast::ast::FoldKind;

    let binder = binder_name.ok_or(LoweringError::UnsupportedAstNode {
        ast_label: "Fold (no binder)",
        span,
    })?;

    // Two recognised iter shapes:
    //
    // - `agents` — the unbounded N²-walk path lowered to
    //   `CgStmt::ForEachAgent`. Visits every alive agent slot.
    // - `spatial.<method>(self, ...)` — the bounded spatial-grid walk
    //   path lowered to `CgStmt::ForEachNeighbor`. Visits only the
    //   3³=27 cells surrounding the calling agent's cell. Today the
    //   `<method>` name is informational (any registered
    //   spatial_query is accepted); the cell-radius is hard-coded to
    //   1 because the runtime sizes its CELL_SIZE constant equal to
    //   the per-fixture perception radius. A future surface will
    //   thread the radius from the call args.
    let spatial_mode = match iter.map(|n| &n.kind) {
        Some(IrExpr::Namespace(NamespaceId::Agents)) => false,
        Some(IrExpr::NamespaceCall { ns: NamespaceId::Spatial, .. }) => true,
        _ => {
            return Err(LoweringError::UnsupportedAstNode {
                ast_label: "Fold (iter is not `agents` or `spatial.<query>`)",
                span,
            });
        }
    };

    // Push the binder onto the fold-binder slot so body reads of
    // `<binder>` / `<binder>.<field>` resolve to per-pair candidate.
    let prev_binder = ctx.fold_binder_name.replace(binder.to_string());

    // Lower the body. For Count, body is a Bool predicate; for Sum,
    // body is the projection (any numeric / vec type).
    let body_id = lower_expr(body, ctx)?;
    let body_ty = typecheck_node(ctx, body_id, span)?;

    // Build the (acc_ty, init, projection) triple per fold kind.
    let (acc_ty, init_id, projection_id) = match kind {
        FoldKind::Count => {
            // Count expects a Bool predicate. Reject otherwise.
            if body_ty != CgTy::Bool {
                ctx.fold_binder_name = prev_binder;
                return Err(LoweringError::TypeCheckFailure {
                    error: TypeError::ClaimedResultMismatch {
                        node: body_id,
                        expected: CgTy::Bool,
                        got: body_ty,
                    },
                    span,
                });
            }
            let zero = add(ctx, CgExpr::Lit(LitValue::I32(0)), span)?;
            let one = add(ctx, CgExpr::Lit(LitValue::I32(1)), span)?;
            let proj = add(
                ctx,
                CgExpr::Select {
                    cond: body_id,
                    then: one,
                    else_: zero,
                    ty: CgTy::I32,
                },
                span,
            )?;
            let init = add(ctx, CgExpr::Lit(LitValue::I32(0)), span)?;
            (CgTy::I32, init, proj)
        }
        FoldKind::Sum => {
            let init = match body_ty {
                CgTy::I32 => add(ctx, CgExpr::Lit(LitValue::I32(0)), span)?,
                CgTy::F32 => add(ctx, CgExpr::Lit(LitValue::F32(0.0)), span)?,
                CgTy::Vec3F32 => add(
                    ctx,
                    CgExpr::Lit(LitValue::Vec3F32 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    }),
                    span,
                )?,
                other => {
                    ctx.fold_binder_name = prev_binder;
                    return Err(LoweringError::TypeCheckFailure {
                        error: TypeError::ClaimedResultMismatch {
                            node: body_id,
                            expected: CgTy::F32,
                            got: other,
                        },
                        span,
                    });
                }
            };
            (body_ty, init, body_id)
        }
        FoldKind::Min | FoldKind::Max => {
            // Filtered out by the caller; defensive.
            ctx.fold_binder_name = prev_binder;
            return Err(LoweringError::UnsupportedAstNode {
                ast_label: "Fold (Min/Max)",
                span,
            });
        }
    };

    // Restore prior fold binder before exiting.
    ctx.fold_binder_name = prev_binder;

    // Allocate a fresh accumulator local. Pick max-existing + 1 over
    // both the AST-bound locals (`local_ids`) and the typed-local map
    // (`local_tys`) so the id is disjoint from both registries.
    let next_id = ctx
        .local_ids
        .values()
        .map(|id| id.0)
        .chain(ctx.local_tys.keys().map(|id| id.0))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let acc_local = LocalId(next_id);
    ctx.record_local_ty(acc_local, acc_ty);

    // Push the fold loop onto pending pre-stmts so the surrounding
    // stmt-list driver injects it before the consumer of this fold's
    // result. Variant choice: spatial-iter folds become
    // ForEachNeighbor (bounded 27-cell walk); plain-`agents` folds
    // become ForEachAgent (unbounded N² walk).
    let stmt = if spatial_mode {
        CgStmt::ForEachNeighbor {
            acc_local,
            acc_ty,
            init: init_id,
            projection: projection_id,
            // Hard-coded cell-radius for v1: CELL_SIZE is sized to
            // match the per-fixture perception radius, so a single-
            // cell neighborhood (3³ cells) covers everything inside
            // that radius. Larger fold radii would bump this.
            radius_cells: 1,
        }
    } else {
        CgStmt::ForEachAgent {
            acc_local,
            acc_ty,
            init: init_id,
            projection: projection_id,
        }
    };
    let stmt_id = ctx
        .builder
        .add_stmt(stmt)
        .map_err(|e| LoweringError::BuilderRejected { error: e, span })?;
    ctx.pending_pre_stmts.push(stmt_id);

    // Return a read of the accumulator. Consumers (Let, Assign,
    // Binary, …) pick this up as a normal CgExpr.
    add(
        ctx,
        CgExpr::ReadLocal {
            local: acc_local,
            ty: acc_ty,
        },
        span,
    )
}

/// All other namespace/method pairs surface as
/// [`LoweringError::UnsupportedNamespaceCall`] for now.
fn lower_namespace_call(
    ns: NamespaceId,
    method: &str,
    args: &[IrCallArg],
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    match (ns, method) {
        (NamespaceId::Rng, m) => {
            // `rng.action()`, `rng.sample()`, etc. — argument list (if
            // any) is consumed at op-level (the seed/agent/tick are
            // implicit). We accept zero args here.
            if !args.is_empty() {
                return Err(LoweringError::NamespaceCallArityMismatch {
                    ns,
                    method: m.to_string(),
                    expected: 0,
                    got: args.len(),
                    span,
                });
            }
            let purpose = match m {
                "action" => RngPurpose::Action,
                "sample" => RngPurpose::Sample,
                "shuffle" => RngPurpose::Shuffle,
                "conception" => RngPurpose::Conception,
                _ => {
                    return Err(LoweringError::UnsupportedNamespaceCall {
                        ns,
                        method: method.to_string(),
                        span,
                    })
                }
            };
            add(
                ctx,
                CgExpr::Rng {
                    purpose,
                    ty: CgTy::U32,
                },
                span,
            )
        }
        (NamespaceId::Agents, field) if args.len() == 1 => {
            // `agents.<field>(<expr>)` — typed agent-field read where
            // the target slot is computed by `<expr>`. The DSL surfaces
            // this for cross-agent reads (`agents.hp(target)` etc.).
            //
            // **Registry-first dispatch**: if `(Agents, field)` is in
            // the namespace registry (e.g.
            // `agents.is_hostile_to(target)` registered as a
            // single-arg method returning bool), the registry path
            // wins. The agent-field read is the fallback for
            // unregistered method names that match an `AgentFieldId`.
            // This ordering means the registry is the source of truth
            // for the symbol surface; falling back to agent-field
            // resolution stays compatible with the DSL's existing
            // `agents.hp(target)` shape.
            if let Some(def) = ctx
                .namespace_registry
                .namespaces
                .get(&ns)
                .and_then(|nd| nd.methods.get(field))
            {
                return lower_registered_namespace_call(ns, field, args, span, ctx, def.clone());
            }
            let target_expr = &args[0].value;
            let target_id = lower_expr(target_expr, ctx)?;
            let target_ty = typecheck_node(ctx, target_id, target_expr.span)?;
            if target_ty != CgTy::AgentId {
                return Err(LoweringError::IllTypedExpression {
                    expected: CgTy::AgentId,
                    got: target_ty,
                    span,
                });
            }
            let field_id = AgentFieldId::from_snake(field).ok_or_else(|| {
                LoweringError::UnknownAgentField {
                    field_name: field.to_string(),
                    span,
                }
            })?;
            // `data_handle_ty` produces the right `CgTy` for whatever
            // primitive the field carries — we use it to satisfy the
            // type checker's claimed-result rule on `Read`.
            let handle = DataHandle::AgentField {
                field: field_id,
                target: AgentRef::Target(target_id),
            };
            // Sanity: the field's primitive type must round-trip
            // through `data_handle_ty` — otherwise `Read` wouldn't
            // produce a meaningful CgExpr.
            let _ty = data_handle_ty(&handle);
            add(ctx, CgExpr::Read(handle), span)
        }
        _ => {
            // Registry fallback: any `(ns, method)` pair registered in
            // `namespace_registry` lowers to `CgExpr::NamespaceCall` —
            // covers `agents.is_hostile_to`, `agents.engaged_with_or`,
            // `query.nearest_hostile_to_or`, and any future namespace
            // method whose schema is recorded in the registry.
            if let Some(def) = ctx
                .namespace_registry
                .namespaces
                .get(&ns)
                .and_then(|nd| nd.methods.get(method))
            {
                return lower_registered_namespace_call(ns, method, args, span, ctx, def.clone());
            }
            Err(LoweringError::UnsupportedNamespaceCall {
                ns,
                method: method.to_string(),
                span,
            })
        }
    }
}

/// Lower a registered namespace-method call to a typed
/// [`CgExpr::NamespaceCall`]. Validates arity against the registry
/// schema; arg types are not enforced here (the type checker already
/// validated each argument's claimed type). Used by both the
/// `(Agents, _)` and the catch-all arms of
/// [`lower_namespace_call`].
fn lower_registered_namespace_call(
    ns: NamespaceId,
    method: &str,
    args: &[IrCallArg],
    span: Span,
    ctx: &mut LoweringCtx<'_>,
    def: super::super::program::MethodDef,
) -> Result<CgExprId, LoweringError> {
    if args.len() != def.arg_tys.len() {
        return Err(LoweringError::NamespaceCallArityMismatch {
            ns,
            method: method.to_string(),
            expected: def.arg_tys.len(),
            got: args.len(),
            span,
        });
    }
    let mut arg_ids = Vec::with_capacity(args.len());
    for a in args {
        arg_ids.push(lower_expr(&a.value, ctx)?);
    }
    add(
        ctx,
        CgExpr::NamespaceCall {
            ns,
            method: method.to_string(),
            args: arg_ids,
            ty: def.return_ty,
        },
        span,
    )
}

/// Confirm `AgentFieldTy` doesn't have a closed-set match arm gap. The
/// primitive-type set is referenced indirectly via [`data_handle_ty`];
/// this helper isn't called from production code, but documents the
/// invariant the lowering depends on (every `AgentFieldTy` has a
/// non-`ViewKey` `CgTy` representation).
#[allow(dead_code)]
fn _agent_field_ty_invariant(t: AgentFieldTy) -> CgTy {
    match t {
        AgentFieldTy::F32 => CgTy::F32,
        AgentFieldTy::U32 => CgTy::U32,
        AgentFieldTy::I16 => CgTy::I32,
        AgentFieldTy::Bool => CgTy::Bool,
        AgentFieldTy::Vec3 => CgTy::Vec3F32,
        AgentFieldTy::EnumU8 => CgTy::U32,
        AgentFieldTy::OptAgentId => CgTy::AgentId,
        AgentFieldTy::OptEnumU32 => CgTy::U32,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::expr::pretty;
    use dsl_ast::ast::Span as AstSpan;
    use dsl_ast::ir::LocalRef;

    // ---- helpers ----

    fn span(start: usize, end: usize) -> AstSpan {
        AstSpan::new(start, end)
    }

    fn node(kind: IrExpr) -> IrExprNode {
        IrExprNode {
            kind,
            span: span(0, 0),
        }
    }

    fn arg(value: IrExprNode) -> IrCallArg {
        let s = value.span;
        IrCallArg {
            name: None,
            value,
            span: s,
        }
    }

    fn local_self() -> IrExprNode {
        node(IrExpr::Local(LocalRef(0), "self".to_string()))
    }

    fn field_self(name: &str) -> IrExprNode {
        node(IrExpr::Field {
            base: Box::new(local_self()),
            field_name: name.to_string(),
            field: None,
        })
    }

    fn lower_to_string(ast: &IrExprNode) -> Result<String, LoweringError> {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let id = lower_expr(ast, &mut ctx)?;
        let prog = builder.finish();
        let node = &prog.exprs[id.0 as usize];
        Ok(pretty(node, &prog.exprs))
    }

    // ---- Literals ----

    #[test]
    fn literal_bool_lowers() {
        let ast = node(IrExpr::LitBool(true));
        assert_eq!(lower_to_string(&ast).unwrap(), "(lit true)");
    }

    #[test]
    fn literal_int_positive_picks_u32() {
        let ast = node(IrExpr::LitInt(5));
        assert_eq!(lower_to_string(&ast).unwrap(), "(lit 5u32)");
    }

    #[test]
    fn literal_int_negative_picks_i32() {
        let ast = node(IrExpr::LitInt(-3));
        assert_eq!(lower_to_string(&ast).unwrap(), "(lit -3i32)");
    }

    #[test]
    fn literal_float_lowers_f32() {
        let ast = node(IrExpr::LitFloat(1.5));
        assert_eq!(lower_to_string(&ast).unwrap(), "(lit 1.5f32)");
    }

    #[test]
    fn literal_int_overflow_u32_rejected() {
        let v = (u32::MAX as i64) + 1;
        let ast = node(IrExpr::LitInt(v));
        let err = lower_to_string(&ast).unwrap_err();
        match err {
            LoweringError::LiteralOutOfRange { value, target, .. } => {
                assert_eq!(value, v);
                assert_eq!(target, CgTy::U32);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn literal_int_overflow_i32_rejected() {
        let v = (i32::MIN as i64) - 1;
        let ast = node(IrExpr::LitInt(v));
        let err = lower_to_string(&ast).unwrap_err();
        match err {
            LoweringError::LiteralOutOfRange { value, target, .. } => {
                assert_eq!(value, v);
                assert_eq!(target, CgTy::I32);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // ---- Field access (the plan's `agent.hp` example) ----

    #[test]
    fn self_hp_lowers_to_read_agent_field() {
        let ast = field_self("hp");
        assert_eq!(lower_to_string(&ast).unwrap(), "(read agent.self.hp)");
    }

    #[test]
    fn self_pos_lowers_to_read_vec3_field() {
        let ast = field_self("pos");
        assert_eq!(lower_to_string(&ast).unwrap(), "(read agent.self.pos)");
    }

    #[test]
    fn self_alive_lowers_to_read_bool_field() {
        let ast = field_self("alive");
        assert_eq!(lower_to_string(&ast).unwrap(), "(read agent.self.alive)");
    }

    #[test]
    fn unknown_self_field_rejected() {
        // `nonexistent_field` is neither a real `AgentFieldId` nor a
        // virtual field synthesizer — must surface as
        // `UnknownAgentField`. (Historically this used `hp_pct`; that
        // name is now a virtual field synthesized to `hp / max_hp`,
        // covered by `self_hp_pct_synthesizes_hp_div_max_hp`.)
        let ast = field_self("nonexistent_field");
        let err = lower_to_string(&ast).unwrap_err();
        match err {
            LoweringError::UnknownAgentField { field_name, .. } => {
                assert_eq!(field_name, "nonexistent_field");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ---- Virtual fields (Task 8 of the cg-lowering gap-closure plan) ----

    #[test]
    fn self_hp_pct_synthesizes_hp_div_max_hp() {
        // `self.hp_pct` is a virtual field — no `AgentFieldId::HpPct`
        // exists. The lowering synthesizes `Read(Hp) / Read(MaxHp)`,
        // both targeting `AgentRef::Self_`.
        let ast = field_self("hp_pct");
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let id = lower_expr(&ast, &mut ctx).expect("hp_pct synthesizes");
        let prog = builder.finish();
        let root = &prog.exprs[id.0 as usize];
        // Pretty-printed canonical form: div.f32 of two agent-self reads.
        assert_eq!(
            pretty(root, &prog.exprs),
            "(div.f32 (read agent.self.hp) (read agent.self.max_hp))"
        );
        // And the typed shape — both reads must target Self_, not the
        // per-pair candidate.
        match root {
            CgExpr::Binary { op, lhs, rhs, ty } => {
                assert_eq!(*op, BinaryOp::DivF32);
                assert_eq!(*ty, CgTy::F32);
                match &prog.exprs[lhs.0 as usize] {
                    CgExpr::Read(DataHandle::AgentField { field, target }) => {
                        assert_eq!(*field, AgentFieldId::Hp);
                        assert_eq!(*target, AgentRef::Self_);
                    }
                    other => panic!("unexpected lhs: {other:?}"),
                }
                match &prog.exprs[rhs.0 as usize] {
                    CgExpr::Read(DataHandle::AgentField { field, target }) => {
                        assert_eq!(*field, AgentFieldId::MaxHp);
                        assert_eq!(*target, AgentRef::Self_);
                    }
                    other => panic!("unexpected rhs: {other:?}"),
                }
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    #[test]
    fn target_hp_pct_synthesizes_per_pair_hp_div_max_hp() {
        // Symmetry: in a pair-bound context, `target.hp_pct`
        // synthesizes the same Div, both reads tagged
        // `AgentRef::PerPairCandidate`.
        let ast = field_target("hp_pct");
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.target_local = true;
        let id = lower_expr(&ast, &mut ctx).expect("target.hp_pct synthesizes");
        let prog = builder.finish();
        let root = &prog.exprs[id.0 as usize];
        match root {
            CgExpr::Binary { op, lhs, rhs, ty } => {
                assert_eq!(*op, BinaryOp::DivF32);
                assert_eq!(*ty, CgTy::F32);
                match &prog.exprs[lhs.0 as usize] {
                    CgExpr::Read(DataHandle::AgentField { field, target }) => {
                        assert_eq!(*field, AgentFieldId::Hp);
                        assert_eq!(*target, AgentRef::PerPairCandidate);
                    }
                    other => panic!("unexpected lhs: {other:?}"),
                }
                match &prog.exprs[rhs.0 as usize] {
                    CgExpr::Read(DataHandle::AgentField { field, target }) => {
                        assert_eq!(*field, AgentFieldId::MaxHp);
                        assert_eq!(*target, AgentRef::PerPairCandidate);
                    }
                    other => panic!("unexpected rhs: {other:?}"),
                }
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    #[test]
    fn field_on_non_self_local_rejected() {
        let ast = node(IrExpr::Field {
            base: Box::new(node(IrExpr::Local(LocalRef(1), "target".to_string()))),
            field_name: "hp".to_string(),
            field: None,
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(err, LoweringError::UnsupportedFieldBase { .. }));
    }

    // ---- target.<field> binding (Task 5.5a) ----

    fn field_target(name: &str) -> IrExprNode {
        node(IrExpr::Field {
            base: Box::new(node(IrExpr::Local(LocalRef(1), "target".to_string()))),
            field_name: name.to_string(),
            field: None,
        })
    }

    #[test]
    fn target_field_with_target_local_lowers_to_per_pair_candidate_read() {
        // target.alive in a context where ctx.target_local is true
        // resolves to `Read(AgentField { target: PerPairCandidate, .. })`.
        let ast = field_target("alive");
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.target_local = true;
        let id = lower_expr(&ast, &mut ctx).expect("lowers");
        let prog = builder.finish();
        let node_at = &prog.exprs[id.0 as usize];
        // Expect: (read agent.per_pair_candidate.alive) under the
        // `pretty` pretty-printer.
        assert_eq!(
            pretty(node_at, &prog.exprs),
            "(read agent.per_pair_candidate.alive)"
        );
        // Confirm the typed handle shape exactly — Display goes through
        // the `agent.per_pair_candidate.<field>` form.
        match node_at {
            CgExpr::Read(DataHandle::AgentField { field, target }) => {
                assert_eq!(*field, AgentFieldId::Alive);
                assert_eq!(*target, AgentRef::PerPairCandidate);
            }
            other => panic!("unexpected lowered expr: {other:?}"),
        }
    }

    #[test]
    fn target_field_without_target_local_rejects_with_unsupported_field_base() {
        // Regression: outside a pair-bound context (`target_local =
        // false`, the default), `target.<field>` must NOT route through
        // `PerPairCandidate`. The lowering surfaces the same typed
        // deferral as any other unbound receiver.
        let ast = field_target("alive");
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // ctx.target_local left false (the default).
        let err = lower_expr(&ast, &mut ctx).expect_err("must reject");
        match err {
            LoweringError::UnsupportedFieldBase { field_name, .. } => {
                assert_eq!(field_name, "alive");
            }
            other => panic!("expected UnsupportedFieldBase, got {other:?}"),
        }
    }

    #[test]
    fn target_field_unknown_name_under_target_local_rejects_with_unknown_agent_field() {
        // target.<bogus> in a target-bound context: the receiver is
        // now recognised, but the field name is neither a real
        // `AgentFieldId` nor a virtual field. The lowering surfaces
        // the same `UnknownAgentField` defect as the self-side case —
        // keeps the error surface symmetric. (Historically this used
        // `hp_pct`; that name is now a virtual field — see
        // `target_hp_pct_synthesizes_per_pair_hp_div_max_hp`.)
        let ast = field_target("nonexistent_field");
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.target_local = true;
        let err = lower_expr(&ast, &mut ctx).expect_err("must reject");
        match err {
            LoweringError::UnknownAgentField { field_name, .. } => {
                assert_eq!(field_name, "nonexistent_field");
            }
            other => panic!("expected UnknownAgentField, got {other:?}"),
        }
    }

    // ---- The plan's specific rejection — `agent.alive < 5` ----

    #[test]
    fn agent_alive_lt_5_rejects_with_typed_error() {
        // agent.alive : Bool
        // 5            : U32
        // → mismatched binary operand types
        let ast = node(IrExpr::Binary(
            BinOp::Lt,
            Box::new(field_self("alive")),
            Box::new(node(IrExpr::LitInt(5))),
        ));
        let err = lower_to_string(&ast).unwrap_err();
        match err {
            LoweringError::BinaryOperandTyMismatch {
                op,
                lhs_ty,
                rhs_ty,
                ..
            } => {
                assert_eq!(op, BinOp::Lt);
                assert_eq!(lhs_ty, CgTy::Bool);
                assert_eq!(rhs_ty, CgTy::U32);
            }
            other => panic!("expected BinaryOperandTyMismatch, got {other:?}"),
        }
    }

    // ---- Signed/unsigned integer literal coercion ----

    /// `delta != 0` where `delta: i32` (event-field shape) and `0`
    /// defaults to `LitValue::U32` at lowering. The coercion in
    /// `lower_binary` should re-emit the literal as `I32` so the
    /// binary type-check accepts the shape.
    #[test]
    fn binary_i32_field_neq_u32_literal_coerces_literal_to_i32() {
        // self.slow_factor_q8 (i16 -> CgTy::I32) != 0
        let ast = node(IrExpr::Binary(
            BinOp::NotEq,
            Box::new(field_self("slow_factor_q8")),
            Box::new(node(IrExpr::LitInt(0))),
        ));
        let s = lower_to_string(&ast).unwrap();
        assert_eq!(
            s, "(ne.i32 (read agent.self.slow_factor_q8) (lit 0i32))",
            "expected i32-typed binary with literal coerced to I32, got {s:?}"
        );
    }

    /// Symmetric: literal on lhs, i32 read on rhs.
    #[test]
    fn binary_u32_literal_gt_i32_field_coerces_literal_to_i32() {
        // 0 > self.slow_factor_q8
        let ast = node(IrExpr::Binary(
            BinOp::Gt,
            Box::new(node(IrExpr::LitInt(0))),
            Box::new(field_self("slow_factor_q8")),
        ));
        let s = lower_to_string(&ast).unwrap();
        assert_eq!(
            s, "(gt.i32 (lit 0i32) (read agent.self.slow_factor_q8))",
            "expected i32-typed binary with lhs literal coerced to I32, got {s:?}"
        );
    }

    /// `delta > 0` (strict) — same i32-vs-u32 shape as the inequality
    /// case; verifies the `Gt` branch that surfaced in the
    /// post-Task-1 diagnostics.
    #[test]
    fn binary_i32_field_gt_u32_literal_coerces_literal_to_i32() {
        let ast = node(IrExpr::Binary(
            BinOp::Gt,
            Box::new(field_self("slow_factor_q8")),
            Box::new(node(IrExpr::LitInt(0))),
        ));
        let s = lower_to_string(&ast).unwrap();
        assert_eq!(
            s, "(gt.i32 (read agent.self.slow_factor_q8) (lit 0i32))"
        );
    }

    /// Two non-literal i32/u32 operands stay rejected — the
    /// coercion intentionally fires only when one side is a
    /// literal, so genuine field-vs-field mismatches surface as
    /// typing bugs.
    #[test]
    fn binary_i32_field_neq_u32_field_still_rejected_when_neither_is_literal() {
        // self.slow_factor_q8 (I32) != self.level (U32)
        let ast = node(IrExpr::Binary(
            BinOp::NotEq,
            Box::new(field_self("slow_factor_q8")),
            Box::new(field_self("level")),
        ));
        let err = lower_to_string(&ast).unwrap_err();
        match err {
            LoweringError::BinaryOperandTyMismatch {
                lhs_ty, rhs_ty, ..
            } => {
                assert_eq!(lhs_ty, CgTy::I32);
                assert_eq!(rhs_ty, CgTy::U32);
            }
            other => panic!("expected BinaryOperandTyMismatch, got {other:?}"),
        }
    }

    // ---- BinaryOp coverage ----

    #[test]
    fn binary_arithmetic_f32() {
        let ast = node(IrExpr::Binary(
            BinOp::Add,
            Box::new(field_self("hp")),
            Box::new(field_self("max_hp")),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(add.f32 (read agent.self.hp) (read agent.self.max_hp))"
        );
    }

    #[test]
    fn binary_arithmetic_u32() {
        let ast = node(IrExpr::Binary(
            BinOp::Sub,
            Box::new(field_self("level")),
            Box::new(node(IrExpr::LitInt(1))),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(sub.u32 (read agent.self.level) (lit 1u32))"
        );
    }

    #[test]
    fn binary_comparison_f32_lt() {
        // self.hp < self.max_hp
        let ast = node(IrExpr::Binary(
            BinOp::Lt,
            Box::new(field_self("hp")),
            Box::new(field_self("max_hp")),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(lt.f32 (read agent.self.hp) (read agent.self.max_hp))"
        );
    }

    #[test]
    fn binary_comparison_u32_le() {
        // self.level <= 5
        let ast = node(IrExpr::Binary(
            BinOp::LtEq,
            Box::new(field_self("level")),
            Box::new(node(IrExpr::LitInt(5))),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(le.u32 (read agent.self.level) (lit 5u32))"
        );
    }

    #[test]
    fn binary_equality_bool() {
        // self.alive == self.alive
        let ast = node(IrExpr::Binary(
            BinOp::Eq,
            Box::new(field_self("alive")),
            Box::new(field_self("alive")),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(eq.bool (read agent.self.alive) (read agent.self.alive))"
        );
    }

    #[test]
    fn binary_equality_agent_id() {
        // self.engaged_with == self.engaged_with
        let ast = node(IrExpr::Binary(
            BinOp::Eq,
            Box::new(field_self("engaged_with")),
            Box::new(field_self("engaged_with")),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(eq.agent_id (read agent.self.engaged_with) (read agent.self.engaged_with))"
        );
    }

    #[test]
    fn binary_logical_and() {
        let ast = node(IrExpr::Binary(
            BinOp::And,
            Box::new(field_self("alive")),
            Box::new(field_self("alive")),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(and (read agent.self.alive) (read agent.self.alive))"
        );
    }

    #[test]
    fn binary_logical_or() {
        let ast = node(IrExpr::Binary(
            BinOp::Or,
            Box::new(node(IrExpr::LitBool(true))),
            Box::new(node(IrExpr::LitBool(false))),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(or (lit true) (lit false))"
        );
    }

    #[test]
    fn binary_mod_unsupported() {
        let ast = node(IrExpr::Binary(
            BinOp::Mod,
            Box::new(node(IrExpr::LitInt(7))),
            Box::new(node(IrExpr::LitInt(3))),
        ));
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::UnsupportedBinaryOp { op: BinOp::Mod, .. }
        ));
    }

    #[test]
    fn binary_logical_and_on_non_bool_rejected() {
        let ast = node(IrExpr::Binary(
            BinOp::And,
            Box::new(node(IrExpr::LitInt(1))),
            Box::new(node(IrExpr::LitInt(2))),
        ));
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(err, LoweringError::IllTypedExpression { .. }));
    }

    // ---- PerUnit (Phase 6 Task 1) ----

    /// `<expr> per_unit <delta>` lowers as `expr * delta` per the AST
    /// docstring's outside-scoring semantic. Inside scoring contexts
    /// the iterate-over-view-storage semantic differs, but for empty
    /// view storage the result is identical (0 * delta = 0). This
    /// test verifies the rejection (`UnsupportedAstNode { ast_label:
    /// "PerUnit" }`) is gone and the lowering produces the expected
    /// `mul.f32` shape.
    #[test]
    fn per_unit_lowers_as_multiplication() {
        let ast = node(IrExpr::PerUnit {
            expr: Box::new(node(IrExpr::LitFloat(2.0))),
            delta: Box::new(node(IrExpr::LitFloat(0.01))),
        });
        let s = lower_to_string(&ast).unwrap();
        assert_eq!(
            s, "(mul.f32 (lit 2.0f32) (lit 0.01f32))",
            "expected per_unit to lower as mul.f32, got {s:?}"
        );
    }

    // ---- UnaryOp coverage ----

    #[test]
    fn unary_not_bool() {
        let ast = node(IrExpr::Unary(UnOp::Not, Box::new(field_self("alive"))));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(not.bool (read agent.self.alive))"
        );
    }

    #[test]
    fn unary_neg_f32() {
        let ast = node(IrExpr::Unary(UnOp::Neg, Box::new(field_self("hp"))));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(neg.f32 (read agent.self.hp))"
        );
    }

    #[test]
    fn unary_neg_i32() {
        // self.slow_factor_q8 is i16 widened to i32 in CG IR.
        let ast = node(IrExpr::Unary(
            UnOp::Neg,
            Box::new(field_self("slow_factor_q8")),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(neg.i32 (read agent.self.slow_factor_q8))"
        );
    }

    #[test]
    fn unary_not_on_u32_rejected() {
        let ast = node(IrExpr::Unary(UnOp::Not, Box::new(field_self("level"))));
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(err, LoweringError::IllTypedExpression { .. }));
    }

    #[test]
    fn unary_neg_on_bool_rejected() {
        let ast = node(IrExpr::Unary(UnOp::Neg, Box::new(field_self("alive"))));
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(err, LoweringError::IllTypedExpression { .. }));
    }

    // ---- Builtins ----

    #[test]
    fn distance_builtin_lowers() {
        // distance(self.pos, self.pos)  — DSL spec example.
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Distance,
            vec![arg(field_self("pos")), arg(field_self("pos"))],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(builtin.distance (read agent.self.pos) (read agent.self.pos))"
        );
    }

    #[test]
    fn planar_distance_builtin_lowers() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::PlanarDistance,
            vec![arg(field_self("pos")), arg(field_self("pos"))],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(builtin.planar_distance (read agent.self.pos) (read agent.self.pos))"
        );
    }

    #[test]
    fn z_separation_builtin_lowers() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::ZSeparation,
            vec![arg(field_self("pos")), arg(field_self("pos"))],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(builtin.z_separation (read agent.self.pos) (read agent.self.pos))"
        );
    }

    #[test]
    fn distance_arity_mismatch_rejected() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Distance,
            vec![arg(field_self("pos"))],
        ));
        let err = lower_to_string(&ast).unwrap_err();
        match err {
            LoweringError::BuiltinArityMismatch {
                builtin: Builtin::Distance,
                expected,
                got,
                ..
            } => {
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn distance_with_non_vec3_args_fails_typecheck() {
        // distance(self.hp, self.hp) — operands must be Vec3, not F32.
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Distance,
            vec![arg(field_self("hp")), arg(field_self("hp"))],
        ));
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(err, LoweringError::TypeCheckFailure { .. }));
    }

    #[test]
    fn min_f32_pairwise_lowers() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Min,
            vec![arg(field_self("hp")), arg(field_self("max_hp"))],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(builtin.min.f32 (read agent.self.hp) (read agent.self.max_hp))"
        );
    }

    #[test]
    fn max_u32_pairwise_lowers() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Max,
            vec![
                arg(field_self("level")),
                arg(node(IrExpr::LitInt(7))),
            ],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(builtin.max.u32 (read agent.self.level) (lit 7u32))"
        );
    }

    #[test]
    fn clamp_f32_lowers() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Clamp,
            vec![
                arg(field_self("hp")),
                arg(node(IrExpr::LitFloat(0.0))),
                arg(field_self("max_hp")),
            ],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(builtin.clamp.f32 (read agent.self.hp) (lit 0.0f32) (read agent.self.max_hp))"
        );
    }

    #[test]
    fn saturating_add_u32_lowers() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::SaturatingAdd,
            vec![
                arg(field_self("level")),
                arg(node(IrExpr::LitInt(1))),
            ],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(builtin.saturating_add.u32 (read agent.self.level) (lit 1u32))"
        );
    }

    #[test]
    fn entity_builtin_lowers() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Entity,
            vec![arg(field_self("engaged_with"))],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(builtin.entity (read agent.self.engaged_with))"
        );
    }

    #[test]
    fn floor_builtin_lowers() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Floor,
            vec![arg(field_self("hp"))],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(builtin.floor (read agent.self.hp))"
        );
    }

    #[test]
    fn ceil_ln_log2_log10_round_lower() {
        // Quick smoke test that all five additional unary-f32 builtins
        // share the same path.
        for (b, label) in [
            (Builtin::Ceil, "ceil"),
            (Builtin::Round, "round"),
            (Builtin::Ln, "ln"),
            (Builtin::Log2, "log2"),
            (Builtin::Log10, "log10"),
        ] {
            let ast = node(IrExpr::BuiltinCall(b, vec![arg(field_self("hp"))]));
            let s = lower_to_string(&ast).unwrap();
            assert_eq!(
                s,
                format!("(builtin.{label} (read agent.self.hp))"),
                "builtin {label}"
            );
        }
    }

    #[test]
    fn sqrt_builtin_rewrites_to_unary() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Sqrt,
            vec![arg(field_self("hp"))],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(sqrt.f32 (read agent.self.hp))"
        );
    }

    #[test]
    fn abs_f32_rewrites_to_unary() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Abs,
            vec![arg(field_self("hp"))],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(abs.f32 (read agent.self.hp))"
        );
    }

    #[test]
    fn abs_i32_rewrites_to_unary() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Abs,
            vec![arg(field_self("slow_factor_q8"))],
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(abs.i32 (read agent.self.slow_factor_q8))"
        );
    }

    #[test]
    fn abs_on_bool_rejected() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Abs,
            vec![arg(field_self("alive"))],
        ));
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::NumericBuiltinNonNumericOperand { .. }
        ));
    }

    #[test]
    fn min_on_bool_rejected() {
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Min,
            vec![
                arg(field_self("alive")),
                arg(field_self("alive")),
            ],
        ));
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::NumericBuiltinNonNumericOperand { .. }
        ));
    }

    #[test]
    fn min_with_mixed_numeric_types_rejected() {
        // min(self.hp, self.level) — F32 vs U32.
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Min,
            vec![arg(field_self("hp")), arg(field_self("level"))],
        ));
        let err = lower_to_string(&ast).unwrap_err();
        // The two operand lowerings succeed independently; the
        // pairwise-numeric helper rejects the mix.
        match err {
            LoweringError::BuiltinOperandMismatch {
                builtin: Builtin::Min,
                lhs_ty,
                rhs_ty,
                ..
            } => {
                assert_eq!(lhs_ty, CgTy::F32);
                assert_eq!(rhs_ty, CgTy::U32);
            }
            other => panic!("expected BuiltinOperandMismatch(Min), got {other:?}"),
        }
    }

    #[test]
    fn clamp_with_mixed_numeric_types_rejected() {
        // clamp(self.hp, 0u32, self.max_hp) — first/last are F32,
        // middle slot is U32. The clamp lowering picks `nty` from the
        // first operand, then rejects the second when it doesn't match.
        let ast = node(IrExpr::BuiltinCall(
            Builtin::Clamp,
            vec![
                arg(field_self("hp")),
                arg(node(IrExpr::LitInt(0))),
                arg(field_self("max_hp")),
            ],
        ));
        let err = lower_to_string(&ast).unwrap_err();
        match err {
            LoweringError::BuiltinOperandMismatch {
                builtin: Builtin::Clamp,
                lhs_ty,
                rhs_ty,
                ..
            } => {
                assert_eq!(lhs_ty, CgTy::F32);
                assert_eq!(rhs_ty, CgTy::U32);
            }
            other => panic!("expected BuiltinOperandMismatch(Clamp), got {other:?}"),
        }
    }

    #[test]
    fn quantifier_builtins_unsupported() {
        for b in [Builtin::Forall, Builtin::Exists, Builtin::Count, Builtin::Sum] {
            let ast = node(IrExpr::BuiltinCall(b, vec![]));
            let err = lower_to_string(&ast).unwrap_err();
            assert!(
                matches!(err, LoweringError::UnsupportedBuiltin { .. }),
                "expected UnsupportedBuiltin for {b:?}, got {err:?}"
            );
        }
    }

    // ---- Conditional (Select) ----

    #[test]
    fn if_then_else_lowers_to_select() {
        // if self.alive then 1.0 else 0.0
        let ast = node(IrExpr::If {
            cond: Box::new(field_self("alive")),
            then_expr: Box::new(node(IrExpr::LitFloat(1.0))),
            else_expr: Some(Box::new(node(IrExpr::LitFloat(0.0)))),
        });
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(select (read agent.self.alive) (lit 1.0f32) (lit 0.0f32))"
        );
    }

    #[test]
    fn if_with_non_bool_cond_rejected() {
        // if self.hp then 1.0 else 0.0 — `cond` is f32, not bool.
        let ast = node(IrExpr::If {
            cond: Box::new(field_self("hp")),
            then_expr: Box::new(node(IrExpr::LitFloat(1.0))),
            else_expr: Some(Box::new(node(IrExpr::LitFloat(0.0)))),
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(err, LoweringError::IllTypedExpression { .. }));
    }

    #[test]
    fn if_with_arms_mismatch_rejected() {
        // if self.alive then 1.0 else 1u32
        let ast = node(IrExpr::If {
            cond: Box::new(field_self("alive")),
            then_expr: Box::new(node(IrExpr::LitFloat(1.0))),
            else_expr: Some(Box::new(node(IrExpr::LitInt(1)))),
        });
        let err = lower_to_string(&ast).unwrap_err();
        match err {
            LoweringError::SelectArmMismatch { then_ty, else_ty, .. } => {
                assert_eq!(then_ty, CgTy::F32);
                assert_eq!(else_ty, CgTy::U32);
            }
            other => panic!("expected SelectArmMismatch, got {other:?}"),
        }
    }

    #[test]
    fn if_without_else_rejected() {
        let ast = node(IrExpr::If {
            cond: Box::new(field_self("alive")),
            then_expr: Box::new(node(IrExpr::LitFloat(1.0))),
            else_expr: None,
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::UnsupportedAstNode {
                ast_label: "If(without-else)",
                ..
            }
        ));
    }

    // ---- RNG / namespace calls ----

    #[test]
    fn rng_action_lowers_to_rng_node() {
        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Rng,
            method: "action".to_string(),
            args: vec![],
        });
        assert_eq!(lower_to_string(&ast).unwrap(), "(rng action)");
    }

    #[test]
    fn rng_sample_lowers() {
        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Rng,
            method: "sample".to_string(),
            args: vec![],
        });
        assert_eq!(lower_to_string(&ast).unwrap(), "(rng sample)");
    }

    #[test]
    fn rng_unknown_purpose_rejected() {
        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Rng,
            method: "uniform".to_string(),
            args: vec![],
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::UnsupportedNamespaceCall { .. }
        ));
    }

    #[test]
    fn rng_with_extra_args_rejected_with_namespace_arity() {
        // `rng.action(<extra>)` — RNG draws are nullary at the
        // expression layer; passing args surfaces the typed
        // namespace-call arity mismatch rather than the builtin one.
        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Rng,
            method: "action".to_string(),
            args: vec![arg(node(IrExpr::LitInt(0)))],
        });
        let err = lower_to_string(&ast).unwrap_err();
        match err {
            LoweringError::NamespaceCallArityMismatch {
                ns,
                method,
                expected,
                got,
                ..
            } => {
                assert_eq!(ns, NamespaceId::Rng);
                assert_eq!(method, "action");
                assert_eq!(expected, 0);
                assert_eq!(got, 1);
            }
            other => panic!("expected NamespaceCallArityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn agents_pos_with_target_expr_lowers_to_target_read() {
        // agents.pos(self.engaged_with) — engaged_with is AgentId, so
        // the resulting Read uses AgentRef::Target(child_id).
        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Agents,
            method: "pos".to_string(),
            args: vec![arg(field_self("engaged_with"))],
        });
        let s = lower_to_string(&ast).unwrap();
        // The target-id varies based on arena ordering, but the prefix
        // is stable.
        assert!(
            s.starts_with("(read agent.target(#"),
            "unexpected lowering: {s}"
        );
        assert!(s.ends_with(").pos)"));
    }

    #[test]
    fn agents_field_with_non_agent_id_arg_rejected() {
        // agents.hp(self.hp) — arg is f32, not AgentId.
        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Agents,
            method: "hp".to_string(),
            args: vec![arg(field_self("hp"))],
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(err, LoweringError::IllTypedExpression { .. }));
    }

    #[test]
    fn unsupported_namespace_call_typed_error() {
        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Cascade,
            method: "iterations".to_string(),
            args: vec![],
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::UnsupportedNamespaceCall { .. }
        ));
    }

    #[test]
    fn unsupported_non_config_namespace_field_typed_error() {
        let ast = node(IrExpr::NamespaceField {
            ns: NamespaceId::World,
            field: "tick".to_string(),
            ty: dsl_ast::ir::IrType::U64,
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::UnsupportedNamespaceField { .. }
        ));
    }

    // ---- Registry-driven namespace lowering (Task 4 of CG lowering gap closure) ----

    /// Build a `LoweringCtx` whose namespace registry has the
    /// `agents.is_hostile_to(a, b)` method registered. Mirrors the
    /// driver's `populate_namespace_registry`. Used by the registry-
    /// path tests below.
    fn ctx_with_agents_is_hostile_to_registered<'a>(
        builder: &'a mut CgProgramBuilder,
    ) -> LoweringCtx<'a> {
        use crate::cg::program::{MethodDef, NamespaceDef};
        let mut ctx = LoweringCtx::new(builder);
        let mut agents = NamespaceDef {
            name: "agents".to_string(),
            ..NamespaceDef::default()
        };
        agents.methods.insert(
            "is_hostile_to".to_string(),
            MethodDef {
                return_ty: CgTy::Bool,
                arg_tys: vec![CgTy::AgentId, CgTy::AgentId],
                wgsl_fn_name: "agents_is_hostile_to".to_string(),
                wgsl_stub: "fn agents_is_hostile_to(a: u32, b: u32) -> bool { return false; }"
                    .to_string(),
            },
        );
        ctx.namespace_registry
            .namespaces
            .insert(NamespaceId::Agents, agents);
        ctx
    }

    #[test]
    fn agents_is_hostile_to_registry_lowers_to_namespace_call() {
        // `agents.is_hostile_to(self, self)` with the registry entry
        // present → CgExpr::NamespaceCall { ns: Agents, method:
        // "is_hostile_to", args: [self, self], ty: Bool }.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = ctx_with_agents_is_hostile_to_registered(&mut builder);

        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Agents,
            method: "is_hostile_to".to_string(),
            args: vec![arg(local_self()), arg(local_self())],
        });
        let id = lower_expr(&ast, &mut ctx).expect("lowers");
        let prog = builder.finish();
        let lowered = &prog.exprs[id.0 as usize];
        match lowered {
            CgExpr::NamespaceCall {
                ns, method, args, ty,
            } => {
                assert_eq!(*ns, NamespaceId::Agents);
                assert_eq!(method, "is_hostile_to");
                assert_eq!(args.len(), 2);
                assert_eq!(*ty, CgTy::Bool);
            }
            other => panic!("expected NamespaceCall, got {other:?}"),
        }
    }

    #[test]
    fn world_tick_registry_lowers_to_namespace_field() {
        // `world.tick` with a `World.tick` field registered → typed
        // `CgExpr::NamespaceField { ns: World, field: "tick", ty: U32 }`.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        use crate::cg::program::{FieldDef, NamespaceDef, WgslAccessForm};
        let mut world = NamespaceDef {
            name: "world".to_string(),
            ..NamespaceDef::default()
        };
        world.fields.insert(
            "tick".to_string(),
            FieldDef {
                ty: CgTy::U32,
                wgsl_access: WgslAccessForm::PreambleLocal {
                    local_name: "tick".to_string(),
                },
            },
        );
        ctx.namespace_registry
            .namespaces
            .insert(NamespaceId::World, world);

        let ast = node(IrExpr::NamespaceField {
            ns: NamespaceId::World,
            field: "tick".to_string(),
            ty: dsl_ast::ir::IrType::U32,
        });
        let id = lower_expr(&ast, &mut ctx).expect("lowers");
        let prog = builder.finish();
        let lowered = &prog.exprs[id.0 as usize];
        match lowered {
            CgExpr::NamespaceField { ns, field, ty } => {
                assert_eq!(*ns, NamespaceId::World);
                assert_eq!(field, "tick");
                assert_eq!(*ty, CgTy::U32);
            }
            other => panic!("expected NamespaceField, got {other:?}"),
        }
    }

    #[test]
    fn registered_namespace_call_arity_mismatch_typed_error() {
        // `agents.is_hostile_to(self)` with a 2-arg registry entry →
        // typed NamespaceCallArityMismatch.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = ctx_with_agents_is_hostile_to_registered(&mut builder);

        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Agents,
            method: "is_hostile_to".to_string(),
            args: vec![arg(local_self())],
        });
        let err = lower_expr(&ast, &mut ctx).expect_err("must reject");
        match err {
            LoweringError::NamespaceCallArityMismatch {
                ns,
                method,
                expected,
                got,
                ..
            } => {
                assert_eq!(ns, NamespaceId::Agents);
                assert_eq!(method, "is_hostile_to");
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("expected NamespaceCallArityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn unregistered_query_call_falls_through_to_unsupported() {
        // `query.nearest_hostile_to_or` with no registry entry →
        // typed UnsupportedNamespaceCall (the catch-all arm).
        let ast = node(IrExpr::NamespaceCall {
            ns: NamespaceId::Query,
            method: "nearest_hostile_to_or".to_string(),
            args: vec![],
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::UnsupportedNamespaceCall { .. }
        ));
    }

    // ---- Config NamespaceField → ConfigConst (Task 5.5c, Patch 1) ----

    #[test]
    fn lowers_namespace_field_to_config_const() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_config_const(
            NamespaceId::Config,
            "combat.attack_range".to_string(),
            ConfigConstId(0),
        );

        let ast = node(IrExpr::NamespaceField {
            ns: NamespaceId::Config,
            field: "combat.attack_range".to_string(),
            ty: dsl_ast::ir::IrType::F32,
        });
        let id = lower_expr(&ast, &mut ctx).unwrap();
        let prog = builder.finish();
        let node = &prog.exprs[id.0 as usize];
        let s = pretty(node, &prog.exprs);
        assert!(s.starts_with("(read config"), "got pretty: {s}");
    }

    #[test]
    fn unknown_namespace_field_returns_typed_error() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // No registry entries.
        let ast = node(IrExpr::NamespaceField {
            ns: NamespaceId::Config,
            field: "combat.attack_range".to_string(),
            ty: dsl_ast::ir::IrType::F32,
        });
        let err = lower_expr(&ast, &mut ctx).expect_err("must be typed error");
        match err {
            LoweringError::UnknownConfigField { ns, field, .. } => {
                assert_eq!(ns, NamespaceId::Config);
                assert_eq!(field, "combat.attack_range");
            }
            other => panic!("expected UnknownConfigField, got {other:?}"),
        }
    }

    // ---- Lazy view inlining (Task 5.5c, Patch 2) ----

    #[test]
    fn inlines_lazy_view_at_call_site() {
        // Lazy view body: just `LocalRef(0)` (the view's first param).
        // Calling with `LitBool(true)` should inline the literal,
        // bypassing `BuiltinId::ViewCall`.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ast_ref = AstViewRef(0);
        let view_id = ViewId(0);
        ctx.register_view(ast_ref, view_id);
        let snapshot = LazyViewSnapshot {
            param_locals: vec![LocalRef(0)],
            body: node(IrExpr::Local(LocalRef(0), "a".to_string())),
        };
        ctx.register_lazy_view_body(view_id, snapshot);

        let ast = node(IrExpr::ViewCall(
            ast_ref,
            vec![arg(node(IrExpr::LitBool(true)))],
        ));
        let id = lower_expr(&ast, &mut ctx).unwrap();
        let prog = builder.finish();
        let node = &prog.exprs[id.0 as usize];
        // Should be the literal, not a builtin.view_call.
        assert_eq!(pretty(node, &prog.exprs), "(lit true)");
    }

    #[test]
    fn materialized_view_call_uses_builtin_view_call() {
        // No lazy body registered → call falls through to the
        // materialized BuiltinId::ViewCall path.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ast_ref = AstViewRef(0);
        let view_id = ViewId(0);
        ctx.register_view(ast_ref, view_id);
        ctx.register_view_signature(view_id, vec![CgTy::AgentId], CgTy::F32);

        let ast = node(IrExpr::ViewCall(
            ast_ref,
            vec![arg(field_self("engaged_with"))],
        ));
        let id = lower_expr(&ast, &mut ctx).unwrap();
        let prog = builder.finish();
        let node = &prog.exprs[id.0 as usize];
        let s = pretty(node, &prog.exprs);
        assert!(s.starts_with("(builtin.view_call."), "got: {s}");
    }

    #[test]
    fn lazy_view_arity_mismatch_returns_typed_error() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ast_ref = AstViewRef(0);
        let view_id = ViewId(0);
        ctx.register_view(ast_ref, view_id);
        // 2-param body, called with 1 arg.
        let snapshot = LazyViewSnapshot {
            param_locals: vec![LocalRef(0), LocalRef(1)],
            body: node(IrExpr::Local(LocalRef(0), "a".to_string())),
        };
        ctx.register_lazy_view_body(view_id, snapshot);

        let ast = node(IrExpr::ViewCall(
            ast_ref,
            vec![arg(node(IrExpr::LitBool(true)))],
        ));
        let err = lower_expr(&ast, &mut ctx).expect_err("arity mismatch");
        match err {
            LoweringError::ViewCallArityMismatch {
                view, expected, got, ..
            } => {
                assert_eq!(view, view_id);
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("expected ViewCallArityMismatch, got {other:?}"),
        }
    }

    // ---- ViewCall ----

    #[test]
    fn view_call_with_registered_signature_lowers() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ast_ref = AstViewRef(0);
        let view_id = ViewId(0);
        ctx.register_view(ast_ref, view_id);
        ctx.register_view_signature(view_id, vec![CgTy::AgentId], CgTy::Bool);

        // view::is_hostile(self.engaged_with)
        let ast = node(IrExpr::ViewCall(
            ast_ref,
            vec![arg(field_self("engaged_with"))],
        ));
        let id = lower_expr(&ast, &mut ctx).unwrap();
        let prog = builder.finish();
        let node = &prog.exprs[id.0 as usize];
        assert_eq!(
            pretty(node, &prog.exprs),
            "(builtin.view_call.#0 (read agent.self.engaged_with))"
        );
    }

    #[test]
    fn view_call_unknown_ref_rejected() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ast = node(IrExpr::ViewCall(AstViewRef(99), vec![]));
        let err = lower_expr(&ast, &mut ctx).unwrap_err();
        assert!(matches!(err, LoweringError::UnknownView { .. }));
    }

    // ---- Local references ----

    #[test]
    fn bare_local_self_lowers_to_agent_self_id() {
        // Task 5.5d: bare `self` resolves to `CgExpr::AgentSelfId`.
        let ast = local_self();
        assert_eq!(lower_to_string(&ast).unwrap(), "(agent self_id)");
    }

    #[test]
    fn bare_target_in_pair_bound_lowers_to_per_pair_candidate_id() {
        let ast = node(IrExpr::Local(LocalRef(1), "target".to_string()));
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.target_local = true;
        let id = lower_expr(&ast, &mut ctx).expect("lowers");
        let prog = builder.finish();
        let node_at = &prog.exprs[id.0 as usize];
        assert_eq!(pretty(node_at, &prog.exprs), "(agent per_pair_candidate_id)");
    }

    #[test]
    fn bare_target_outside_pair_bound_rejected() {
        let ast = node(IrExpr::Local(LocalRef(1), "target".to_string()));
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // ctx.target_local left false.
        let err = lower_expr(&ast, &mut ctx).expect_err("must reject");
        match err {
            LoweringError::UnsupportedLocalBinding { name, .. } => {
                assert_eq!(name, "target");
            }
            other => panic!("expected UnsupportedLocalBinding, got {other:?}"),
        }
    }

    #[test]
    fn bare_unknown_local_rejected() {
        let ast = node(IrExpr::Local(LocalRef(2), "foo".to_string()));
        let err = lower_to_string(&ast).expect_err("must reject");
        match err {
            LoweringError::UnsupportedLocalBinding { name, .. } => {
                assert_eq!(name, "foo");
            }
            other => panic!("expected UnsupportedLocalBinding, got {other:?}"),
        }
    }

    #[test]
    fn target_neq_self_lowers_in_pair_bound() {
        // (target != self) under target_local = true.
        let ast = node(IrExpr::Binary(
            BinOp::NotEq,
            Box::new(node(IrExpr::Local(LocalRef(1), "target".to_string()))),
            Box::new(node(IrExpr::Local(LocalRef(0), "self".to_string()))),
        ));
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.target_local = true;
        let id = lower_expr(&ast, &mut ctx).expect("lowers");
        let prog = builder.finish();
        let node_at = &prog.exprs[id.0 as usize];
        assert_eq!(
            pretty(node_at, &prog.exprs),
            "(ne.agent_id (agent per_pair_candidate_id) (agent self_id))"
        );
    }

    #[test]
    fn let_bound_local_read_lowers_to_read_local() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_local(LocalRef(7), LocalId(3));
        ctx.record_local_ty(LocalId(3), CgTy::F32);

        let ast = node(IrExpr::Local(LocalRef(7), "x".to_string()));
        let id = lower_expr(&ast, &mut ctx).expect("lowers");
        let prog = builder.finish();
        let node_at = &prog.exprs[id.0 as usize];
        assert_eq!(pretty(node_at, &prog.exprs), "(read_local local#3 f32)");
    }

    #[test]
    fn let_bound_local_without_recorded_ty_rejected() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_local(LocalRef(7), LocalId(3));
        // Note: no ty recorded.

        let ast = node(IrExpr::Local(LocalRef(7), "x".to_string()));
        let err = lower_expr(&ast, &mut ctx).expect_err("must reject");
        match err {
            LoweringError::UnknownLocalType { local, .. } => {
                assert_eq!(local, LocalId(3));
            }
            other => panic!("expected UnknownLocalType, got {other:?}"),
        }
    }

    // ---- Unsupported AST shapes — typed deferral ----

    #[test]
    fn lit_string_unsupported() {
        let ast = node(IrExpr::LitString("foo".to_string()));
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::UnsupportedAstNode {
                ast_label: "LitString",
                ..
            }
        ));
    }

    /// `FoldKind::Min` / `Max` still surface as
    /// `UnsupportedAstNode` deferrals — Sum + Count are wired through
    /// the N²-fold `CgStmt::ForEachAgent` path; Min/Max would need
    /// distinct accumulator init (NEG_INFINITY / INFINITY) and a
    /// per-iteration reduction op different from `+`. No fixture
    /// asks for them yet.
    #[test]
    fn fold_min_max_unsupported() {
        for kind in [dsl_ast::ast::FoldKind::Min, dsl_ast::ast::FoldKind::Max] {
            let ast = node(IrExpr::Fold {
                kind,
                binder: Some(LocalRef(0)),
                binder_name: Some("x".to_string()),
                iter: Some(Box::new(node(IrExpr::Namespace(NamespaceId::Agents)))),
                body: Box::new(node(IrExpr::LitFloat(1.0))),
            });
            let err = lower_to_string(&ast).unwrap_err();
            assert!(
                matches!(err, LoweringError::UnsupportedAstNode { ast_label, .. } if ast_label.starts_with("Fold")),
                "Min/Max should defer; got: {err:?}"
            );
        }
    }

    /// `FoldKind::Count` over `agents` lowers to a `CgStmt::ForEachAgent`
    /// (pushed to `pending_pre_stmts`) plus a `CgExpr::ReadLocal`
    /// reading the populated accumulator. The expression's pretty-
    /// printed form is just the read; the loop is one stmt-arena entry
    /// over.
    #[test]
    fn fold_count_lowers_to_for_each_agent() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ast = node(IrExpr::Fold {
            kind: dsl_ast::ast::FoldKind::Count,
            binder: Some(LocalRef(0)),
            binder_name: Some("x".to_string()),
            iter: Some(Box::new(node(IrExpr::Namespace(NamespaceId::Agents)))),
            body: Box::new(node(IrExpr::LitBool(true))),
        });
        let id = lower_expr(&ast, &mut ctx).expect("Count fold lowers");
        // The ForEachAgent stmt was pushed onto pending_pre_stmts.
        assert_eq!(
            ctx.pending_pre_stmts.len(),
            1,
            "ForEachAgent stmt must land on pending_pre_stmts"
        );
        let prog = builder.finish();
        let read = &prog.exprs[id.0 as usize];
        // The fold expression evaluates to a ReadLocal of the
        // accumulator local (typed I32).
        assert!(
            matches!(read, CgExpr::ReadLocal { ty: CgTy::I32, .. }),
            "Count fold expr must read i32 accumulator; got: {read:?}"
        );
    }

    #[test]
    fn struct_lit_unsupported() {
        let ast = node(IrExpr::StructLit {
            name: "X".to_string(),
            ctor: None,
            fields: vec![],
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::UnsupportedAstNode {
                ast_label: "StructLit",
                ..
            }
        ));
    }

    #[test]
    fn ability_tag_unsupported() {
        let ast = node(IrExpr::AbilityTag {
            tag: dsl_ast::ir::AbilityTag::Physical,
        });
        let err = lower_to_string(&ast).unwrap_err();
        assert!(matches!(
            err,
            LoweringError::UnsupportedAstNode { ast_label: "AbilityTag", .. }
        ));
    }

    // ---- Span propagation ----

    #[test]
    fn lowering_error_carries_node_span() {
        // Span propagation through `UnknownAgentField`. Use a name
        // that's neither a real `AgentFieldId` nor a virtual field —
        // `hp_pct` was the historical choice but is now synthesized.
        let mut bad = field_self("nonexistent_field");
        bad.span = span(11, 22);
        let err = lower_to_string(&bad).unwrap_err();
        match err {
            LoweringError::UnknownAgentField { span: s, .. } => {
                assert_eq!(s.start, 11);
                assert_eq!(s.end, 22);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ---- Plan/spec example: `mask Attack` predicate fragment ----

    #[test]
    fn distance_lt_attack_range_full_predicate() {
        // (distance(self.pos, self.pos) < self.attack_range)
        // — analogue of `distance(self, t) < AGGRO_RANGE` in the spec
        // example, with `t` substituted by `self` so we don't need a
        // target binding (Task 2.1 doesn't yet wire those).
        let ast = node(IrExpr::Binary(
            BinOp::Lt,
            Box::new(node(IrExpr::BuiltinCall(
                Builtin::Distance,
                vec![arg(field_self("pos")), arg(field_self("pos"))],
            ))),
            Box::new(field_self("attack_range")),
        ));
        assert_eq!(
            lower_to_string(&ast).unwrap(),
            "(lt.f32 (builtin.distance (read agent.self.pos) (read agent.self.pos)) (read agent.self.attack_range))"
        );
    }

    // ---- LoweringError Display sanity ----

    #[test]
    fn lowering_error_display_includes_span_and_reason() {
        let e = LoweringError::UnknownAgentField {
            field_name: "hp_pct".to_string(),
            span: span(3, 9),
        };
        let s = format!("{}", e);
        assert!(s.contains("hp_pct"));
        assert!(s.contains("3..9"));
    }

    #[test]
    fn lowering_error_display_unsupported_ast_node() {
        let e = LoweringError::UnsupportedAstNode {
            ast_label: "Fold",
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("Fold"));
    }
}
