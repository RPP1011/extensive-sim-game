# Task 5.5c Patch Spec

## Overview

Closes 3 of the 10 deferred AST shapes in the CG lowering pipeline at HEAD `b50a6744`:

1. **Patch 1 — NamespaceField → ConfigConst.** Replace today's typed-deferral on `IrExpr::NamespaceField { ns: Config, .. }` with a real `Read(DataHandle::ConfigConst { id })`. Drives an `LoweringCtx`-side allocator (`config_const_ids`) populated by a new driver `populate_config_consts` walk over `Compilation::configs`.
2. **Patch 2 — Lazy view inlining at call sites.** When `lower_view_call` resolves to an AST view whose `kind == ViewKind::Lazy`, substitute the view's `ViewBodyIR::Expr` body at the call site (binder substitution on `IrExpr::Local(LocalRef, _)` over the view's `params`). Eliminates the `BuiltinSignature::ViewCall` type-check failure that Task 2.1 noted: lazy views never need a registered signature because they no longer reach the `BuiltinId::ViewCall` shape. Materialized view calls continue to lower through `BuiltinId::ViewCall`; their signatures are populated as a by-product (this also retires the "no view-call signature registration" Limitation in `driver.rs`).
3. **Patch 3 — EngagementQuery driver routing.** `mask_spatial_kind` becomes a predicate-aware classifier: a mask whose post-inlined predicate references engagement-flavoured access (`agents.is_hostile_to(...)`, `view::is_hostile(...)` after Patch 2 inlines, or `agents.engaged_with(...)`) routes to `SpatialQueryKind::EngagementQuery`. All other from-bearing masks keep `KinQuery`.

## Pre-conditions

Before applying, the apply agent should read these files at HEAD `b50a6744`:

- `crates/dsl_compiler/src/cg/lower/expr.rs` (lines 47–268 for `LoweringCtx`, lines 287–488 for `lower_expr`'s match arms, lines 362–371 for the current `IrExpr::NamespaceField` arm, lines 1117–1150 for `lower_view_call`).
- `crates/dsl_compiler/src/cg/lower/view.rs` (lines 137–268 for the lazy-view path; the body capture happens *here* so the driver doesn't grow a parallel walk over `comp.views`).
- `crates/dsl_compiler/src/cg/lower/driver.rs` (lines 84–96 for the "Limitations" docstring update; lines 158–249 for entry-point sequencing; lines 397–434 for `populate_views`; lines 449–473 for `mask_spatial_kind`).
- `crates/dsl_compiler/src/cg/lower/error.rs` (lines 222–246 for nearby error variant shapes; the `UnsupportedNamespaceField` variant is retired by Patch 1 unless the `field` doesn't resolve, in which case a new `UnknownConfigField` variant carries the structural reason).
- `crates/dsl_compiler/src/cg/data_handle.rs` (line 54 for `ConfigConstId`, lines 666–753 for `DataHandle` variants — `ConfigConst { id }` already exists).
- `crates/dsl_compiler/src/cg/program.rs` (lines 233 + 674–685 for the `intern_config_const_name` table — already exists).
- `crates/dsl_ast/src/ir.rs` (lines 235–246 for the `NamespaceField` and `NamespaceCall` shapes; lines 745–795 for `ViewIR`, `ViewKind`, `ViewBodyIR`; lines 815–820 for `IrParam`; lines 894–907 for `ConfigIR`/`ConfigFieldIR`; line 222 for `IrExpr::Local(LocalRef, String)`).
- `crates/dsl_compiler/src/cg/expr.rs` (lines 1080–1108 for the type-check `BuiltinSignature::ViewCall` resolver path; understand that lazy inlining sidesteps this entirely — only materialized views still surface as `BuiltinId::ViewCall`).
- `assets/sim/views.sim` (the canonical fixtures: `is_hostile`, `is_stunned`, `slow_factor` are lazy; `threat_level`, `engaged_with`, `my_enemies`, `kin_fear`, `pack_focus`, `rally_boost`, `standing`, `memory` are materialized).
- `assets/sim/masks.sim` (the canonical Attack/MoveToward/Cast masks for engagement-routing fixture seeds).

IR shapes the apply agent needs to internalise:

- `IrExpr::NamespaceField { ns: NamespaceId, field: String, ty: IrType }` carries a pre-resolved `ty` (the resolver collapsed `config.<block>.<field>` into one node). The `field` is `"<block>.<field>"` — see ir.rs:944-947 — so the lookup key for Patch 1 is `(NamespaceId::Config, "<block>.<field>")`.
- `IrExpr::Local(LocalRef, String)` is the substitution target inside a lazy view body; `LocalRef` is a typed newtype (`ref_newtype!(LocalRef)` at ir.rs:39) that the resolver assigns to each view param.
- `ViewIR::params: Vec<IrParam>` and `IrParam::local: LocalRef` provide the i-th-positional-param → LocalRef map for inlining.

## Cross-patch interactions

Anticipated conflicts and shared state with **5.5d** (the next AST batch — likely `Let` / `For` / fielded `Emit` / namespace setters per the plan's Task 5.5 list at `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md` lines 781–790) and **5.6a/5.6b** (per-kind body lowering: MaskPredicate, ScoringArgmax, SpatialQuery WGSL bodies):

1. **`LoweringCtx` field additions.** Patches 1+2 add two new fields:
   - `config_const_ids: HashMap<(NamespaceId, String), ConfigConstId>` — Patch 1.
   - `lazy_view_bodies: HashMap<ViewId, LazyViewSnapshot>` — Patch 2 (struct defined below).
   Both compose freely with 5.5d additions; 5.5d's likely additions (per-event-binder local maps for fielded `Emit`, `For`-loop binders) live on disjoint keys. Coordinate with 5.5d author to keep `LoweringCtx::new` initialisers grouped (one block per task) to minimise rebase noise.

2. **`view_signatures` overlap.** Patch 2 *also* populates `ctx.view_signatures` for every materialized view (the lazy ones are inlined and never need a signature). This was on the books as a separate Phase 3 schedule-synthesis concern (see driver.rs:87-96 docstring); Patch 2 retires it. If 5.5d / 5.6a wants to pre-populate signatures itself, it'll find the entries already present — `register_view_signature` returns the prior value and tests assert exclusive allocation, so a duplicate register surfaces immediately.

3. **Driver phase ordering.** Patch 1 adds a registry-population phase (`populate_config_consts`) that must run **before** `lower_all_masks` / `lower_all_views` / `lower_all_physics` / `lower_all_scoring`. Patch 2 adds a registry-population phase (`populate_view_bodies_and_signatures`) that **also** must run before the per-construct lowerings. Both slot into Phase 1 alongside `populate_views` (driver.rs lines 168–174). Order within Phase 1 is irrelevant; document the new walks in the Phase 1 section of the driver module docstring.

4. **`SpatialQueryKind` refinement (Patch 3).** This is the only patch that changes a value the cycle gate observes (`mask_spatial_kind`'s output drives the per-mask `DispatchShape::PerPair { source: PerPairSource::SpatialQuery(kind) }` which feeds `collect_required_spatial_kinds`). 5.6a's `SpatialQuery` WGSL body work assumes `EngagementQuery` has a real distinct body; this patch is what *exposes* Engagement to the body-emit pipeline for the first time. If 5.6a lands first, Patch 3 will cause its previously-untouched `EngagementQuery` body code path to start firing — a positive coverage event, not a regression, but the 5.6a author should be looped in.

5. **Test name collisions.** The new tests in Patch 1 and Patch 2 are placed in `expr.rs::tests`; ensure neither name collides with what 5.5d adds (suggested prefix: `lowers_namespace_field_*` and `inlines_lazy_view_*`).

## Patch 1: NamespaceField → ConfigConst

### Files

- `crates/dsl_compiler/src/cg/lower/expr.rs` — `LoweringCtx` field; helper; `IrExpr::NamespaceField` arm in `lower_expr`.
- `crates/dsl_compiler/src/cg/lower/driver.rs` — new `populate_config_consts` Phase 1 walker; entry-point wiring.
- `crates/dsl_compiler/src/cg/lower/error.rs` — new `UnknownConfigField` variant; `Display` arm.

### Anchor 1.1: `LoweringCtx` field

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`
**Anchor:** the struct literal block `LoweringCtx { ... }` at lines 54–140; specifically insert the new field directly after `pub view_signatures: HashMap<ViewId, (Vec<CgTy>, CgTy)>,` at line 65.

**Add field:**

```rust
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
```

**Update `LoweringCtx::new`** at line 150 — add `config_const_ids: HashMap::new(),` directly after the `view_signatures: HashMap::new(),` line at line 154.

**Update imports** at lines 30–32 — `crate::cg::data_handle` import should add `ConfigConstId` (already used downstream — verify the symbol is present in `data_handle.rs:54`):

```rust
use crate::cg::data_handle::{
    AgentFieldId, AgentFieldTy, AgentRef, CgExprId, ConfigConstId, DataHandle, RngPurpose, ViewId,
};
```

### Anchor 1.2: `register_config_const` helper on `LoweringCtx`

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`
**Anchor:** the impl block of helpers at lines 142–268; insert after `register_view_signature` (line 260–267).

**Add helper:**

```rust
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
```

### Anchor 1.3: `IrExpr::NamespaceField` arm in `lower_expr`

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`
**Anchor:** lines 362–371 in `lower_expr`. Currently:

```rust
IrExpr::NamespaceField { ns, field, .. } => {
    // No expression-level NamespaceField lowering is wired in
    // Task 2.1. The op-level driver will lift `config.<…>`
    // through a `ConfigConstId` later.
    Err(LoweringError::UnsupportedNamespaceField {
        ns: *ns,
        field: field.clone(),
        span,
    })
}
```

**Replace with:**

```rust
IrExpr::NamespaceField { ns, field, .. } => {
    // `config.<block>.<field>` (the only NamespaceField shape the
    // resolver produces today — see `dsl_ast::ir` line 944) lowers
    // to a typed `Read(ConfigConst { id })`. Other namespaces have
    // no `NamespaceField` data path today; they keep the typed
    // deferral. Task 5.5c.
    if *ns == NamespaceId::Config {
        match ctx.config_const_ids.get(&(*ns, field.clone())) {
            Some(id) => add(ctx, CgExpr::Read(DataHandle::ConfigConst { id: *id }), span),
            None => Err(LoweringError::UnknownConfigField {
                ns: *ns,
                field: field.clone(),
                span,
            }),
        }
    } else {
        Err(LoweringError::UnsupportedNamespaceField {
            ns: *ns,
            field: field.clone(),
            span,
        })
    }
}
```

### Anchor 1.4: New error variant

**File:** `crates/dsl_compiler/src/cg/lower/error.rs`
**Anchor:** insert the new variant directly after `UnsupportedNamespaceField` at lines 238–245.

```rust
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
```

**Add a `Display` arm** in the `impl fmt::Display for LoweringError` block. Find the existing `UnsupportedNamespaceField` arm at line 871 in error.rs and add a parallel arm immediately after it:

```rust
LoweringError::UnknownConfigField { ns, field, span } => write!(
    f,
    "[{}..{}] config field `{}.{}` has no ConfigConstId — driver registry missed it",
    span.start, span.end, ns.snake(), field
),
```

(Match the formatting / phrasing of the surrounding `UnsupportedNamespaceField` Display arm at line 871.)

### Anchor 1.5: Driver `populate_config_consts`

**File:** `crates/dsl_compiler/src/cg/lower/driver.rs`
**Anchor:** insert as a new helper directly after `populate_views` ends at line 434.

```rust
/// Allocate one [`ConfigConstId`] per (block, field) pair across
/// every [`ConfigIR`] in source order, register each into
/// `ctx.config_const_ids` keyed on
/// `(NamespaceId::Config, "<block>.<field>")`, and intern the
/// human-readable name on the builder for diagnostics +
/// pretty-printing. The id allocation is deterministic — the
/// flat numeric `i` reflects walk order.
///
/// A duplicate registration (same (block, field) pair across two
/// `ConfigIR`s) is a driver-side defect; surfaced as a typed
/// [`LoweringError::DuplicateConfigConstInRegistry`] diagnostic
/// with last-write-wins semantics.
fn populate_config_consts(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) {
    let mut next_id: u32 = 0;
    for cfg in &comp.configs {
        for fld in &cfg.fields {
            let id = ConfigConstId(next_id);
            next_id += 1;
            let key = format!("{}.{}", cfg.name, fld.name);
            if let Some(prior) =
                ctx.register_config_const(NamespaceId::Config, key.clone(), id)
            {
                diagnostics.push(LoweringError::DuplicateConfigConstInRegistry {
                    key: key.clone(),
                    prior_id: prior,
                    new_id: id,
                });
            }
            if let Err(e) = ctx.builder.intern_config_const_name(id, key) {
                diagnostics.push(LoweringError::BuilderRejected {
                    error: e,
                    span: fld.span,
                });
            }
        }
    }
}
```

**Add the `DuplicateConfigConstInRegistry` variant** to `LoweringError` (in error.rs, mirror the shape of `DuplicateViewInRegistry` already present — search for it; insert next to it):

```rust
/// Two `(block, field)` pairs in `Compilation::configs` resolved
/// to the same registry key. Driver-side defect; last-write-wins
/// semantics in place. Mirrors
/// [`Self::DuplicateViewInRegistry`].
DuplicateConfigConstInRegistry {
    key: String,
    prior_id: crate::cg::data_handle::ConfigConstId,
    new_id: crate::cg::data_handle::ConfigConstId,
},
```

(Add a parallel `Display` arm; format the way `DuplicateViewInRegistry`'s arm formats.)

**Wire into the driver entry point** at lines 168–174. Today the Phase 1 sequence is:

```rust
let event_rings = populate_event_kinds(comp, &mut ctx, &mut diagnostics);
populate_variants_from_enums(comp, &mut ctx, &mut diagnostics);
populate_actions(comp, &mut ctx, &mut diagnostics);
populate_views(comp, &mut ctx, &mut diagnostics);
```

**Insert** the call:

```rust
populate_config_consts(comp, &mut ctx, &mut diagnostics);
```

between `populate_actions` and `populate_views` (alphabetical / consistent with the Phase 1 docstring's ordering — but order is irrelevant for correctness; the slot is open).

**Imports to add at the top of `driver.rs`:**

- `dsl_ast::ir::NamespaceId` — append to the existing `dsl_ast::ir::{...}` import at lines 100–102.
- `crate::cg::data_handle::ConfigConstId` — append to the existing `crate::cg::data_handle::{...}` import at line 104.

### Tests to add (in `expr.rs::tests`)

1. **`lowers_namespace_field_to_config_const`** — happy path. Construct an `IrExpr::NamespaceField { ns: NamespaceId::Config, field: "combat.attack_range".into(), ty: IrType::F32 }`. Pre-register `(NamespaceId::Config, "combat.attack_range") → ConfigConstId(0)` on the ctx. Lower; assert the pretty form starts with `(read config_const.` (the existing `DataHandle::fmt_with` uses `write_named_id(f, names.name_for(IdKind::ConfigConst, id.0), id.0)` per `data_handle.rs:826-829`, so an unnamed-resolver pretty form is `(read config_const.#0)`).

2. **`unknown_namespace_field_returns_typed_error`** — fallback for the Config namespace with no registry entry. Lower without registering; assert `LoweringError::UnknownConfigField { field, .. }` with `field == "combat.attack_range"`.

3. **`non_config_namespace_field_keeps_unsupported_deferral`** — preserves Task 2.1's behaviour for non-Config namespaces. Lower an `IrExpr::NamespaceField { ns: NamespaceId::World, field: "tick".into(), .. }`; assert `LoweringError::UnsupportedNamespaceField { .. }` (the existing test `namespace_field_typed_error` at expr.rs:2179–2190 already covers this — verify it still passes after the patch and rename to `unsupported_non_config_namespace_field_typed_error` for clarity, or leave as-is).

4. **(Driver-level)** in `driver.rs::tests`: **`populate_config_consts_allocates_per_block_field_in_source_order`**. Construct a `Compilation` with two `ConfigIR` blocks (e.g., `combat` with `attack_range`, `aggro_range`; `movement` with `move_speed_mps`). Run `populate_config_consts`. Assert `ctx.config_const_ids` contains four entries with ids 0..3 in the documented order, and the interner has matching names.

5. **(Driver-level)** **`populate_config_consts_flags_duplicate_key`**. Pre-seed the registry with a colliding entry for `(NamespaceId::Config, "combat.attack_range")` → `ConfigConstId(99)`, then run `populate_config_consts` against a one-block-one-field Compilation. Assert one `DuplicateConfigConstInRegistry { key: "combat.attack_range", prior_id: ConfigConstId(99), new_id: ConfigConstId(0) }` diagnostic.

## Patch 2: Lazy view inlining at call sites

### Files

- `crates/dsl_compiler/src/cg/lower/expr.rs` — `LoweringCtx` field; `lower_view_call` rewrite; substitution helper.
- `crates/dsl_compiler/src/cg/lower/view.rs` — capture lazy-view bodies into the new `ctx` field on the `(ViewKind::Lazy, ViewBodyIR::Expr)` arm; populate materialized view signatures on the materialized arm.
- `crates/dsl_compiler/src/cg/lower/driver.rs` — module-level "Limitations" docstring update (the lazy-inline-and-signature-registration limitation retires).

### Anchor 2.1: `LazyViewSnapshot` + `LoweringCtx::lazy_view_bodies`

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`
**Anchor:** insert the new struct + field directly after the `LoweringCtx` field block (near the new `config_const_ids` field from Patch 1).

```rust
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
```

**Add the field on `LoweringCtx`** (insert directly after `config_const_ids`):

```rust
/// `ViewId` → captured lazy-view body for at-call-site inlining.
/// Populated by [`super::view::lower_view`]'s lazy arm in Phase 2;
/// consumed by [`lower_view_call`]. A view absent from this map
/// is materialized — the call lowers through
/// `BuiltinId::ViewCall { view }` as before, with the type checker
/// resolving against `ctx.view_signatures`. Task 5.5c.
pub lazy_view_bodies: HashMap<ViewId, LazyViewSnapshot>,
```

**Update `LoweringCtx::new`** to initialise `lazy_view_bodies: HashMap::new(),`.

### Anchor 2.2: `register_lazy_view_body` helper on `LoweringCtx`

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`
**Anchor:** insert after the new `register_config_const` helper (Patch 1 anchor 1.2).

```rust
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
```

### Anchor 2.3: `lower_view_call` rewrite

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`
**Anchor:** lines 1117–1150 — `lower_view_call`. Currently always lowers to `BuiltinId::ViewCall { view: view_id }`.

**Replace with:**

```rust
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

    // Materialized-view path: lower as a typed `BuiltinId::ViewCall`.
    let mut arg_ids = Vec::with_capacity(args.len());
    for a in args {
        let id = lower_expr(&a.value, ctx)?;
        arg_ids.push(id);
    }
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
```

### Anchor 2.4: `substitute_locals` helper

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`
**Anchor:** insert as a module-private helper directly after `lower_view_call` (after the new closing brace).

```rust
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
            args: args.iter().map(|a| IrCallArg {
                name: a.name.clone(),
                value: substitute_locals(&a.value, binders),
                span: a.span,
            }).collect(),
        },
        // Pass-through for shapes that carry no `IrExprNode` children we
        // need to descend through. The set is intentionally exhaustive
        // over `IrExpr` (no wildcard) — adding a new variant to `IrExpr`
        // forces a substitution decision here.
        IrExpr::LitBool(_) | IrExpr::LitInt(_) | IrExpr::LitFloat(_)
            | IrExpr::LitString(_) | IrExpr::Event(_) | IrExpr::Entity(_)
            | IrExpr::View(_) | IrExpr::Verb(_) | IrExpr::Namespace(_)
            | IrExpr::NamespaceField { .. } | IrExpr::EnumVariant { .. }
            | IrExpr::AbilityHint | IrExpr::AbilityHintLit(_)
            | IrExpr::AbilityRange | IrExpr::AbilityOnCooldown(_)
            | IrExpr::AbilityTag { .. } | IrExpr::Raw(_)
            | IrExpr::BeliefsAccessor { .. } | IrExpr::BeliefsConfidence { .. }
            | IrExpr::BeliefsView { .. } => expr.kind.clone(),
        // Forms that *could* carry locals but the lazy-view body
        // surface in real .sim files doesn't yet exercise — keep
        // them recursing so the walk is robust if the AST shape
        // expands. Each arm reconstructs the variant with
        // substituted children.
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
        // miscompiled. Document as a known limitation.
        IrExpr::Quantifier { .. } | IrExpr::Fold { .. } | IrExpr::Match { .. }
            | IrExpr::StructLit { .. } | IrExpr::Ctor { .. }
            | IrExpr::PerUnit { .. } => expr.kind.clone(),
    };
    IrExprNode { kind, span }
}
```

(NOTE: this is a long arm-match; the apply agent should sanity-check that every `IrExpr` variant at HEAD `b50a6744` is covered. Run `cargo check` after applying — the compiler will flag any missed variant.)

### Anchor 2.5: Lazy-body capture in `lower_view`

**File:** `crates/dsl_compiler/src/cg/lower/view.rs`
**Anchor:** the `(ViewKind::Lazy, ViewBodyIR::Expr(_))` arm in `lower_view`, lines 243–267.

**Today** the arm is:

```rust
(ViewKind::Lazy, ViewBodyIR::Expr(_)) => {
    // [docstring omitted]
    if !handler_resolutions.is_empty() {
        return Err(LoweringError::ViewHandlerResolutionLengthMismatch { ... });
    }
    intern_view_name(view_id, ir, ctx)?;
    Ok(Vec::new())
}
```

**Replace with:**

```rust
(ViewKind::Lazy, ViewBodyIR::Expr(body)) => {
    // [keep existing docstring, expand to mention 5.5c capture]
    if !handler_resolutions.is_empty() {
        return Err(LoweringError::ViewHandlerResolutionLengthMismatch {
            view: view_id,
            expected: 0,
            got: handler_resolutions.len(),
            span: ir.span,
        });
    }
    intern_view_name(view_id, ir, ctx)?;

    // Task 5.5c: capture the body for at-call-site inlining.
    // `param_locals` carries the i-th positional param's LocalRef so
    // the substitute_locals walk in `lower_view_call` can replace
    // each param-local reference with the matching call-site arg.
    let snapshot = super::expr::LazyViewSnapshot {
        param_locals: ir.params.iter().map(|p| p.local).collect(),
        body: body.clone(),
    };
    if let Some(_prior) = ctx.register_lazy_view_body(view_id, snapshot) {
        // Driver-side defect — the same view_id was registered twice.
        // Last-write-wins; surface as a typed diagnostic. No
        // dedicated variant — reuse `BuilderRejected` would obscure
        // the cause, so add a variant in error.rs:
        return Err(LoweringError::DuplicateLazyViewBodyRegistration {
            view: view_id,
            span: ir.span,
        });
    }
    Ok(Vec::new())
}
```

**Add `DuplicateLazyViewBodyRegistration` variant** to `LoweringError` (in error.rs, near the other `Duplicate*` variants). Display arm: `"view #{} lazy body registered twice (driver defect)"`.

### Anchor 2.6: Materialized view signature population

**File:** `crates/dsl_compiler/src/cg/lower/view.rs`
**Anchor:** the `(ViewKind::Materialized(hint), ViewBodyIR::Fold { .. })` arm in `lower_view`, lines 269–292.

After `intern_view_name(view_id, ir, ctx)?;` at line 283, **insert**:

```rust
// Task 5.5c: register the view's typed signature for
// `BuiltinSignature::ViewCall` resolution at the type-check
// layer. Without this, expression-level `IrExpr::ViewCall` to
// a materialized view fails type-check with
// `TypeError::ViewSignatureUnresolved`.
let arg_tys: Vec<CgTy> = ir
    .params
    .iter()
    .map(|p| ir_type_to_cg_ty(&p.ty))
    .collect();
let result_ty = ir_type_to_cg_ty(&ir.return_ty);
ctx.register_view_signature(view_id, arg_tys, result_ty);
```

**Add a free helper `ir_type_to_cg_ty`** in `view.rs` (or in `expr.rs` if cross-pass reuse is desired — recommended location: a new top-level helper in `expr.rs` since lazy inlining could in theory use it later for diagnostics):

```rust
/// Map a `dsl_ast::ir::IrType` to its `CgTy` representation,
/// using the same primitive-set the existing
/// `_agent_field_ty_invariant` documents for agent fields. Used
/// by view-signature population (Task 5.5c). Returns
/// `CgTy::ViewKey { view }` for unmappable types — those are
/// view-typed values whose CG representation needs the view id;
/// the caller is responsible for never invoking this on a return
/// type that doesn't resolve to a primitive.
fn ir_type_to_cg_ty(ty: &dsl_ast::ir::IrType) -> CgTy {
    use dsl_ast::ir::IrType as T;
    match ty {
        T::Bool => CgTy::Bool,
        T::U8 | T::U16 | T::U32 => CgTy::U32,
        T::I8 | T::I16 | T::I32 => CgTy::I32,
        T::F32 => CgTy::F32,
        T::Vec3 => CgTy::Vec3F32,
        T::AgentId => CgTy::AgentId,
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
```

(Match the narrowing behaviour of `_agent_field_ty_invariant` at expr.rs:1252–1264 where applicable. The unsupported-fallback behaviour is intentional: the materialized views in `assets/sim/views.sim` only return `f32` and `i32`/`AgentId`; if a future view returns a more exotic type, the type-check failure will surface the mismatch downstream — better than a panic at view registration.)

### Anchor 2.7: Driver "Limitations" docstring update

**File:** `crates/dsl_compiler/src/cg/lower/driver.rs`
**Anchor:** the module docstring's "Limitations" section, lines 84–96.

**Remove** the bullet at lines 87–96 ("No view-call signature registration..."). The lazy-inlining + materialized-signature population work together to retire it.

(Optional: replace with a 1-line note that lazy view inlining now happens in `lower_view_call` consulting `ctx.lazy_view_bodies`.)

### Tests to add

In `expr.rs::tests` (lazy view inlining):

1. **`inlines_lazy_view_at_call_site`** — happy path. Register a lazy view body of the form `IrExpr::Local(LocalRef(0), "a")` (the view's first param) into `ctx.lazy_view_bodies`. Lower an `IrExpr::ViewCall(ast_ref, vec![arg(node(IrExpr::LitBool(true)))])`. Assert the resulting `CgExprId` resolves to a `(lit true)` (the substituted call arg) — NOT a `(builtin.view_call.#0 ...)`.

2. **`inlines_lazy_view_with_field_access`** — model `is_hostile(self, target)`. Register a lazy view body of form `agents.is_hostile_to(<a>, <b>)` where `<a>` and `<b>` are the param locals. Pass `(self, self.engaged_with)` as the arguments (need ctx to be a target-binding-aware context for the latter; for the test, use two literal `LitInt`s instead). Assert the lowered form contains the namespace-call structure with the substituted args.

3. **`materialized_view_call_uses_builtin_view_call`** — verifies the materialized path is intact. Register a materialized view's signature via `register_view_signature`; do *not* register a body. Lower a call; assert pretty form starts with `(builtin.view_call.`.

4. **`lazy_view_arity_mismatch_returns_typed_error`** — register a 2-param lazy body; call with 1 arg. Assert `LoweringError::ViewCallArityMismatch { view, expected: 2, got: 1, .. }`.

5. **(View-pass-level)** in `view.rs::tests`: extend `lowers_lazy_view_to_zero_ops_with_name_interned` (line 854) to ALSO assert `ctx.lazy_view_bodies.len() == 1` and that the captured snapshot's `param_locals` matches the view's params and `body` round-trips.

6. **(View-pass-level)** new test `materialized_view_registers_signature` — synthesise a materialized view; lower; assert `ctx.view_signatures.get(&view_id).is_some()`.

**New error variants needed** (already mentioned above):

- `LoweringError::ViewCallArityMismatch { view: ViewId, expected: usize, got: usize, span }`.
- `LoweringError::DuplicateLazyViewBodyRegistration { view: ViewId, span }`.

Both carry a `Display` arm matching the surrounding family's phrasing.

## Patch 3: EngagementQuery driver routing

### Files

- `crates/dsl_compiler/src/cg/lower/driver.rs` — `mask_spatial_kind` upgrade; new helper(s) for predicate scanning.

### Anchor 3.1: `mask_spatial_kind` upgrade

**File:** `crates/dsl_compiler/src/cg/lower/driver.rs`
**Anchor:** lines 463–473. Currently:

```rust
fn mask_spatial_kind(mask: &MaskIR) -> Option<SpatialQueryKind> {
    if mask.candidate_source.is_some() {
        Some(SpatialQueryKind::KinQuery)
    } else {
        None
    }
}
```

**Replace with:**

```rust
fn mask_spatial_kind(mask: &MaskIR) -> Option<SpatialQueryKind> {
    if mask.candidate_source.is_none() {
        return None;
    }
    if predicate_uses_engagement_relationship(&mask.predicate) {
        Some(SpatialQueryKind::EngagementQuery)
    } else {
        Some(SpatialQueryKind::KinQuery)
    }
}

/// Scan `expr` for any access pattern indicating an
/// engagement-target relationship in the candidate filter:
///
/// - `agents.is_hostile_to(_, _)` (stdlib hostility check).
/// - `agents.engaged_with(_)` (engagement read).
/// - A call to a view named `is_hostile` (the canonical lazy
///   view in `assets/sim/views.sim`; resolves to a
///   `IrExpr::ViewCall` whose AST `ViewRef` indexes into
///   `Compilation::views` — but since this scan runs over the
///   raw AST predicate, we don't have that lookup available
///   here; routing through the CALL site's *unresolved* form
///   isn't viable post-resolve, so we use `ViewCall(_, _)` as
///   a wildcard signal that the predicate calls *any* view).
///
/// The third bullet is intentionally conservative: every
/// from-bearing mask in `assets/sim/masks.sim` that calls a
/// view does so to filter for hostility (Attack uses
/// `is_hostile`; MoveToward uses no view). If a future mask
/// uses a non-hostility view in its predicate (e.g.,
/// `view::slow_factor(target) > 100`), this routing would
/// incorrectly pick `EngagementQuery`. Refining the test to
/// resolve the called view and gate on its name is a follow-up
/// when a counterexample lands; today it conservatively widens
/// from `KinQuery` to `EngagementQuery` (the latter walks
/// hostile neighbours, which is a strict superset of the
/// in-use set today).
///
/// Alternative considered: thread `Compilation::views` through
/// `mask_spatial_kind` so the scan can resolve a ViewCall to
/// its declared name and gate on `name == "is_hostile"`. This
/// is the morally-correct shape; punted because it'd extend
/// `mask_spatial_kind`'s signature, which `lower_all_masks`
/// would also need to thread, which adds 4-5 lines of plumbing
/// for the same routing decision. Recommend: revisit when the
/// counterexample arrives.
fn predicate_uses_engagement_relationship(expr: &IrExprNode) -> bool {
    use dsl_ast::ir::{IrExpr, NamespaceId};
    match &expr.kind {
        IrExpr::NamespaceCall { ns: NamespaceId::Agents, method, args }
            if method == "is_hostile_to" || method == "engaged_with" =>
        {
            // Defensive: still walk args in case they nest the
            // pattern (e.g., `agents.is_hostile_to(self,
            // agents.engaged_with(self))`), though the OR with
            // the outer match makes this redundant for today's
            // masks.
            let _ = args.iter().any(|a| predicate_uses_engagement_relationship(&a.value));
            true
        }
        IrExpr::ViewCall(_, args) => {
            // Conservative: treat any ViewCall in the predicate
            // as an engagement-flavoured filter. See doc above.
            // Still recurse into args so a nested non-view
            // engagement signal (e.g., the `agents.engaged_with`
            // case above) gets caught in compositions.
            let _ = args.iter().any(|a| predicate_uses_engagement_relationship(&a.value));
            true
        }
        // Recurse into children for every shape that carries
        // sub-expressions. Closed-set match per the project
        // convention.
        IrExpr::Field { base, .. } => predicate_uses_engagement_relationship(base),
        IrExpr::Binary(_, l, r) => {
            predicate_uses_engagement_relationship(l)
                || predicate_uses_engagement_relationship(r)
        }
        IrExpr::Unary(_, a) => predicate_uses_engagement_relationship(a),
        IrExpr::If { cond, then_expr, else_expr } => {
            predicate_uses_engagement_relationship(cond)
                || predicate_uses_engagement_relationship(then_expr)
                || else_expr
                    .as_ref()
                    .map(|e| predicate_uses_engagement_relationship(e))
                    .unwrap_or(false)
        }
        IrExpr::BuiltinCall(_, args)
        | IrExpr::NamespaceCall { args, .. }
        | IrExpr::VerbCall(_, args)
        | IrExpr::UnresolvedCall(_, args) => args
            .iter()
            .any(|a| predicate_uses_engagement_relationship(&a.value)),
        IrExpr::Index(l, r) | IrExpr::In(l, r) | IrExpr::Contains(l, r) => {
            predicate_uses_engagement_relationship(l)
                || predicate_uses_engagement_relationship(r)
        }
        IrExpr::List(items) | IrExpr::Tuple(items) => items
            .iter()
            .any(|e| predicate_uses_engagement_relationship(e)),
        // Leaves and binder-introducing forms — no engagement
        // signal can hide in them given today's predicate
        // surface (the resolver rejects quantifiers / folds /
        // matches in mask predicates).
        _ => false,
    }
}
```

### Anchor 3.2: Imports

**File:** `crates/dsl_compiler/src/cg/lower/driver.rs`
**Anchor:** the existing `dsl_ast::ir` import block at lines 100–102. Add `IrExpr, IrExprNode, NamespaceId` (the closure already uses NamespaceId::Config from Patch 1; merging is fine).

### Anchor 3.3: Update module-level "Limitations" docstring

**File:** `crates/dsl_compiler/src/cg/lower/driver.rs`
**Anchor:** lines 71–81 (the "Per-mask spatial query selection" bullet).

**Replace** with a note documenting the new behaviour and the conservative-widening caveat (see the doc on `predicate_uses_engagement_relationship` above for content). One concrete suggestion:

```
//! - **Per-mask spatial query selection.** The driver routes
//!   from-bearing masks to `EngagementQuery` when their predicate
//!   references engagement-flavoured access patterns
//!   (`agents.is_hostile_to`, `agents.engaged_with`, or any
//!   `IrExpr::ViewCall` — conservative widening; see
//!   `predicate_uses_engagement_relationship`). All other
//!   from-bearing masks route to `KinQuery`. Refining the
//!   ViewCall test to gate on the called view's name is a
//!   follow-up — punted because no current counterexample
//!   exists in `assets/sim/masks.sim`.
```

### Tests to add (in `driver.rs::tests`)

1. **`mask_spatial_kind_routes_engagement_for_is_hostile_to`** — synthesise a `MaskIR` whose `candidate_source` is `Some(...)` and whose `predicate` is `IrExpr::NamespaceCall { ns: Agents, method: "is_hostile_to", args: [...] }`. Assert `mask_spatial_kind(&mask) == Some(SpatialQueryKind::EngagementQuery)`.

2. **`mask_spatial_kind_routes_engagement_for_view_call`** — predicate is `IrExpr::ViewCall(AstViewRef(0), vec![])`. Assert routes to `EngagementQuery`.

3. **`mask_spatial_kind_routes_engagement_for_engaged_with`** — predicate is `IrExpr::NamespaceCall { ns: Agents, method: "engaged_with", args: [...] }`. Assert routes to `EngagementQuery`.

4. **`mask_spatial_kind_routes_kin_for_alive_only`** — predicate is `IrExpr::NamespaceCall { ns: Agents, method: "alive", args: [...] }` (the MoveToward shape). Assert routes to `KinQuery`.

5. **`mask_spatial_kind_returns_none_when_no_candidate_source`** — `candidate_source: None`. Assert `None` (preserves Task 2.1 behaviour).

6. **`mask_spatial_kind_walks_into_binary_branches`** — predicate is `agents.alive(target) && agents.is_hostile_to(self, target)`. Assert `EngagementQuery` (the engagement signal is on the right of the `&&`).

## Test plan summary

- **Patch 1 (Config):** 5 tests
  - `lowers_namespace_field_to_config_const` (expr.rs)
  - `unknown_namespace_field_returns_typed_error` (expr.rs)
  - `unsupported_non_config_namespace_field_typed_error` (expr.rs — preserve / rename existing)
  - `populate_config_consts_allocates_per_block_field_in_source_order` (driver.rs)
  - `populate_config_consts_flags_duplicate_key` (driver.rs)
- **Patch 2 (Lazy view inlining):** 6 tests
  - `inlines_lazy_view_at_call_site` (expr.rs)
  - `inlines_lazy_view_with_field_access` (expr.rs)
  - `materialized_view_call_uses_builtin_view_call` (expr.rs)
  - `lazy_view_arity_mismatch_returns_typed_error` (expr.rs)
  - `lowers_lazy_view_captures_body_snapshot` (view.rs — extend or new)
  - `materialized_view_registers_signature` (view.rs)
- **Patch 3 (EngagementQuery routing):** 6 tests
  - `mask_spatial_kind_routes_engagement_for_is_hostile_to`
  - `mask_spatial_kind_routes_engagement_for_view_call`
  - `mask_spatial_kind_routes_engagement_for_engaged_with`
  - `mask_spatial_kind_routes_kin_for_alive_only`
  - `mask_spatial_kind_returns_none_when_no_candidate_source`
  - `mask_spatial_kind_walks_into_binary_branches`

Pre-flight check: 755 lib + 10 xtask tests pass at HEAD. After applying, `cargo test -p dsl_compiler` should net **+17 lib tests** (Patch 1: 5; Patch 2: 6; Patch 3: 6) — total expected 772 lib + 10 xtask if no other test impact. Patch 2's materialized-signature population is the only patch that COULD shift another test's outcome — if any pre-existing test on a `.sim` integration fixture currently fails because of `TypeError::ViewSignatureUnresolved` and was carried as a deferral diagnostic, that diagnostic should disappear post-patch.

## Apply notes

### Sequencing constraints

- **Patch 1 is independent.** Apply first, run `cargo test -p dsl_compiler`, confirm green.
- **Patch 2 depends on no other patch but lands cleaner before Patch 3** (Patch 3's predicate scanner may eventually want to *resolve* lazy view calls to gate on their name — Patch 2 lays down the registry that future refinement could use). Apply second.
- **Patch 3 is independent of Patch 1 + 2** at the code level (it touches only `driver.rs::mask_spatial_kind`), but is documented last because it's the most behaviourally-narrow change. Apply third.

### Per-patch verification

After each patch:

1. `cargo build -p dsl_compiler` — must succeed (catches missed match arms in `substitute_locals` and `predicate_uses_engagement_relationship`).
2. `cargo test -p dsl_compiler` — must show the new tests passing, and net `lib + xtask` count match the expected delta.
3. After all three patches, run the full `cargo test` to verify integration tests on `.sim` fixtures (the `compile-dsl` end-to-end tests) don't regress; expect a *reduction* in deferral diagnostics on the `assets/sim/*.sim` integration fixture (per the plan's "45 typed deferrals → ?" headline at line 725).

### Known scope risks

1. **`substitute_locals` exhaustiveness.** The walk over `IrExpr` is long; the apply agent should let `cargo check` flag any new variant introduced upstream. The closed-set match (no wildcard) is intentional — wildcards would silently swallow new variants.

2. **`ir_type_to_cg_ty` partial mapping.** `IrType` has 30+ variants; only the primitives needed by today's view returns are mapped. The fallback to `CgTy::U32` is *not* silent: any view whose params/return resolve to an unsupported type will eventually fail type-check downstream when the registered signature is consulted. If that turns out to be a real issue (e.g., a view returning `Vec3` which doesn't exist today but is plausible), promote `ir_type_to_cg_ty` to return `Result<CgTy, LoweringError>` and surface a typed `UnsupportedViewParamType` variant.

3. **Patch 3's conservative widening on `ViewCall`.** Documented above. If a counterexample mask appears (a from-bearing mask whose predicate calls a non-hostility view), Patch 3 will mis-route it to `EngagementQuery`. The mitigation is to thread `Compilation::views` into `mask_spatial_kind` and gate on the resolved view's `name`. Today there is no such mask in `assets/sim/masks.sim`; if one lands later, the per-mask test suite will catch the routing change as a behavioural divergence.

### No scope reductions discovered

The audit's framing (3 deferred shapes) holds: each of the three is mechanically tractable without expanding the surface beyond the specified files. The work is primarily in `expr.rs` (Patch 1 + 2) and `driver.rs` (Patch 1 + 3); `view.rs` gains a single body-capture call; `error.rs` gains 3-4 new variants. Total estimated touched LoC: ~280 lines added, ~20 lines replaced, across 4 files plus tests.
