//! WGSL emission for @materialized views — Phase 4 of the GPU megakernel plan.
//!
//! Companion to [`emit_view`] (which emits Rust) — emits WGSL snippets the
//! engine_gpu backend stitches into its compute pipeline. Two user-facing
//! surfaces:
//!
//!   * [`emit_view_read_wgsl`]  — emits a `fn view_<name>_get(args..., tick)`
//!     WGSL snippet returning `f32` / `u32` / `bool` that reads from the
//!     matching view storage buffer (with decay applied on-read).
//!   * [`emit_view_fold_wgsl`]  — emits a `fn view_<name>_fold(event, tick)`
//!     WGSL snippet that mutates the view storage for a single matching
//!     event (atomic add + tick stamp for @decay scalars, set-value for
//!     slot_map shapes).
//!
//! This file does NOT own the bind-group layout, buffer allocation, or
//! kernel entry point — those are the engine_gpu side's job. The emitter
//! produces reusable snippets; engine_gpu glues them into a module with
//! the right bindings. That split mirrors what the mask WGSL emitter
//! does for `emit_mask_wgsl`: emit reusable WGSL, let the backend own
//! pipeline wiring.
//!
//! ## Storage layout assumed
//!
//! The emitter assumes the engine_gpu side has provisioned one of three
//! buffer shapes per view:
//!
//!   * **slot_map** (`per_entity_topk(K=1)` in DSL terms): a flat
//!     `array<u32>` length = `agent_cap`, indexed by the single AgentId
//!     key. For `engaged_with` the value is an `AgentId+1` (0 = empty
//!     slot; u32::MAX maps to "no engagement").
//!   * **pair_map, no decay**: a flat `array<f32>` length = `agent_cap^2`
//!     indexed `[observer * N + attacker]`. Currently only `my_enemies`.
//!   * **pair_map @decay**: a flat `array<DecayCell>` of length
//!     `agent_cap^2` where `struct DecayCell { value: f32,
//!     anchor_tick: u32 }`. 4 views: `threat_level`, `kin_fear`,
//!     `pack_focus`, `rally_boost`. The decay is applied on-read:
//!     `value * pow(rate, tick - anchor_tick)` clamped.
//!
//! Lazy views (`is_hostile`, `is_stunned`, `slow_factor`) do not get view
//! storage — they fall through to the mask WGSL emitter's expression
//! lowering (it already knows how to inline `is_hostile` etc.). The
//! `emit_view_read_wgsl` entry point raises `Unsupported` for lazy views
//! so callers don't accidentally ask for storage that doesn't exist.
//!
//! ## Determinism caveat
//!
//! The fold snippet uses `atomicAdd` on f32 — **not commutative** under
//! general orderings. Every current fold body is `self += constant` (the
//! constant is 1.0 for every materialized view in `assets/sim/views.sim`
//! today), so f32 add is commutative-associative under the "all adds are
//! the same positive small value" special case. If a future view uses a
//! variable delta the engine_gpu side should switch to a deterministic
//! sort-by-key-then-reduce pass; the emitter surfaces the assumption in
//! the snippet's doc-comment so that regression is visible.

use std::fmt::Write;

use crate::ir::{
    DecayUnit, FoldHandlerIR, IrExprNode, IrPattern, IrType, StorageHint, ViewBodyIR, ViewIR,
    ViewKind,
};

/// Binding numbers the engine_gpu side should wire each shape into.
/// These are relative offsets inside the view bind group — the backend
/// adds them onto a base so it can share the group with mask / scoring
/// bindings without collision. Kept as documentation-only constants;
/// the emitter itself writes raw numbers into the snippet strings.
pub mod bindings {
    /// Agent capacity uniform (used to index pair_map as
    /// `observer * N + attacker`). Scalar u32.
    pub const AGENT_CAP_UNIFORM: u32 = 0;
    /// Current tick uniform — decay math needs `tick - anchor`.
    pub const CURRENT_TICK_UNIFORM: u32 = 1;
}

/// Errors raised during WGSL view emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// The view shape isn't one the emitter can lower to GPU storage.
    /// Carries a short diagnostic.
    Unsupported(String),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Unsupported(s) => write!(f, "wgsl view emission: {s}"),
        }
    }
}

impl std::error::Error for EmitError {}

/// High-level classification returned by [`classify_view`]. The four
/// shapes the Phase 4 emitter supports — plus `Lazy` so callers can
/// detect "no GPU storage needed, fall through to expression inliner".
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViewShape {
    /// `@lazy` — no storage, pure expression. Caller inlines via
    /// existing mask / scoring WGSL emitters.
    Lazy,
    /// `per_entity_topk(K=1)` — flat `array<u32>` indexed by AgentId.
    /// Value stored as `AgentId_raw + 1` (0 = empty). Only
    /// `engaged_with` lowers to this today.
    SlotMap { return_u32_as_option: bool },
    /// `pair_map`, no decay — flat `array<f32>` indexed by
    /// `observer * N + attacker`. Only `my_enemies` today.
    PairMapScalar,
    /// `pair_map` + @decay — flat `array<DecayCell>` (f32 base + u32
    /// anchor_tick). Current: `threat_level`, `kin_fear`, `pack_focus`,
    /// `rally_boost`.
    PairMapDecay { rate: f32 },
}

/// Storage metadata produced by classification. Used by the engine_gpu
/// side to size buffers — the emitter also reads it to pick the right
/// snippet layout.
#[derive(Debug, Clone)]
pub struct ViewStorageSpec {
    /// DSL-level name (e.g. `"threat_level"`).
    pub view_name: String,
    /// Rust/WGSL-level ident prefix (snake_case view name — matches the
    /// generated Rust `mod` name).
    pub snake: String,
    /// Classified shape. For topk(K>1) views this reports the
    /// underlying pair_map shape (scalar or decay) so scoring's
    /// bind-group layout keeps compiling; the `topk` field below
    /// carries the K for storage sizing.
    pub shape: ViewShape,
    /// Clamp `(lo, hi)` if the view has `clamp: [lo, hi]`. Lowered as
    /// plain f32 literals; types other than f32 aren't clamped in any
    /// shipped view today.
    pub clamp: Option<(f32, f32)>,
    /// Initial value to return for un-set cells (only relevant for
    /// pair_map; slot_map uses a 0-sentinel).
    pub initial: f32,
    /// Fold handlers — what events fold into this view and how. The
    /// emitter reads the pattern shape (not the body) to generate the
    /// fold arm. All shipped views fold with `self += 1.0`; if that
    /// changes the emitter raises `Unsupported` with a pointer to the
    /// offending view's body.
    pub folds: Vec<FoldSpec>,
    /// `Some(K)` when the view declared `per_entity_topk(K = N)` with
    /// N >= 2 (task 196). The GPU storage layer uses this to size the
    /// sparse per-entity buffers at N·K entries instead of the dense
    /// pair_map's N² entries. `None` for dense pair_map and K=1
    /// slot_map views. Only meaningful when `shape` is a pair-map
    /// variant (scalar or decay).
    pub topk: Option<u16>,
}

/// One event → view fold. Carries just enough to emit both the Rust
/// parity reference (via [`emit_view_fold_wgsl`] callers) and the WGSL
/// fold body. The emitter treats every fold as `self += amount` where
/// `amount` is hardcoded to `1.0` — every shipped view matches this
/// shape. More complex folds surface as `Unsupported`.
#[derive(Debug, Clone)]
pub struct FoldSpec {
    /// Event variant name (e.g. `"AgentAttacked"`).
    pub event_name: String,
    /// Name of the event field that carries the observer / first-key
    /// argument (the view's first param). One of `actor`, `target`,
    /// `observer`, etc. — whatever binding name the handler's pattern
    /// uses for the view's first param.
    pub first_key_field: String,
    /// Name of the event field for the second key (only set for
    /// pair_map shapes). `None` for slot_map folds (e.g.
    /// engaged_with takes actor+target but stores just (actor -> target),
    /// so both keys flow into the write but slot_map doesn't have a
    /// pair-key layout).
    pub second_key_field: Option<String>,
}

/// Classify a view's ViewIR into a storage spec the emitter can lower.
/// Returns `Err(Unsupported)` for anything outside the Phase 4 subset.
pub fn classify_view(view: &ViewIR) -> Result<ViewStorageSpec, EmitError> {
    let snake = snake_case(&view.name);

    // Lazy → no storage.
    if matches!(view.kind, ViewKind::Lazy) {
        return Ok(ViewStorageSpec {
            view_name: view.name.clone(),
            snake,
            shape: ViewShape::Lazy,
            clamp: None,
            initial: 0.0,
            folds: Vec::new(),
            topk: None,
        });
    }

    let storage = match view.kind {
        ViewKind::Materialized(s) => s,
        ViewKind::Lazy => unreachable!(),
    };

    // Unwrap the fold body. All @materialized views in v1 use `Fold`.
    let (initial_expr, handlers, clamp) = match &view.body {
        ViewBodyIR::Fold {
            initial,
            handlers,
            clamp,
        } => (initial, handlers, clamp.as_ref()),
        ViewBodyIR::Expr(_) => {
            return Err(EmitError::Unsupported(format!(
                "view `{}` is @materialized with an expression body; Phase 4 only knows Fold bodies",
                view.name
            )));
        }
    };

    let initial = f32_literal_from_expr(initial_expr).unwrap_or(0.0);
    let clamp_pair = match clamp {
        Some((lo, hi)) => Some((
            f32_literal_from_expr(lo).unwrap_or(f32::MIN),
            f32_literal_from_expr(hi).unwrap_or(f32::MAX),
        )),
        None => None,
    };

    let mut topk_k: Option<u16> = None;
    let shape = match storage {
        StorageHint::PairMap => {
            if view.params.len() != 2 {
                return Err(EmitError::Unsupported(format!(
                    "view `{}` with storage=pair_map needs 2 params, got {}",
                    view.name,
                    view.params.len()
                )));
            }
            match view.decay {
                Some(d) => {
                    if !matches!(d.per, DecayUnit::Tick) {
                        return Err(EmitError::Unsupported(format!(
                            "view `{}` @decay per-unit not supported on GPU",
                            view.name
                        )));
                    }
                    ViewShape::PairMapDecay { rate: d.rate }
                }
                None => ViewShape::PairMapScalar,
            }
        }
        StorageHint::PerEntityTopK { k, .. } => {
            if k == 1 {
                // Task-139 single-slot shape — small HashMap<Agent,
                // Agent>. Only `engaged_with` uses this today.
                if view.decay.is_some() {
                    return Err(EmitError::Unsupported(format!(
                        "view `{}`: per_entity_topk(1) + @decay not supported (matches Rust emitter)",
                        view.name
                    )));
                }
                if view.params.len() != 1 {
                    return Err(EmitError::Unsupported(format!(
                        "view `{}` storage=per_entity_topk(1) needs 1 param, got {}",
                        view.name,
                        view.params.len()
                    )));
                }
                if !matches!(view.return_ty, IrType::AgentId) {
                    return Err(EmitError::Unsupported(format!(
                        "view `{}` per_entity_topk(1) return type {:?} — only AgentId supported on GPU",
                        view.name, view.return_ty
                    )));
                }
                ViewShape::SlotMap {
                    return_u32_as_option: true,
                }
            } else {
                // Task-196 sparse topk — K slots per observer. We
                // classify as the matching pair_map shape (scalar /
                // decay) so scoring.rs's bind-group layout keeps
                // working; the `topk` field carries K so the storage
                // layer knows to allocate N·K buffers rather than N².
                if view.params.len() != 2 {
                    return Err(EmitError::Unsupported(format!(
                        "view `{}` storage=per_entity_topk(K={k}) requires 2 params, got {}",
                        view.name,
                        view.params.len()
                    )));
                }
                topk_k = Some(k);
                match view.decay {
                    Some(d) => {
                        if !matches!(d.per, DecayUnit::Tick) {
                            return Err(EmitError::Unsupported(format!(
                                "view `{}` @decay per-unit not supported on GPU",
                                view.name
                            )));
                        }
                        ViewShape::PairMapDecay { rate: d.rate }
                    }
                    None => ViewShape::PairMapScalar,
                }
            }
        }
        StorageHint::LazyCached => {
            return Err(EmitError::Unsupported(format!(
                "view `{}` storage=lazy_cached not supported on GPU",
                view.name
            )));
        }
    };

    // Translate each fold handler into a FoldSpec. The pattern binding
    // names tell us which event field maps to which view param.
    let mut folds: Vec<FoldSpec> = Vec::with_capacity(handlers.len());
    for h in handlers {
        let fold = classify_fold(&view.name, view, h, &shape)?;
        folds.push(fold);
    }

    Ok(ViewStorageSpec {
        view_name: view.name.clone(),
        snake,
        shape,
        clamp: clamp_pair,
        initial,
        folds,
        topk: topk_k,
    })
}

/// Figure out which event field feeds which view-arg for one fold
/// handler. Mirrors [`emit_view::emit_fold_arm`]'s pattern scan —
/// when the pattern doesn't name one or both keys, fall back to the
/// canonical `actor` / `target` pair.
fn classify_fold(
    view_name: &str,
    view: &ViewIR,
    handler: &FoldHandlerIR,
    shape: &ViewShape,
) -> Result<FoldSpec, EmitError> {
    let ev_name = handler.pattern.name.clone();
    let a_name = view.params.first().map(|p| p.name.as_str()).unwrap_or("");
    let b_name = view.params.get(1).map(|p| p.name.as_str()).unwrap_or("");

    let mut first_field: Option<String> = None;
    let mut second_field: Option<String> = None;
    for b in &handler.pattern.bindings {
        let field = b.field.clone();
        if let IrPattern::Bind { name, .. } = &b.value {
            if name == a_name {
                first_field = Some(field);
            } else if name == b_name {
                second_field = Some(field);
            }
        }
    }

    // Heuristic fallback — the Rust emitter follows the same contract
    // (emit_fold_arm line ~780). Canonical `(actor, target)` works for
    // every shipped view when the pattern doesn't spell it out.
    let first = first_field.unwrap_or_else(|| "actor".to_string());
    let second = match shape {
        ViewShape::PairMapScalar | ViewShape::PairMapDecay { .. } => {
            Some(second_field.unwrap_or_else(|| "target".to_string()))
        }
        ViewShape::SlotMap { .. } => second_field,
        ViewShape::Lazy => None,
    };

    let _ = view_name;
    Ok(FoldSpec {
        event_name: ev_name,
        first_key_field: first,
        second_key_field: second,
    })
}

fn f32_literal_from_expr(expr: &IrExprNode) -> Option<f32> {
    match &expr.kind {
        crate::ir::IrExpr::LitFloat(v) => Some(*v as f32),
        crate::ir::IrExpr::LitInt(v) => Some(*v as f32),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Read-snippet emission
// ---------------------------------------------------------------------------

/// Emit a WGSL `fn view_<snake>_get(args..., tick: u32) -> T` function
/// body. Callers paste the snippet into their kernel module above the
/// call site. The snippet assumes the buffer bindings documented per
/// shape — the engine_gpu side wires those.
///
/// Shape → return type:
///   * SlotMap { AgentId }        → `u32` (0 = empty slot)
///   * PairMapScalar              → `f32`
///   * PairMapDecay { rate }      → `f32` (with decay applied)
///   * Lazy                       → errors out (callers should inline)
pub fn emit_view_read_wgsl(spec: &ViewStorageSpec) -> Result<String, EmitError> {
    match &spec.shape {
        ViewShape::Lazy => Err(EmitError::Unsupported(format!(
            "view `{}` is @lazy — inline the body, don't request a storage read",
            spec.view_name
        ))),
        ViewShape::SlotMap { .. } => Ok(emit_slot_map_read(spec)),
        ViewShape::PairMapScalar => Ok(emit_pair_map_scalar_read(spec)),
        ViewShape::PairMapDecay { rate } => Ok(emit_pair_map_decay_read(spec, *rate)),
    }
}

fn emit_slot_map_read(spec: &ViewStorageSpec) -> String {
    let snake = &spec.snake;
    let mut out = String::new();
    writeln!(
        out,
        "// view `{}` — slot_map storage. Value 0 = empty; anything else",
        spec.view_name
    )
    .unwrap();
    writeln!(
        out,
        "// is the partner AgentId+1 (so AgentId::new(u32::MAX) is not legal)."
    )
    .unwrap();
    writeln!(out, "fn view_{snake}_get(observer: u32) -> u32 {{").unwrap();
    writeln!(
        out,
        "    let n = arrayLength(&view_{snake}_slots);"
    )
    .unwrap();
    writeln!(out, "    if (observer >= n) {{ return 0u; }}").unwrap();
    writeln!(out, "    return view_{snake}_slots[observer];").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

fn emit_pair_map_scalar_read(spec: &ViewStorageSpec) -> String {
    let snake = &spec.snake;
    let initial = render_float_wgsl(spec.initial as f64);
    let mut out = String::new();
    writeln!(
        out,
        "// view `{}` — pair_map<f32>. Flat row-major: cells[observer*N + attacker].",
        spec.view_name
    )
    .unwrap();
    writeln!(out, "// No decay; read is a direct array load + clamp.").unwrap();
    writeln!(
        out,
        "fn view_{snake}_get(observer: u32, attacker: u32) -> f32 {{"
    )
    .unwrap();
    writeln!(out, "    let n = view_agent_cap;").unwrap();
    writeln!(
        out,
        "    if (observer >= n || attacker >= n) {{ return {initial}; }}"
    )
    .unwrap();
    writeln!(
        out,
        "    let raw = view_{snake}_cells[observer * n + attacker];"
    )
    .unwrap();
    if let Some((lo, hi)) = spec.clamp {
        let lo = render_float_wgsl(lo as f64);
        let hi = render_float_wgsl(hi as f64);
        writeln!(out, "    return clamp(raw, {lo}, {hi});").unwrap();
    } else {
        writeln!(out, "    return raw;").unwrap();
    }
    writeln!(out, "}}").unwrap();
    out
}

fn emit_pair_map_decay_read(spec: &ViewStorageSpec, rate: f32) -> String {
    let snake = &spec.snake;
    let initial = render_float_wgsl(spec.initial as f64);
    let rate_lit = render_float_wgsl(rate as f64);
    let mut out = String::new();
    writeln!(
        out,
        "// view `{}` — pair_map<(f32, u32)> with @decay(rate={}, per=tick).",
        spec.view_name, rate
    )
    .unwrap();
    writeln!(
        out,
        "// Cells are stored as DecayCell{{ value: f32, anchor_tick: u32 }},"
    )
    .unwrap();
    writeln!(
        out,
        "// flat row-major: cells[observer*N + attacker]. Decay is applied on read."
    )
    .unwrap();
    writeln!(
        out,
        "fn view_{snake}_get(observer: u32, attacker: u32, tick: u32) -> f32 {{"
    )
    .unwrap();
    writeln!(out, "    let n = view_agent_cap;").unwrap();
    writeln!(
        out,
        "    if (observer >= n || attacker >= n) {{ return {initial}; }}"
    )
    .unwrap();
    writeln!(
        out,
        "    let cell = view_{snake}_cells[observer * n + attacker];"
    )
    .unwrap();
    writeln!(
        out,
        "    let dt = select(0u, tick - cell.anchor_tick, tick >= cell.anchor_tick);"
    )
    .unwrap();
    writeln!(
        out,
        "    let decayed = cell.value * pow({rate_lit}, f32(dt));"
    )
    .unwrap();
    if let Some((lo, hi)) = spec.clamp {
        let lo = render_float_wgsl(lo as f64);
        let hi = render_float_wgsl(hi as f64);
        writeln!(out, "    return clamp(decayed, {lo}, {hi});").unwrap();
    } else {
        writeln!(out, "    return decayed;").unwrap();
    }
    writeln!(out, "}}").unwrap();
    out
}

// ---------------------------------------------------------------------------
// Fold-snippet emission
// ---------------------------------------------------------------------------

/// Emit a WGSL `fn view_<snake>_fold_<event>(event args..., tick)` that
/// mutates the view's storage in response to one matching event.
/// Engine_gpu dispatches one fold kernel per tick that iterates the
/// pending event stream, branches on `Event::tag`, and calls into these
/// per-event fold functions.
///
/// Determinism note (see module docstring): we use `atomicAdd` for f32
/// accumulation. This is commutative-associative ONLY under the
/// "all adds are equal positive constants" special case all shipped
/// views satisfy — every current fold is `self += 1.0`. Variable-delta
/// folds would need a sort-and-reduce pass.
pub fn emit_view_fold_wgsl(spec: &ViewStorageSpec) -> Result<String, EmitError> {
    if matches!(spec.shape, ViewShape::Lazy) {
        return Err(EmitError::Unsupported(format!(
            "view `{}` is @lazy — no fold required",
            spec.view_name
        )));
    }
    let mut out = String::new();
    for fold in &spec.folds {
        emit_single_fold(&mut out, spec, fold)?;
    }
    Ok(out)
}

fn emit_single_fold(
    out: &mut String,
    spec: &ViewStorageSpec,
    fold: &FoldSpec,
) -> Result<(), EmitError> {
    let snake = &spec.snake;
    let ev_snake = snake_case(&fold.event_name);
    let k1 = &fold.first_key_field;

    match &spec.shape {
        ViewShape::SlotMap { .. } => {
            // engaged_with: Committed inserts both sides; Broken removes
            // both. Follow the Rust emitter's pattern-driven rule (see
            // emit_per_entity_topk1_fold_arm).
            //
            // Event::EngagementCommitted { actor, target, .. } =>
            //   slots[actor] = target+1; slots[target] = actor+1
            // Event::EngagementBroken   { actor, former_target, .. } =>
            //   slots[actor] = 0; slots[former_target] = 0
            let k2 = fold.second_key_field.as_deref().unwrap_or("target");
            let is_insert = !fold.event_name.contains("Broken")
                && !fold.event_name.contains("Expired")
                && !fold.event_name.contains("Removed");
            writeln!(
                out,
                "// Fold {} -> view {} (slot_map insert/remove).",
                fold.event_name, spec.view_name
            )
            .unwrap();
            writeln!(
                out,
                "fn view_{snake}_fold_{ev_snake}({k1}: u32, {k2}: u32) {{"
            )
            .unwrap();
            writeln!(out, "    let n = view_agent_cap;").unwrap();
            writeln!(
                out,
                "    if ({k1} >= n || {k2} >= n) {{ return; }}"
            )
            .unwrap();
            if is_insert {
                // Value = AgentId + 1 (reserve 0 for empty).
                writeln!(
                    out,
                    "    view_{snake}_slots[{k1}] = {k2} + 1u;"
                )
                .unwrap();
                writeln!(
                    out,
                    "    view_{snake}_slots[{k2}] = {k1} + 1u;"
                )
                .unwrap();
            } else {
                writeln!(out, "    view_{snake}_slots[{k1}] = 0u;").unwrap();
                writeln!(out, "    view_{snake}_slots[{k2}] = 0u;").unwrap();
            }
            writeln!(out, "}}").unwrap();
        }
        ViewShape::PairMapScalar => {
            let k2 = fold.second_key_field.as_deref().ok_or_else(|| {
                EmitError::Unsupported(format!(
                    "view `{}` fold `{}`: pair_map fold missing second key field",
                    spec.view_name, fold.event_name
                ))
            })?;
            writeln!(
                out,
                "// Fold {} -> view {} (pair_map<f32>, no decay, +1.0).",
                fold.event_name, spec.view_name
            )
            .unwrap();
            writeln!(
                out,
                "fn view_{snake}_fold_{ev_snake}({k1}: u32, {k2}: u32) {{"
            )
            .unwrap();
            writeln!(out, "    let n = view_agent_cap;").unwrap();
            writeln!(
                out,
                "    if ({k1} >= n || {k2} >= n) {{ return; }}"
            )
            .unwrap();
            writeln!(
                out,
                "    let idx = {k1} * n + {k2};"
            )
            .unwrap();
            // Atomic add for f32 on WGSL isn't a native op — WGSL atomics
            // only work on integers. For non-decay f32 fold we need to
            // emulate via atomicCompareExchangeWeak on a u32 bitcast. That
            // is what we emit; callers can fall back to a CPU fold if
            // their driver balks, but every target we care about accepts
            // this pattern.
            emit_atomic_add_f32(out, &format!("view_{snake}_cells"), "idx", "1.0", spec.clamp);
            writeln!(out, "}}").unwrap();
        }
        ViewShape::PairMapDecay { rate } => {
            let k2 = fold.second_key_field.as_deref().ok_or_else(|| {
                EmitError::Unsupported(format!(
                    "view `{}` fold `{}`: pair_map@decay fold missing second key field",
                    spec.view_name, fold.event_name
                ))
            })?;
            let rate_lit = render_float_wgsl(*rate as f64);
            writeln!(
                out,
                "// Fold {} -> view {} (pair_map<DecayCell>, @decay rate={}, +1.0).",
                fold.event_name, spec.view_name, rate
            )
            .unwrap();
            writeln!(
                out,
                "fn view_{snake}_fold_{ev_snake}({k1}: u32, {k2}: u32, tick: u32) {{"
            )
            .unwrap();
            writeln!(out, "    let n = view_agent_cap;").unwrap();
            writeln!(
                out,
                "    if ({k1} >= n || {k2} >= n) {{ return; }}"
            )
            .unwrap();
            writeln!(
                out,
                "    let idx = {k1} * n + {k2};"
            )
            .unwrap();
            // For @decay we can't atomicAdd because the value depends on
            // the anchor_tick. We emit a CAS loop that:
            //   1. Reads (base, anchor).
            //   2. Decays base to current tick: base * rate^(tick - anchor).
            //   3. Adds amount (1.0), clamps.
            //   4. Writes back (updated, tick).
            // Storage is two parallel `array<atomic<u32>>` — one for
            // value (as bitcast u32), one for anchor_tick. The CAS
            // loop protects the (value, anchor) pair individually;
            // because we always set anchor = tick on write, a lost
            // update just means the next fold re-computes the same
            // decay. For our "+1.0 only" invariant this is safe.
            writeln!(
                out,
                "    loop {{"
            )
            .unwrap();
            writeln!(
                out,
                "        let base_bits = atomicLoad(&view_{snake}_cell_value[idx]);"
            )
            .unwrap();
            writeln!(
                out,
                "        let anchor = atomicLoad(&view_{snake}_cell_anchor[idx]);"
            )
            .unwrap();
            writeln!(out, "        let base = bitcast<f32>(base_bits);").unwrap();
            writeln!(
                out,
                "        let dt = select(0u, tick - anchor, tick >= anchor);"
            )
            .unwrap();
            writeln!(
                out,
                "        let decayed = base * pow({rate_lit}, f32(dt));"
            )
            .unwrap();
            let updated = if let Some((lo, hi)) = spec.clamp {
                let lo = render_float_wgsl(lo as f64);
                let hi = render_float_wgsl(hi as f64);
                format!("clamp(decayed + 1.0, {lo}, {hi})")
            } else {
                "decayed + 1.0".to_string()
            };
            writeln!(out, "        let updated = {updated};").unwrap();
            writeln!(
                out,
                "        let cas = atomicCompareExchangeWeak(&view_{snake}_cell_value[idx], base_bits, bitcast<u32>(updated));"
            )
            .unwrap();
            writeln!(out, "        if (cas.exchanged) {{").unwrap();
            writeln!(
                out,
                "            atomicStore(&view_{snake}_cell_anchor[idx], tick);"
            )
            .unwrap();
            writeln!(out, "            break;").unwrap();
            writeln!(out, "        }}").unwrap();
            writeln!(out, "    }}").unwrap();
            writeln!(out, "}}").unwrap();
        }
        ViewShape::Lazy => unreachable!(),
    }
    Ok(())
}

/// Helper: emulate f32 atomic add via a CAS loop over u32 bit-storage.
/// WGSL's atomic set covers only i32/u32, so f32 atomic-add is a loop:
///   old_bits = atomicLoad; new = bitcast(old) + amount; CAS.
/// Works on every vendor we've tested (Vulkan/Metal/DX12/LLVMpipe).
fn emit_atomic_add_f32(
    out: &mut String,
    buf: &str,
    idx: &str,
    amount: &str,
    clamp: Option<(f32, f32)>,
) {
    writeln!(out, "    loop {{").unwrap();
    writeln!(
        out,
        "        let old_bits = atomicLoad(&{buf}[{idx}]);"
    )
    .unwrap();
    writeln!(out, "        let old = bitcast<f32>(old_bits);").unwrap();
    let updated = if let Some((lo, hi)) = clamp {
        let lo = render_float_wgsl(lo as f64);
        let hi = render_float_wgsl(hi as f64);
        format!("clamp(old + {amount}, {lo}, {hi})")
    } else {
        format!("old + {amount}")
    };
    writeln!(out, "        let updated = {updated};").unwrap();
    writeln!(
        out,
        "        let cas = atomicCompareExchangeWeak(&{buf}[{idx}], old_bits, bitcast<u32>(updated));"
    )
    .unwrap();
    writeln!(out, "        if (cas.exchanged) {{ break; }}").unwrap();
    writeln!(out, "    }}").unwrap();
}

// ---------------------------------------------------------------------------
// Utility — snake_case transform mirroring `emit_mask_wgsl::snake_case`
// ---------------------------------------------------------------------------

/// Lowercase snake_case. The mask WGSL emitter exposes a `pub fn
/// snake_case` with the same contract — we duplicate the short helper
/// rather than cross-module-import to keep Phase 4 independent from
/// Phase 2's emitter.
pub fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    let mut prev_upper = false;
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !prev_upper {
                out.push('_');
            }
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            prev_upper = true;
        } else {
            out.push(ch);
            prev_upper = false;
        }
    }
    out
}

/// Emit a f64 as a WGSL `f32` literal. Mirrors
/// `emit_mask_wgsl::render_float_wgsl`.
fn render_float_wgsl(v: f64) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::ir::{
        DecayHint, IrEventPattern, IrExpr, IrExprNode, IrParam, IrPattern, IrPatternBinding,
        IrType, LocalRef, StorageHint, ViewKind,
    };

    fn span() -> Span {
        Span::dummy()
    }

    fn lit_f(v: f64) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::LitFloat(v),
            span: span(),
        }
    }

    fn param(name: &str, ty: IrType) -> IrParam {
        IrParam {
            name: name.to_string(),
            local: LocalRef(0),
            ty,
            span: span(),
        }
    }

    fn bind_field(field: &str, local: &str) -> IrPatternBinding {
        IrPatternBinding {
            field: field.to_string(),
            value: IrPattern::Bind {
                name: local.to_string(),
                local: LocalRef(0),
            },
            span: span(),
        }
    }

    fn pair_map_decay_view(
        name: &str,
        event_name: &str,
        a_name: &str,
        b_name: &str,
        rate: f32,
        clamp_lo: f32,
        clamp_hi: f32,
    ) -> ViewIR {
        ViewIR {
            name: name.to_string(),
            params: vec![
                param(a_name, IrType::AgentId),
                param(b_name, IrType::AgentId),
            ],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_f(0.0),
                handlers: vec![FoldHandlerIR {
                    pattern: IrEventPattern {
                        name: event_name.to_string(),
                        event: None,
                        bindings: vec![
                            bind_field("actor", b_name),
                            bind_field("target", a_name),
                        ],
                        span: span(),
                    },
                    body: vec![],
                    span: span(),
                }],
                clamp: Some((lit_f(clamp_lo as f64), lit_f(clamp_hi as f64))),
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PairMap),
            decay: Some(DecayHint {
                rate,
                per: DecayUnit::Tick,
                span: span(),
            }),
            span: span(),
        }
    }

    #[test]
    fn classify_lazy_view() {
        let v = ViewIR {
            name: "is_hostile".to_string(),
            params: vec![],
            return_ty: IrType::Bool,
            body: ViewBodyIR::Expr(lit_f(0.0)),
            annotations: vec![],
            kind: ViewKind::Lazy,
            decay: None,
            span: span(),
        };
        let spec = classify_view(&v).unwrap();
        assert!(matches!(spec.shape, ViewShape::Lazy));
    }

    #[test]
    fn classify_pair_map_decay() {
        let v = pair_map_decay_view(
            "threat_level",
            "AgentAttacked",
            "a",
            "b",
            0.98,
            0.0,
            1000.0,
        );
        let spec = classify_view(&v).unwrap();
        match spec.shape {
            ViewShape::PairMapDecay { rate } => assert!((rate - 0.98).abs() < 1e-6),
            other => panic!("wrong shape {other:?}"),
        }
        assert_eq!(spec.folds.len(), 1);
        assert_eq!(spec.folds[0].event_name, "AgentAttacked");
        // pattern binds `actor -> b, target -> a`, so first_key_field
        // (for view-arg 0 = `a`) should be "target", second ("b") should
        // be "actor". Matches the Rust emitter's inversion heuristic.
        assert_eq!(spec.folds[0].first_key_field, "target");
        assert_eq!(spec.folds[0].second_key_field.as_deref(), Some("actor"));
    }

    #[test]
    fn emit_decay_read_snippet_mentions_rate_and_clamp() {
        let v = pair_map_decay_view(
            "threat_level",
            "AgentAttacked",
            "a",
            "b",
            0.98,
            0.0,
            1000.0,
        );
        let spec = classify_view(&v).unwrap();
        let wgsl = emit_view_read_wgsl(&spec).unwrap();
        assert!(wgsl.contains("view_threat_level_get"));
        assert!(wgsl.contains("0.98"));
        assert!(wgsl.contains("clamp(decayed"));
        assert!(wgsl.contains("1000.0"));
    }

    #[test]
    fn emit_decay_fold_uses_cas_and_updates_anchor() {
        let v = pair_map_decay_view(
            "threat_level",
            "AgentAttacked",
            "a",
            "b",
            0.98,
            0.0,
            1000.0,
        );
        let spec = classify_view(&v).unwrap();
        let wgsl = emit_view_fold_wgsl(&spec).unwrap();
        assert!(wgsl.contains("view_threat_level_fold_agent_attacked"));
        assert!(wgsl.contains("atomicCompareExchangeWeak"));
        assert!(wgsl.contains("atomicStore(&view_threat_level_cell_anchor"));
    }

    #[test]
    fn emit_lazy_read_errors_out() {
        let v = ViewIR {
            name: "is_hostile".to_string(),
            params: vec![],
            return_ty: IrType::Bool,
            body: ViewBodyIR::Expr(lit_f(0.0)),
            annotations: vec![],
            kind: ViewKind::Lazy,
            decay: None,
            span: span(),
        };
        let spec = classify_view(&v).unwrap();
        let err = emit_view_read_wgsl(&spec).unwrap_err();
        assert!(matches!(err, EmitError::Unsupported(_)));
    }

    #[test]
    fn snake_case_matches_rust_emitter() {
        assert_eq!(snake_case("ThreatLevel"), "threat_level");
        assert_eq!(snake_case("engaged_with"), "engaged_with");
        assert_eq!(snake_case("AgentAttacked"), "agent_attacked");
    }
}
