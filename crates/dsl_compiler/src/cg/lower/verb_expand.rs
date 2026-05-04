//! Lower-time `verb` desugaring — Slice A of the verb / probe / metric
//! emit plan (`docs/superpowers/plans/2026-05-03-verb-probe-metric-emit.md`).
//!
//! # What this does
//!
//! `verb <Name>(<params>) = action <Action>(<args>) when <pred> emit <ev>
//! score <expr>` is composition sugar for three primitive declarations:
//!
//! 1. A `mask <Name>(<params>) when <pred>` entry (gates the action's
//!    targets).
//! 2. A physics cascade handler that emits `<ev>` when the action fires.
//! 3. A `<Name> = <expr>` entry on the scoring table (utility row).
//!
//! Per spec §2.6:
//!
//! > The compiler expands a `verb` into: (1) a mask entry narrowing an
//! > existing primitive, (2) a cascade handler that emits the declared
//! > event on successful action application, (3) a scoring entry
//! > appended to the scoring table.
//!
//! This pass walks `comp.verbs` and injects the synthesised primitives
//! into `comp.masks` / `comp.physics` / `comp.scoring` so the existing
//! mask / physics / scoring lowering passes pick them up automatically
//! — there is **no new emit file**, only a lower-time arena rewrite.
//!
//! # Naming convention
//!
//! Every synthesised entry carries the prefix `verb_<verb_name>` so it
//! never collides with hand-written declarations of the same name.
//! The aspirational `verb`-bearing fixtures in `assets/sim/`
//! (`predator_prey.sim`, `crowd_navigation.sim`) are design references
//! and don't parse end-to-end today (they predate today's resolver
//! coverage); the unit tests in this module exercise the expansion via
//! inline DSL strings, mirroring the precedent established by the
//! invariant emit pass (`crates/dsl_compiler/src/cg/emit/invariants.rs`).
//! When a runtime-bound `*_min.sim` fixture grows a `verb` decl, the
//! expansion runs against it automatically — the lowering pre-pass
//! has no per-fixture branching.
//!
//! # Supported expansion shape (initial scope)
//!
//! The initial expansion covers two of the three primitive injections
//! and surfaces the third as a typed deferral:
//!
//! - **Mask injection (always attempted):** the verb's `when` predicate
//!   becomes the mask predicate. `IrActionHeadShape` mirrors the verb's
//!   parameters minus the implicit `self` (positional, in source order).
//!   When the verb has no `when` clause, the predicate is `LitBool(true)`
//!   — an unconditional always-on mask.
//!
//! - **Scoring injection (when `verb.scoring` is `Some`):** the verb's
//!   score expression becomes a `Head = expr` row appended to the FIRST
//!   `ScoringIR` block in the compilation. When no scoring block exists
//!   yet, a synthetic block is created.
//!
//! - **Cascade injection (closed 2026-05-03):** the verb expander
//!   synthesises a per-`Compilation`-singleton `ActionSelected` event
//!   kind (payload `{ actor: AgentId, action_id: U32, target: AgentId }`)
//!   the first time a verb with non-empty `emit` is seen. Each such
//!   verb gets a stable `action_id: u32` (allocated in source order
//!   over verbs with non-empty emit) and a synthesised `physics
//!   verb_chronicle_<name>` rule whose handler binds
//!   `on ActionSelected { actor: <verb's `self` LocalRef>, action_id:
//!   <fresh LocalRef>, target: <verb's `target` LocalRef when present;
//!   else fresh> }`, guards on `where action_id_local == <verb's id
//!   literal>`, and runs the verb's `emit` clauses as the body. The
//!   synthesised event + physics land in `comp.events` / `comp.physics`
//!   before driver Phase-1 registry walks pick them up — no new IR
//!   variant is introduced.
//!
//!   The runtime side (the scoring kernel writing `ActionSelected` per
//!   tick when a verb's row wins the per-agent argmax) is a separate
//!   slice; the compiler now closes the IR-shape gap so the cascade
//!   physics handler exists in the lowered program. A verb whose body
//!   reads `<bound>.<field>` (e.g., `self.pos`) on the synthesised
//!   binding still surfaces as an `UnsupportedFieldBase` from the
//!   downstream physics lowering — that's an orthogonal expression-
//!   layer gap, not a verb-expansion regression.
//!
//! Today's verbs in fixture files are limited shapes: `verb Strike`
//! (predator_prey), `verb Wait` / `verb MoveToward` (crowd_navigation).
//! Wait/MoveToward expand to mask + scoring (no `emit`); Strike
//! additionally synthesises a cascade physics handler on
//! `ActionSelected`.

use dsl_ast::ast::{BinOp, Span};
use dsl_ast::ir::{
    Compilation, EventField, EventIR, EventRef, IrActionHead, IrActionHeadShape,
    IrEventPattern, IrExpr, IrExprNode, IrPattern, IrPatternBinding,
    IrPhysicsPattern, IrStmt, IrType, LocalRef, MaskIR, PhysicsHandlerIR, PhysicsIR,
    ScoringEntryIR, ScoringIR, VerbIR,
};

use super::error::LoweringError;

/// Reserved name for the synthesised "an action was selected this tick"
/// event the verb expander injects on demand. The payload is fixed:
/// `{ actor: AgentId, action_id: U32, target: AgentId }`. The runtime
/// side (the scoring kernel writing the event when a verb-derived row
/// wins the per-agent argmax) is a separate slice; the compiler injects
/// the IR shape so cascade handlers can be authored against it.
pub const ACTION_SELECTED_EVENT_NAME: &str = "ActionSelected";

/// Synthetic field name for the "no target" sentinel an actor uses
/// when the verb's action has no target binder (e.g., `verb Wait(self)
/// = action Hold ...`). Today we still emit the field — the value is
/// reused as `actor` for shape symmetry; downstream cascade bodies
/// gate on `action_id` and ignore the target.
const ACTION_SELECTED_FIELD_ACTOR: &str = "actor";
const ACTION_SELECTED_FIELD_ACTION_ID: &str = "action_id";
const ACTION_SELECTED_FIELD_TARGET: &str = "target";

/// Result of [`expand_verbs`] — the rewritten compilation plus any
/// diagnostics emitted while attempting per-verb expansion. The
/// rewritten compilation is always returned (best-effort), even when
/// some verbs surfaced SKIP diagnostics.
pub struct VerbExpansionOutcome {
    pub compilation: Compilation,
    pub diagnostics: Vec<LoweringError>,
}

/// Expand every `verb` declaration in `comp` into its constituent
/// mask + scoring + cascade primitives. Returns a new [`Compilation`]
/// that owns the injected primitives plus a diagnostic list naming
/// any verb shape the pass could not fully expand.
///
/// The original `comp` is consumed by reference and cloned; the
/// caller decides whether to retain the pre-expansion shape (e.g.,
/// for diff tooling) or discard it.
///
/// # Cascade injection contract
///
/// The first verb with a non-empty `emit` triggers a one-time
/// injection of the `ActionSelected` event kind into `comp.events`
/// (idempotent — re-runs of the pass on the same compilation reuse
/// the prior entry). Each emit-bearing verb then gets:
///
/// - A stable `action_id: u32` allocated in source-order over the
///   subset of verbs with non-empty emit (verb #0 with emit → 0,
///   verb #1 with emit → 1, …). Verbs without emit don't consume an
///   id.
/// - A synthesised [`PhysicsIR`] named `verb_chronicle_<verb_name>`
///   whose handler binds `on ActionSelected { actor, action_id,
///   target }` and gates the verb's `emit` clauses on
///   `action_id == <verb's stable id>`.
///
/// The synthesised event + physics land in `comp.events` /
/// `comp.physics`; the existing driver's Phase-1 registry walks
/// (`populate_event_kinds`, `lower_all_physics`) pick them up
/// automatically — no new IR variant is introduced.
pub fn expand_verbs(comp: &Compilation) -> VerbExpansionOutcome {
    let mut out = comp.clone();
    let mut diagnostics = Vec::new();

    // Iterate in source order so synthesised entries are deterministic.
    // Borrowing `out.verbs` for iteration would conflict with the
    // mutating injects below; clone the small VerbIR list once.
    let verbs = out.verbs.clone();

    // Pre-pass: inject the ActionSelected event kind once if any verb
    // has a non-empty emit. Returning the EventRef so each verb's
    // synthesised cascade handler binds against the same kind id (the
    // driver assigns EventKindId by source order in `comp.events`).
    let action_selected_ref = if verbs.iter().any(|v| !v.emits.is_empty()) {
        Some(inject_action_selected_event(&mut out))
    } else {
        None
    };

    // Stable per-verb action_id allocator. Each verb with a non-empty
    // emit consumes the next id; verbs without emit are skipped so
    // the id space is contiguous over the cascade-bearing subset.
    let mut next_action_id: u32 = 0;
    for verb in &verbs {
        let assigned_action_id = if verb.emits.is_empty() {
            None
        } else {
            let id = next_action_id;
            next_action_id += 1;
            Some(id)
        };
        expand_one_verb(
            verb,
            assigned_action_id,
            action_selected_ref,
            &mut out,
            &mut diagnostics,
        );
    }

    VerbExpansionOutcome {
        compilation: out,
        diagnostics,
    }
}

/// Inject (or reuse) the singleton `ActionSelected` event kind into
/// `comp.events`. Idempotent: a re-run on the same compilation walks
/// `comp.events` for an existing entry by name and returns its
/// `EventRef` rather than appending a duplicate.
///
/// The fixed payload is `{ actor: AgentId, action_id: U32, target:
/// AgentId }`. Adding an entry to `comp.events` is sufficient — the
/// driver's [`super::driver::populate_event_kinds`] walks the table
/// in source order and assigns each entry an `EventKindId(i)` plus
/// an `EventLayout`. The `EventField` shape for `AgentId` and `U32`
/// already maps to one u32 word in
/// [`super::driver::cg_ty_for_event_field`].
fn inject_action_selected_event(comp: &mut Compilation) -> EventRef {
    if let Some((idx, _)) = comp
        .events
        .iter()
        .enumerate()
        .find(|(_, e)| e.name == ACTION_SELECTED_EVENT_NAME)
    {
        return EventRef(idx as u16);
    }
    let idx = comp.events.len();
    let span = Span::dummy();
    comp.events.push(EventIR {
        name: ACTION_SELECTED_EVENT_NAME.to_string(),
        fields: vec![
            EventField {
                name: ACTION_SELECTED_FIELD_ACTOR.to_string(),
                ty: IrType::AgentId,
                span,
            },
            EventField {
                name: ACTION_SELECTED_FIELD_ACTION_ID.to_string(),
                ty: IrType::U32,
                span,
            },
            EventField {
                name: ACTION_SELECTED_FIELD_TARGET.to_string(),
                ty: IrType::AgentId,
                span,
            },
        ],
        tags: Vec::new(),
        annotations: Vec::new(),
        span,
    });
    EventRef(idx as u16)
}

/// Per-verb expansion: inject mask + (optional) scoring entry +
/// (when the verb declares a non-empty `emit`) a synthesised
/// cascade physics handler that fires `on ActionSelected { … }` and
/// runs the verb's emit clauses gated on the verb's stable
/// `action_id`.
///
/// `assigned_action_id` is `Some(id)` when the verb has a non-empty
/// `emit` (the caller pre-allocated the id from the cascade-bearing
/// verb subset). `action_selected_ref` is the `EventRef` for the
/// (already-injected) `ActionSelected` event kind — `Some` whenever
/// at least one verb in the compilation needs a cascade. Both are
/// always either both `Some` or both `None` for any given verb.
fn expand_one_verb(
    verb: &VerbIR,
    assigned_action_id: Option<u32>,
    action_selected_ref: Option<EventRef>,
    comp: &mut Compilation,
    diagnostics: &mut Vec<LoweringError>,
) {
    let synthetic_name = format!("verb_{}", verb.name);

    // -- (1) Mask injection ------------------------------------------------
    //
    // Build the mask's IrActionHead from the verb's params, dropping the
    // conventional implicit `self` head (verbs in shipped fixtures all
    // start with `self` as their first param). When the verb has no
    // `when` clause, the mask predicate defaults to `true` — an
    // always-on mask; the scoring row is then the only gate on the
    // verb's selection.
    let mask_head = build_mask_head(verb, &synthetic_name);
    let mask_predicate = match &verb.when {
        Some(p) => p.clone(),
        None => IrExprNode {
            kind: IrExpr::LitBool(true),
            span: verb.span,
        },
    };
    comp.masks.push(MaskIR {
        head: mask_head,
        candidate_source: None,
        predicate: mask_predicate,
        annotations: verb.annotations.clone(),
        span: verb.span,
    });

    // -- (2) Scoring entry injection (when verb.scoring is present) -------
    //
    // The verb's score expression becomes a `Head = expr` row. Append
    // to the FIRST scoring block (which is the convention — most
    // fixtures have exactly one); synthesize a block if the
    // compilation has none yet so a verb-only fixture still produces a
    // scoring kernel.
    if let Some(score_expr) = &verb.scoring {
        // Mirror `build_mask_head`'s shape on the scoring entry's head
        // so the scoring lowering's per-pair candidate binding flow
        // (`scoring.rs:307-345` flips `ctx.target_local = true` when
        // it sees a `Positional([("target", _, AgentId)])` head)
        // resolves `target.<field>` reads in the score body to the
        // per-pair candidate context. Hardcoding `IrActionHeadShape::None`
        // here was Gap #3 in
        // `docs/superpowers/notes/2026-05-04-pair_scoring_probe.md`;
        // sharing the helper means the mask + scoring entries always
        // declare the same head shape (so `target` resolves the same
        // way on both sides of the verb expansion).
        let scoring_head = build_mask_head(verb, &synthetic_name);
        let entry = ScoringEntryIR {
            head: scoring_head,
            expr: score_expr.clone(),
            span: verb.span,
        };
        if let Some(first) = comp.scoring.first_mut() {
            first.entries.push(entry);
        } else {
            comp.scoring.push(ScoringIR {
                entries: vec![entry],
                per_ability_rows: Vec::new(),
                annotations: Vec::new(),
                span: verb.span,
            });
        }
    } else {
        // Verb has no `score` clause — nothing to inject. Not a
        // diagnostic: a verb may legitimately omit scoring (the
        // utility backend selects via some other table).
    }

    // -- (3) Cascade handler injection ----------------------------------
    //
    // For every verb with a non-empty `emit`, synthesise a physics
    // rule named `verb_chronicle_<verb_name>` whose handler binds
    // `on ActionSelected { actor: <verb's `self` LocalRef>, action_id:
    // <fresh LocalRef>, target: <verb's `target` LocalRef when present
    // — else fresh> }`, gates on `action_id_local == <verb's stable
    // id>`, and runs the verb's `emit` clauses as the body. The
    // synthesised event was injected by the per-`expand_verbs` pre-
    // pass; this site just appends the matching `PhysicsIR`.
    //
    // The event-pattern binders deliberately reuse the verb's own
    // `LocalRef`s for `self` / `target` so the verb's emit body —
    // which references `self` / `target` through those same refs —
    // resolves through the binding context the driver's
    // `synthesize_pattern_binding_lets` lays down. A fresh LocalRef
    // is allocated for the `action_id` binder (the verb's params
    // never include it).
    if !verb.emits.is_empty() {
        let event_ref = action_selected_ref
            .expect("ActionSelected event ref is always pre-injected when any verb has emits");
        let action_id = assigned_action_id
            .expect("assigned_action_id is always Some when verb.emits is non-empty");
        let physics = synthesize_cascade_physics(
            verb,
            &synthetic_name,
            action_id,
            event_ref,
        );
        comp.physics.push(physics);
    }
    // Tail use of `diagnostics` — kept in the signature so future
    // typed deferrals can be appended without re-threading the
    // parameter. (`VerbExpansionSkipped` no longer fires.)
    let _ = diagnostics;
}

/// Synthesise the cascade [`PhysicsIR`] for one verb. Caller passes
/// the verb's stable `action_id` and the pre-injected
/// `ActionSelected` event ref.
///
/// Shape:
///
/// ```text
/// physics verb_chronicle_<name> {
///     on ActionSelected { actor: <verb_self>, action_id: <fresh>, target: <verb_target_or_fresh> }
///     where action_id_local == <action_id>
///     {
///         <verb.emits, lifted as IrStmt::Emit>
///     }
/// }
/// ```
///
/// The handler binds the event's `actor` field to the verb's `self`
/// LocalRef and the event's `target` field to the verb's `target`
/// LocalRef (when present). The verb's emit body references those
/// same LocalRefs, so reads of bare `self` / `target` resolve through
/// the binding context's `local_ids` map — see the cascade-injection
/// rationale in [`expand_one_verb`].
fn synthesize_cascade_physics(
    verb: &VerbIR,
    _synthetic_name: &str,
    action_id: u32,
    event_ref: EventRef,
) -> PhysicsIR {
    let span = verb.span;

    // Resolve the verb's `self` and `target` LocalRefs from its param
    // list (param names match the convention every shipped verb
    // follows: `self` first, optional `target` second). Falling back
    // to a fresh LocalRef when a param is absent keeps the binding
    // shape uniform — the body simply doesn't reference the missing
    // binder.
    // Stateful allocator: `fresh_local_after` previously returned the
    // SAME LocalRef on every call (it was a pure `max(verb.params.local) + 1`),
    // which collided distinct synthesised binders onto a single LocalRef.
    // Concretely, when a verb has no `target` param the absent-target
    // fallback and the action_id allocation both produced the same
    // LocalRef; `synthesize_pattern_binding_lets` then recorded the
    // bindings against that single local with conflicting types
    // (action_id: U32 followed by target: AgentId), and the cascade
    // body's `if action_id == 0` failed to lower with
    // `BinaryOperandTyMismatch`. Threading a mutable counter ensures
    // every fallback / synthesis call gets a distinct LocalRef. See
    // `docs/superpowers/notes/2026-05-04-verb-fire-probe.md` (Gap #1).
    let mut next_local = first_unused_local(verb);
    let self_local = verb
        .params
        .iter()
        .find(|p| p.name == "self")
        .map(|p| p.local)
        .unwrap_or_else(|| fresh_local(&mut next_local));
    let target_local = verb
        .params
        .iter()
        .find(|p| p.name == "target")
        .map(|p| p.local)
        .unwrap_or_else(|| fresh_local(&mut next_local));
    let action_id_local = fresh_local(&mut next_local);

    let bindings = vec![
        IrPatternBinding {
            field: ACTION_SELECTED_FIELD_ACTOR.to_string(),
            value: IrPattern::Bind {
                name: "self".to_string(),
                local: self_local,
            },
            span,
        },
        IrPatternBinding {
            field: ACTION_SELECTED_FIELD_ACTION_ID.to_string(),
            value: IrPattern::Bind {
                name: "action_id".to_string(),
                local: action_id_local,
            },
            span,
        },
        IrPatternBinding {
            field: ACTION_SELECTED_FIELD_TARGET.to_string(),
            value: IrPattern::Bind {
                name: "target".to_string(),
                local: target_local,
            },
            span,
        },
    ];

    let pattern = IrPhysicsPattern::Kind(IrEventPattern {
        name: ACTION_SELECTED_EVENT_NAME.to_string(),
        event: Some(event_ref),
        bindings,
        span,
    });

    // Body: lift each `IrEmit` from the verb to an `IrStmt::Emit`,
    // wrapped in `if (action_id_local == <lit>) { … }` so the cascade
    // only fires for the verb that owns this rule.
    //
    // Why not use the `where_clause` field on `PhysicsHandlerIR`?
    // The physics lowering (`lower_one_handler`) lowers the
    // where-clause BEFORE [`super::event_binding::synthesize_pattern_binding_lets`]
    // populates the binder context. Reading `action_id` from the
    // where-clause would surface as `UnsupportedLocalBinding`. The
    // `if` lives inside `body` instead — `lower_stmt_list` runs
    // after the binders are in scope, so the `IrStmt::If` cond
    // resolves the binder cleanly.
    let action_id_lhs = IrExprNode {
        kind: IrExpr::Local(action_id_local, "action_id".to_string()),
        span,
    };
    let action_id_rhs = IrExprNode {
        kind: IrExpr::LitInt(action_id as i64),
        span,
    };
    let action_id_eq = IrExprNode {
        kind: IrExpr::Binary(BinOp::Eq, Box::new(action_id_lhs), Box::new(action_id_rhs)),
        span,
    };
    let emit_stmts: Vec<IrStmt> = verb
        .emits
        .iter()
        .cloned()
        .map(IrStmt::Emit)
        .collect();
    let body: Vec<IrStmt> = vec![IrStmt::If {
        cond: action_id_eq,
        then_body: emit_stmts,
        else_body: None,
        span,
    }];

    let handler = PhysicsHandlerIR {
        pattern,
        where_clause: None,
        body,
        span,
    };

    PhysicsIR {
        name: format!("verb_chronicle_{}", verb.name),
        handlers: vec![handler],
        annotations: Vec::new(),
        cpu_only: false,
        span,
    }
}

/// Compute the first LocalRef id strictly past every LocalRef the
/// verb already uses for its params. Doesn't query the resolver's
/// scope (we're post-resolution); instead picks `max(verb.params.local) + 1`.
/// Verbs with no params start at `LocalRef(0)`. Used as the seed for
/// the [`fresh_local`] stateful allocator below.
fn first_unused_local(verb: &VerbIR) -> u16 {
    let max = verb.params.iter().map(|p| p.local.0).max();
    max.map(|m| m.saturating_add(1)).unwrap_or(0)
}

/// Stateful counterpart to [`first_unused_local`]: returns a fresh
/// [`LocalRef`] and post-increments the counter. Each call yields a
/// distinct LocalRef, so synthesised binders never collide on a
/// single local (see Gap #1 rationale at the call site in
/// [`synthesize_cascade_physics`]).
fn fresh_local(next: &mut u16) -> LocalRef {
    let r = LocalRef(*next);
    *next = next.saturating_add(1);
    r
}

// Suppress dead-code warning: kept for the (still-typed) closed-set
// reason enum on [`LoweringError::VerbExpansionSkipped`]. No call
// site fires today, but the variant carries `VerbSkipReason` and
// removing the enum would force a wider error-surface refactor for a
// follow-on stage. Future deferrals (e.g., a verb shape that fails a
// cascade-precondition check) can re-introduce values without
// reshaping the diagnostic carrier.
#[allow(dead_code)]
fn _verb_skip_reason_kept() -> VerbSkipReason {
    VerbSkipReason::CascadeNeedsActionEvent
}

/// Build the synthesised mask's `IrActionHead` from the verb's
/// parameter list. Mirrors the convention every shipped fixture
/// follows — `self` as the first param, followed by zero or more
/// typed locals — by dropping the leading `self` and threading the
/// remainder through `IrActionHeadShape::Positional`.
///
/// A verb with no params (or only `self`) yields
/// `IrActionHeadShape::None`. A verb whose first param is not
/// literally named `self` is treated as a non-conventional shape and
/// passed through unchanged (every param surfaces as a positional
/// slot); this is conservative — none of today's fixtures hit the
/// non-self-first case so the behaviour is uncovered, but the
/// fallback is safe.
fn build_mask_head(verb: &VerbIR, synthetic_name: &str) -> IrActionHead {
    let mut params = verb.params.iter();
    let first = params.next();
    let rest: Vec<_> = match first {
        Some(p) if p.name == "self" => params
            .map(|p| (p.name.clone(), p.local, narrow_param_ty(&p.ty)))
            .collect(),
        Some(p) => {
            // Non-conventional first param — keep every slot.
            std::iter::once((p.name.clone(), p.local, narrow_param_ty(&p.ty)))
                .chain(params.map(|p| (p.name.clone(), p.local, narrow_param_ty(&p.ty))))
                .collect()
        }
        None => Vec::new(),
    };
    let shape = if rest.is_empty() {
        IrActionHeadShape::None
    } else {
        IrActionHeadShape::Positional(rest)
    };
    IrActionHead {
        name: synthetic_name.to_string(),
        shape,
        span: Span::dummy(),
    }
}

/// Narrow the verb-param type to the concrete `IrType` the mask head
/// expects. Today this is the identity transform — `IrType` already
/// covers every primitive the verb surface supports — but the
/// indirection leaves room for future widening rules (e.g., a verb
/// that types its target as `Hostile` and the mask wants the
/// underlying `AgentId`).
fn narrow_param_ty(ty: &IrType) -> IrType {
    ty.clone()
}

/// Closed-set reason a verb expansion stage skipped. Carried inside
/// [`LoweringError::VerbExpansionSkipped`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerbSkipReason {
    /// Cascade injection requires an `on <ActionSelected>` event
    /// source the current event taxonomy doesn't expose. Verbs with a
    /// non-empty `emit` list defer this stage; mask + scoring still
    /// expand.
    CascadeNeedsActionEvent,
}

impl std::fmt::Display for VerbSkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CascadeNeedsActionEvent => write!(
                f,
                "cascade trigger requires an action-selected event source \
                 (no such event kind in today's taxonomy); mask + scoring \
                 still expand"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(src: &str) -> Compilation {
        let p = crate::parse(src).expect("parse");
        dsl_ast::resolve::resolve(p).expect("resolve")
    }

    #[test]
    fn empty_compilation_passes_through() {
        let comp = Compilation::default();
        let outcome = expand_verbs(&comp);
        assert!(outcome.compilation.verbs.is_empty());
        assert!(outcome.compilation.masks.is_empty());
        assert!(outcome.compilation.scoring.is_empty());
        assert!(outcome.diagnostics.is_empty());
    }

    #[test]
    fn verb_with_when_and_score_injects_mask_and_scoring_entry() {
        let src = r#"
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
scoring {
  Hold = 0.5
}
verb Wait(self) =
  action Hold
  when  self.alive
  score 1.0
"#;
        let comp = compile(src);
        let pre_mask_count = comp.masks.len();
        let pre_scoring_entry_count: usize = comp.scoring.iter().map(|s| s.entries.len()).sum();

        let outcome = expand_verbs(&comp);
        let post_mask_count = outcome.compilation.masks.len();
        let post_scoring_entry_count: usize = outcome
            .compilation
            .scoring
            .iter()
            .map(|s| s.entries.len())
            .sum();

        assert_eq!(
            post_mask_count,
            pre_mask_count + 1,
            "expected one synthesised mask per verb"
        );
        assert_eq!(
            post_scoring_entry_count,
            pre_scoring_entry_count + 1,
            "expected one synthesised scoring entry per verb"
        );
        // Synthesised name carries the `verb_` prefix.
        let injected_mask = outcome.compilation.masks.last().expect("synthesised mask");
        assert_eq!(injected_mask.head.name, "verb_Wait");
        let injected_entry = outcome
            .compilation
            .scoring
            .first()
            .expect("scoring block")
            .entries
            .last()
            .expect("synthesised scoring entry");
        assert_eq!(injected_entry.head.name, "verb_Wait");
        // No emit clause → no SKIP diagnostic.
        assert!(
            outcome.diagnostics.is_empty(),
            "expected zero diagnostics for emit-less verb; got {:?}",
            outcome.diagnostics,
        );
    }

    #[test]
    fn verb_with_emit_injects_cascade_event_and_handler() {
        // 2026-05-03: cascade injection landed. A verb with a non-empty
        // `emit` triggers (1) a singleton ActionSelected event-kind
        // injection into `comp.events` and (2) a synthesised
        // `verb_chronicle_<name>` physics rule listening on
        // ActionSelected, gated on `where action_id == <verb id>`,
        // whose body runs the verb's emits. The mask + scoring stages
        // still expand. No `VerbExpansionSkipped` diagnostic surfaces.
        let src = r#"
event Killed { by: AgentId, prey: AgentId, pos: vec3, }
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
scoring {
  AttackTarget = 1.0
}
verb Strike(self, target: Agent) =
  action AttackTarget(target: target)
  when  self.alive
  emit  Killed { by: self, prey: target, pos: self.pos }
  score 1.0
"#;
        let comp = compile(src);
        let pre_event_count = comp.events.len();
        let pre_physics_count = comp.physics.len();
        let outcome = expand_verbs(&comp);

        // Mask + scoring stages still expand (regression guard).
        assert!(
            outcome
                .compilation
                .masks
                .iter()
                .any(|m| m.head.name == "verb_Strike"),
            "expected verb_Strike mask in expansion"
        );
        assert!(
            outcome
                .compilation
                .scoring
                .iter()
                .flat_map(|s| s.entries.iter())
                .any(|e| e.head.name == "verb_Strike"),
            "expected verb_Strike scoring entry in expansion"
        );

        // Cascade injection — singleton ActionSelected event landed.
        assert_eq!(
            outcome.compilation.events.len(),
            pre_event_count + 1,
            "expected exactly one synthesised ActionSelected event"
        );
        let action_selected = outcome
            .compilation
            .events
            .iter()
            .find(|e| e.name == ACTION_SELECTED_EVENT_NAME)
            .expect("ActionSelected event present");
        assert_eq!(
            action_selected.fields.len(),
            3,
            "ActionSelected payload = {{ actor, action_id, target }}"
        );
        assert_eq!(action_selected.fields[0].name, "actor");
        assert_eq!(action_selected.fields[1].name, "action_id");
        assert_eq!(action_selected.fields[2].name, "target");
        assert!(matches!(action_selected.fields[0].ty, IrType::AgentId));
        assert!(matches!(action_selected.fields[1].ty, IrType::U32));
        assert!(matches!(action_selected.fields[2].ty, IrType::AgentId));

        // Cascade injection — synthesised physics handler landed.
        assert_eq!(
            outcome.compilation.physics.len(),
            pre_physics_count + 1,
            "expected one synthesised verb_chronicle_<verb> physics rule"
        );
        let chronicle = outcome
            .compilation
            .physics
            .iter()
            .find(|p| p.name == "verb_chronicle_Strike")
            .expect("verb_chronicle_Strike physics present");
        assert_eq!(chronicle.handlers.len(), 1);
        let handler = &chronicle.handlers[0];
        match &handler.pattern {
            IrPhysicsPattern::Kind(p) => {
                assert_eq!(p.name, ACTION_SELECTED_EVENT_NAME);
                assert!(p.event.is_some(), "event ref resolved");
                assert_eq!(p.bindings.len(), 3, "actor + action_id + target binders");
            }
            other => panic!("expected Kind-pattern; got {other:?}"),
        }
        // The action_id gate lives inside the body as an `If` (NOT a
        // where-clause) so it lowers AFTER the binder context is
        // populated — see `synthesize_cascade_physics`'s comment.
        assert!(
            handler.where_clause.is_none(),
            "expected no where-clause; gate lives inside body as If"
        );
        assert_eq!(
            handler.body.len(),
            1,
            "verb has one emit; handler body wraps it in one If stmt"
        );
        match &handler.body[0] {
            IrStmt::If { then_body, else_body, .. } => {
                assert!(else_body.is_none());
                assert_eq!(then_body.len(), 1, "one emit inside the gate");
                match &then_body[0] {
                    IrStmt::Emit(e) => assert_eq!(e.event_name, "Killed"),
                    other => panic!("expected Emit inside If; got {other:?}"),
                }
            }
            other => panic!("expected If; got {other:?}"),
        }

        // No SKIP diagnostic — cascade is fully expanded at the IR
        // level. (Downstream lowering of `self.pos` against the
        // synthesised binding is an orthogonal expression-layer gap;
        // not a verb-expansion regression.)
        for diag in &outcome.diagnostics {
            assert!(
                !matches!(diag, LoweringError::VerbExpansionSkipped { .. }),
                "expected zero VerbExpansionSkipped diagnostics; got {diag:?}",
            );
        }
    }

    #[test]
    fn action_selected_event_injection_is_idempotent_across_multiple_verbs() {
        // Two verbs with non-empty emit must share a single
        // ActionSelected event entry; each gets its own
        // `verb_chronicle_<name>` physics rule with a distinct
        // action_id literal in the where-clause.
        let src = r#"
event Killed { by: AgentId, prey: AgentId, pos: vec3, }
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
scoring {
  AttackTarget = 1.0
  Hold = 0.5
}
verb Strike(self, target: Agent) =
  action AttackTarget(target: target)
  when  self.alive
  emit  Killed { by: self, prey: target, pos: self.pos }
  score 1.0
verb StrikeAgain(self, target: Agent) =
  action AttackTarget(target: target)
  when  self.alive
  emit  Killed { by: self, prey: target, pos: self.pos }
  score 0.5
"#;
        let comp = compile(src);
        let pre_event_count = comp.events.len();
        let outcome = expand_verbs(&comp);
        // Singleton — only one ActionSelected event.
        assert_eq!(
            outcome
                .compilation
                .events
                .iter()
                .filter(|e| e.name == ACTION_SELECTED_EVENT_NAME)
                .count(),
            1,
        );
        assert_eq!(outcome.compilation.events.len(), pre_event_count + 1);
        // Two synthesised physics rules, one per verb.
        assert!(outcome
            .compilation
            .physics
            .iter()
            .any(|p| p.name == "verb_chronicle_Strike"));
        assert!(outcome
            .compilation
            .physics
            .iter()
            .any(|p| p.name == "verb_chronicle_StrikeAgain"));
    }

    #[test]
    fn verb_without_scoring_block_synthesises_one() {
        // No `scoring` block in source — the expansion creates one.
        let src = r#"
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
verb Wait(self) =
  action Hold
  when  self.alive
  score 0.5
"#;
        let comp = compile(src);
        assert!(comp.scoring.is_empty(), "precondition: no scoring block");
        let outcome = expand_verbs(&comp);
        assert_eq!(
            outcome.compilation.scoring.len(),
            1,
            "expected one synthesised scoring block when none existed"
        );
        assert_eq!(
            outcome.compilation.scoring[0].entries.len(),
            1,
            "expected one entry in the synthesised block"
        );
    }

    #[test]
    fn verb_mask_head_drops_implicit_self() {
        // `verb Wait(self) = ...` has only `self` — mask shape is
        // `None`. `verb Strike(self, target: Agent) = ...` retains
        // `target` as the only positional slot. No scoring block
        // declared; expansion still produces masks regardless.
        let src = r#"
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
verb Wait(self) =
  action Hold
  when  self.alive
verb Strike(self, target: Agent) =
  action AttackTarget(target: target)
  when  self.alive
"#;
        let comp = compile(src);
        let outcome = expand_verbs(&comp);
        let wait_mask = outcome
            .compilation
            .masks
            .iter()
            .find(|m| m.head.name == "verb_Wait")
            .expect("verb_Wait mask");
        assert!(
            matches!(wait_mask.head.shape, IrActionHeadShape::None),
            "expected None shape after dropping `self`; got {:?}",
            wait_mask.head.shape,
        );
        let strike_mask = outcome
            .compilation
            .masks
            .iter()
            .find(|m| m.head.name == "verb_Strike")
            .expect("verb_Strike mask");
        match &strike_mask.head.shape {
            IrActionHeadShape::Positional(slots) => {
                assert_eq!(slots.len(), 1, "expected one positional slot (`target`)");
                assert_eq!(slots[0].0, "target");
            }
            other => panic!("expected Positional; got {other:?}"),
        }
    }
}
