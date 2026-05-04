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
//! - **Cascade injection — DEFERRED (SKIP):** the verb's `emit` clause
//!   needs an "action selected" event source to wire as the trigger; no
//!   such event kind exists in the current event taxonomy. Surfaced as
//!   a `LoweringError::VerbExpansionSkipped` diagnostic naming the
//!   reason; a future plan that introduces an `ActionSelected` event
//!   ring (or equivalent) lifts the SKIP.
//!
//! Today's verbs in fixture files are limited shapes: `verb Strike`
//! (predator_prey), `verb Wait` / `verb MoveToward` (crowd_navigation).
//! All three expand to mask + scoring; none expand to cascade.

use dsl_ast::ast::Span;
use dsl_ast::ir::{
    Compilation, IrActionHead, IrActionHeadShape, IrExpr, IrExprNode, IrType, MaskIR,
    ScoringEntryIR, ScoringIR, VerbIR,
};

use super::error::LoweringError;

/// Result of [`expand_verbs`] — the rewritten compilation plus any
/// diagnostics emitted while attempting per-verb expansion. The
/// rewritten compilation is always returned (best-effort), even when
/// some verbs surfaced SKIP diagnostics.
pub struct VerbExpansionOutcome {
    pub compilation: Compilation,
    pub diagnostics: Vec<LoweringError>,
}

/// Expand every `verb` declaration in `comp` into its constituent
/// mask + scoring (cascade is deferred — see module docs). Returns a
/// new [`Compilation`] that owns the injected primitives plus a
/// diagnostic list naming any verb shape the pass could not fully
/// expand.
///
/// The original `comp` is consumed by reference and cloned; the
/// caller decides whether to retain the pre-expansion shape (e.g.,
/// for diff tooling) or discard it.
pub fn expand_verbs(comp: &Compilation) -> VerbExpansionOutcome {
    let mut out = comp.clone();
    let mut diagnostics = Vec::new();

    // Iterate in source order so synthesised entries are deterministic.
    // Borrowing `out.verbs` for iteration would conflict with the
    // mutating injects below; clone the small VerbIR list once.
    let verbs = out.verbs.clone();
    for verb in &verbs {
        expand_one_verb(verb, &mut out, &mut diagnostics);
    }

    VerbExpansionOutcome {
        compilation: out,
        diagnostics,
    }
}

/// Per-verb expansion: inject mask + (optional) scoring entry, surface
/// a SKIP diagnostic for the deferred cascade.
fn expand_one_verb(
    verb: &VerbIR,
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
        let entry = ScoringEntryIR {
            head: IrActionHead {
                name: synthetic_name.clone(),
                shape: IrActionHeadShape::None,
                span: verb.span,
            },
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

    // -- (3) Cascade handler injection — DEFERRED -------------------------
    //
    // The cascade trigger needs an "action selected" event source so
    // the synthesised PhysicsHandlerIR has something to bind `on
    // <Event>` against. No such event exists in the current
    // taxonomy; the IR doesn't yet support `on <Action>`-style
    // physics patterns either. Surface a SKIP diagnostic for any
    // verb that has a non-empty `emits` list so the gap is visible
    // at lower time rather than silently absent from the emitted
    // physics kernel set.
    if !verb.emits.is_empty() {
        diagnostics.push(LoweringError::VerbExpansionSkipped {
            verb_name: verb.name.clone(),
            reason: VerbSkipReason::CascadeNeedsActionEvent,
            span: verb.span,
        });
    }
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
    fn verb_with_emit_surfaces_skip_diagnostic() {
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
        let outcome = expand_verbs(&comp);
        // Mask + scoring still expanded.
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
        // Cascade SKIP diagnostic surfaced.
        let skipped = outcome
            .diagnostics
            .iter()
            .filter(|d| {
                matches!(
                    d,
                    LoweringError::VerbExpansionSkipped {
                        verb_name,
                        reason: VerbSkipReason::CascadeNeedsActionEvent,
                        ..
                    } if verb_name == "Strike"
                )
            })
            .count();
        assert_eq!(
            skipped, 1,
            "expected one cascade-deferred SKIP diagnostic for Strike; got {:?}",
            outcome.diagnostics,
        );
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
