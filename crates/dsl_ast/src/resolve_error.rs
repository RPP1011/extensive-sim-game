//! Name-resolution error type. Kept small and span-pinned; full rendering
//! (with source excerpts) is done by the caller if needed.

use crate::ast::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    DuplicateDecl {
        kind: &'static str,
        name: String,
        first: Span,
        second: Span,
    },
    UnknownIdent {
        name: String,
        span: Span,
        suggestions: Vec<String>,
    },
    UnknownField {
        entity: String,
        field: String,
        span: Span,
    },
    PatternArityMismatch {
        expected: usize,
        actual: usize,
        span: Span,
    },
    SelfInTopLevel {
        span: Span,
    },
    TooManyDecls {
        kind: &'static str,
    },
    /// An `@tag_name` annotation on an event references a tag this
    /// compilation doesn't declare. Suggestions are the closest declared
    /// tag names.
    UnknownEventTag {
        name: String,
        span: Span,
        suggestions: Vec<String>,
    },
    /// An event claims `@tag_name` but is missing one of the tag's
    /// required fields, or declares it with a mismatched type.
    EventTagContractViolated {
        event: String,
        tag: String,
        details: Vec<String>,
        span: Span,
    },
    /// A physics `on @tag` handler binds a field the tag doesn't declare.
    TagBindingUnknown {
        tag: String,
        field: String,
        span: Span,
    },
    /// A `@decay(rate=R, per=tick)` annotation is malformed. `detail` carries
    /// the specific constraint violated (missing args, invalid rate range,
    /// unsupported `per` unit, attached to a non-`@materialized` or
    /// non-fold view, etc.).
    InvalidDecayHint {
        detail: String,
        span: Span,
    },
    /// A view's `@materialized` fold body uses a construct outside the
    /// closed operator set documented in spec §2.3. `offending_construct`
    /// is a short human-readable label (e.g. "unresolved call `helper`",
    /// "while loop", "unbounded `for`").
    UdfInViewFoldBody {
        view_name: String,
        offending_construct: String,
        span: Span,
    },
    /// A `mask <Name>(...) when <predicate>` body uses a construct outside
    /// the closed operator set. Mask predicates compile to GPU boolean
    /// kernels (one row per candidate), so pattern-matching control flow
    /// and unbounded iteration are forbidden — use quantifiers
    /// (`forall`/`exists`) or bounded aggregates (`count`/`sum`/…) instead.
    /// Task 155 (commit 9ba805c6) expanded the *physics* surface to
    /// include `for`/`match`, but mask bodies stayed restricted by design
    /// since they lower to SPIR-V kernels.
    UdfInMaskBody {
        mask_name: String,
        offending_construct: String,
        span: Span,
    },
    /// A `scoring { <Action>(...) = <expr> }` entry uses a construct
    /// outside the closed operator set. Scoring rows share the same
    /// GPU kernel surface as mask predicates: no `match`, no `for`, no
    /// user-defined helpers. Task 155 confirmed this stays the contract
    /// even as physics bodies gained the richer CPU surface.
    UdfInScoringBody {
        offending_construct: String,
        span: Span,
    },
    /// A view's `@lazy` / `@materialized` annotation set or `storage = ...`
    /// hint is malformed (missing args, infeasible combination, body shape
    /// mismatch, etc.). See spec §2.3 + §9 D31.
    InvalidViewKind {
        view_name: String,
        detail: String,
        span: Span,
    },
    /// A `beliefs(o).observe(t) with { <field>: … }` statement names a field
    /// that doesn't exist on `BeliefState`. `valid` lists the known-good
    /// names so the error message can be actionable.
    UnknownBeliefField {
        field: String,
        valid: Vec<&'static str>,
        span: Span,
    },
    /// A `physics <name>` rule body contains a construct that can't be
    /// emitted as a SPIR-V kernel. See `compiler/spec.md` §1.2 for the
    /// GPU-emittable surface: POD discipline, bounded inline-array
    /// iteration, no heap, no recursion outside `@terminating_in`-capped
    /// self-emission, no user-defined helpers, no `String` bindings.
    ///
    /// `construct` names the offending shape in author terms (e.g.
    /// `unresolved call 'helper'`, `for-loop over user-defined helper`,
    /// `recursive self-emission of AgentCast`, `String let-binding`).
    /// `reason` expands on which spec clause was violated and what to use
    /// instead.
    NotGpuEmittable {
        physics_name: String,
        construct: String,
        reason: String,
        span: Span,
    },
    /// A `spatial_query <name>(<params>) = ...` declaration's first
    /// two positional binders are not named `self` and `candidate`.
    /// Phase 7 Task 4 fixes the convention: `self` is the querying
    /// agent and `candidate` is the per-pair neighbour the filter
    /// inspects; downstream lowering (Task 5) reaches into the
    /// `target_local` flag via the `candidate` binder.
    SpatialQueryRequiresSelfCandidateBinders {
        decl_name: String,
        span: Span,
    },
    /// A `from spatial.<name>(...)` reference (or its
    /// `spatial::<name>(...)` flat sibling) names a query no
    /// `spatial_query <name>` declaration supplies. Phase 7 Task 4
    /// surfaces this as a typed defect rather than a silent
    /// `NamespaceCall` carry-through; the lowering pass needs a
    /// resolved declaration.
    UnknownSpatialQuery {
        name: String,
        span: Span,
    },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::DuplicateDecl { kind, name, first, second } => write!(
                f,
                "duplicate {kind} declaration `{name}` at bytes {}..{} (first at {}..{})",
                second.start, second.end, first.start, first.end
            ),
            ResolveError::UnknownIdent { name, span, suggestions } => {
                write!(f, "unknown identifier `{name}` at bytes {}..{}", span.start, span.end)?;
                if !suggestions.is_empty() {
                    write!(f, " (did you mean: {}?)", suggestions.join(", "))?;
                }
                Ok(())
            }
            ResolveError::UnknownField { entity, field, span } => write!(
                f,
                "unknown field `{field}` on `{entity}` at bytes {}..{}",
                span.start, span.end
            ),
            ResolveError::PatternArityMismatch { expected, actual, span } => write!(
                f,
                "pattern arity mismatch: expected {expected}, got {actual} at bytes {}..{}",
                span.start, span.end
            ),
            ResolveError::SelfInTopLevel { span } => write!(
                f,
                "`self` used outside a decl that has an implicit self at bytes {}..{}",
                span.start, span.end
            ),
            ResolveError::TooManyDecls { kind } => write!(
                f,
                "too many `{kind}` declarations (16-bit limit exceeded)"
            ),
            ResolveError::UnknownEventTag { name, span, suggestions } => {
                write!(
                    f,
                    "unknown event_tag `@{name}` at bytes {}..{}",
                    span.start, span.end
                )?;
                if !suggestions.is_empty() {
                    write!(f, " (did you mean: {}?)", suggestions.join(", "))?;
                }
                Ok(())
            }
            ResolveError::EventTagContractViolated { event, tag, details, span } => {
                write!(
                    f,
                    "event `{event}` claims `@{tag}` but violates the tag contract at bytes {}..{}: ",
                    span.start, span.end
                )?;
                write!(f, "{}", details.join("; "))
            }
            ResolveError::TagBindingUnknown { tag, field, span } => write!(
                f,
                "tag `@{tag}` does not declare field `{field}` at bytes {}..{}",
                span.start, span.end
            ),
            ResolveError::InvalidDecayHint { detail, span } => write!(
                f,
                "invalid `@decay` annotation at bytes {}..{}: {detail}",
                span.start, span.end
            ),
            ResolveError::UdfInViewFoldBody { view_name, offending_construct, span } => write!(
                f,
                "view `{view_name}` fold body uses `{offending_construct}` at bytes {}..{} \
                 (fold bodies are restricted to the closed operator set: \
                 self += / -= / *= / /=, self = <expr>, if/else, arithmetic, \
                 comparison, logical, count/sum/min/max over bounded collections, \
                 abs/floor/ceil/pow/ln/sqrt/clamp, stdlib 1-hop accessors, \
                 and `let` bindings — no user-defined helpers, no loops, no \
                 cross-view composition; see spec §2.3)",
                span.start, span.end
            ),
            ResolveError::UdfInMaskBody { mask_name, offending_construct, span } => write!(
                f,
                "mask `{mask_name}` body uses `{offending_construct}` at bytes {}..{} \
                 (mask predicates compile to GPU boolean kernels; use quantifiers \
                 `forall`/`exists` or bounded aggregates `count`/`sum`/`min`/`max` \
                 instead of `match`/`for`/user-defined helpers; see spec §2.5)",
                span.start, span.end
            ),
            ResolveError::UdfInScoringBody { offending_construct, span } => write!(
                f,
                "scoring body uses `{offending_construct}` at bytes {}..{} \
                 (scoring rows share the mask kernel surface — no `match`, no `for`, \
                 no user-defined helpers; use `if/else` or per-unit gradient terms \
                 instead; see spec §2.5)",
                span.start, span.end
            ),
            ResolveError::InvalidViewKind { view_name, detail, span } => write!(
                f,
                "invalid view `{view_name}` at bytes {}..{}: {detail}",
                span.start, span.end
            ),
            ResolveError::UnknownBeliefField { field, valid, span } => write!(
                f,
                "unknown belief field `{field}` at bytes {}..{} \
                 (valid: {})",
                span.start,
                span.end,
                valid.join(", ")
            ),
            ResolveError::NotGpuEmittable {
                physics_name,
                construct,
                reason,
                span,
            } => write!(
                f,
                "physics `{physics_name}` body uses `{construct}` at bytes {}..{} \
                 ({reason}; physics bodies must stay GPU-emittable — POD \
                 discipline, bounded inline-array iteration, no heap, no \
                 recursion beyond `@terminating_in`-capped self-emission, no \
                 user-defined helpers, no `String` bindings; see \
                 compiler/spec.md §1.2)",
                span.start, span.end
            ),
            ResolveError::SpatialQueryRequiresSelfCandidateBinders { decl_name, span } => write!(
                f,
                "spatial_query `{decl_name}` at bytes {}..{} must declare \
                 its first two positional binders as `self` and `candidate` \
                 (Phase 7 Task 4: `self` is the querying agent, `candidate` \
                 the per-pair neighbour the filter inspects)",
                span.start, span.end
            ),
            ResolveError::UnknownSpatialQuery { name, span } => write!(
                f,
                "unknown spatial_query `{name}` at bytes {}..{} \
                 (no matching `spatial_query <name>` declaration found)",
                span.start, span.end
            ),
        }
    }
}

impl std::error::Error for ResolveError {}
