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
    /// A view's `@lazy` / `@materialized` annotation set or `storage = ...`
    /// hint is malformed (missing args, infeasible combination, body shape
    /// mismatch, etc.). See spec §2.3 + §9 D31.
    InvalidViewKind {
        view_name: String,
        detail: String,
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
            ResolveError::InvalidViewKind { view_name, detail, span } => write!(
                f,
                "invalid view `{view_name}` at bytes {}..{}: {detail}",
                span.start, span.end
            ),
        }
    }
}

impl std::error::Error for ResolveError {}
