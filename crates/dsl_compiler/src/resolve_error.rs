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
        }
    }
}

impl std::error::Error for ResolveError {}
