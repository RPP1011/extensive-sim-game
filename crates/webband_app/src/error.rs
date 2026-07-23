//! Generation errors. The TS originals `throw` at founding on any broken
//! invariant (the catalog.ts `validateSpec` trust-boundary pattern); here that
//! is a `Result::Err` — the one sanctioned "panic-like" path in this crate.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenError {
    /// `castgen: name space exhausted` / `castgen: band name space exhausted`.
    /// Unreachable in practice (the combo space dwarfs a cast's ~20 names) but
    /// the TS throws, so we surface it.
    NameSpaceExhausted(&'static str),
    /// A generation-time invariant failed (`castgen invariant: ...` in TS).
    Invariant(String),
}

impl fmt::Display for GenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenError::NameSpaceExhausted(what) => {
                write!(f, "castgen: {what} name space exhausted")
            }
            GenError::Invariant(msg) => write!(f, "castgen invariant: {msg}"),
        }
    }
}

impl std::error::Error for GenError {}
