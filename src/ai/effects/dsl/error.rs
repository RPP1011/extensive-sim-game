//! User-friendly error formatting for DSL parse errors.

use std::fmt;

#[derive(Debug, Clone)]
pub struct DslError {
    pub message: String,
    pub line: usize,
    pub col: usize,
    pub context: Option<String>,
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}:{}: {}", self.line, self.col, self.message)?;
        if let Some(ctx) = &self.context {
            write!(f, "\n  | {ctx}")?;
        }
        Ok(())
    }
}

impl std::error::Error for DslError {}

/// Compute line and column from byte offset in input.
pub fn offset_to_line_col(input: &str, offset: usize) -> (usize, usize) {
    let clamped = offset.min(input.len());
    let before = &input[..clamped];
    let line = before.chars().filter(|&c| c == '\n').count() + 1;
    let col = before.rfind('\n').map_or(clamped, |nl| clamped - nl - 1) + 1;
    (line, col)
}

/// Extract the source line at the given line number (1-based).
pub fn extract_line(input: &str, line: usize) -> Option<String> {
    input.lines().nth(line.saturating_sub(1)).map(|s| s.to_string())
}

/// Create a DslError from a winnow parse error.
pub fn from_parse_error(input: &str, err: &str) -> DslError {
    DslError {
        message: err.to_string(),
        line: 1,
        col: 1,
        context: input.lines().next().map(|s| s.to_string()),
    }
}
