//! Parser error type with byte-span, context chain, and rendered source pointer.

use crate::ast::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub span: Span,
    pub context: Vec<String>,
    pub message: String,
    pub rendered: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.rendered)
    }
}

impl std::error::Error for ParseError {}

impl ParseError {
    pub fn new(source: &str, span: Span, context: Vec<String>, message: impl Into<String>) -> Self {
        let message = message.into();
        let rendered = render(source, span, &context, &message);
        ParseError { span, context, message, rendered }
    }

    /// Top-level context string (or the message if no context).
    pub fn top_context(&self) -> &str {
        self.context.last().map(String::as_str).unwrap_or(&self.message)
    }
}

fn render(source: &str, span: Span, context: &[String], message: &str) -> String {
    let (line_num, col, line_start, line_end) = locate(source, span.start);
    let line = &source[line_start..line_end];
    let caret_pad = col.saturating_sub(1);
    let caret_len = (span.end.saturating_sub(span.start)).max(1);
    let mut out = String::new();
    out.push_str(&format!("parse error: {message}\n"));
    if !context.is_empty() {
        for ctx in context.iter().rev() {
            out.push_str(&format!("  while {ctx}\n"));
        }
    }
    out.push_str(&format!("  at line {line_num}, column {col} (byte {}):\n", span.start));
    out.push_str(&format!("  | {line}\n"));
    out.push_str(&format!("  | {}{}", " ".repeat(caret_pad), "^".repeat(caret_len.min(line.len().max(1)))));
    out
}

fn locate(source: &str, byte_pos: usize) -> (usize, usize, usize, usize) {
    let pos = byte_pos.min(source.len());
    let mut line = 1usize;
    let mut line_start = 0usize;
    for (i, b) in source.as_bytes().iter().enumerate().take(pos) {
        if *b == b'\n' {
            line += 1;
            line_start = i + 1;
        }
    }
    let col = pos - line_start + 1;
    let line_end = source[line_start..]
        .find('\n')
        .map(|n| line_start + n)
        .unwrap_or(source.len());
    (line, col, line_start, line_end)
}
