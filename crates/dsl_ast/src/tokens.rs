//! Low-level lexer helpers over `&str` input. Operates on byte offsets against
//! the original source slice via a `Cursor` wrapper.

/// Cursor over the source string. Tracks remaining input + original length so
/// we can compute absolute byte offsets for spans.
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    pub src: &'a str,
    pub pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(src: &'a str) -> Self {
        Cursor { src, pos: 0 }
    }

    pub fn remaining(&self) -> &'a str {
        &self.src[self.pos..]
    }

    pub fn eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    pub fn peek_char(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    pub fn starts_with(&self, s: &str) -> bool {
        self.remaining().starts_with(s)
    }

    pub fn starts_with_char(&self, c: char) -> bool {
        self.peek_char() == Some(c)
    }

    pub fn bump(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.src.len());
    }

    pub fn bump_char(&mut self) -> Option<char> {
        let c = self.peek_char()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    /// Skip whitespace and line comments (`//`, `#`).
    pub fn skip_ws(&mut self) {
        loop {
            let start = self.pos;
            while let Some(c) = self.peek_char() {
                if c.is_whitespace() {
                    self.pos += c.len_utf8();
                } else {
                    break;
                }
            }
            if self.starts_with("//") || self.starts_with_char('#') {
                while let Some(c) = self.peek_char() {
                    if c == '\n' {
                        break;
                    }
                    self.pos += c.len_utf8();
                }
                continue;
            }
            if self.pos == start {
                break;
            }
        }
    }
}

pub fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

pub fn is_ident_cont(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Optionally consume a Rust-style integer-type suffix on a number
/// literal: one of `u8`/`u16`/`u32`/`u64`/`i8`/`i16`/`i32`/`i64`, or
/// bare `u`/`i`. The suffix MUST NOT be followed by another
/// identifier-continuation character (so `1up` does not eat the `u`).
///
/// The suffix is purely informational at the lex layer — it is consumed
/// and discarded. Type-checking against the field's actual storage type
/// (e.g. rejecting `300u8`) is a downstream pass that does not exist
/// yet.
pub fn consume_int_suffix(c: &mut Cursor) {
    // Longest-match-first: try the 2- and 3-char forms before bare `u`/`i`.
    fn after_ok(c: &Cursor, n: usize) -> bool {
        c.src[c.pos + n..].chars().next().map_or(true, |ch| !is_ident_cont(ch))
    }
    for cand in ["u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64"] {
        if c.starts_with(cand) && after_ok(c, cand.len()) {
            c.bump(cand.len());
            return;
        }
    }
    for cand in ["u", "i"] {
        if c.starts_with(cand) && after_ok(c, cand.len()) {
            c.bump(cand.len());
            return;
        }
    }
}

/// Normalize Unicode operator aliases (∧ ∨ ¬) to ASCII equivalents at lex
/// time. We take the simple approach: on encountering these code points we
/// produce the corresponding multi-char token by synthetic rewrite in the
/// parser layer. Here we just expose the mapping.
pub fn unicode_op_ascii(c: char) -> Option<&'static str> {
    match c {
        '∧' => Some("&&"),
        '∨' => Some("||"),
        '¬' => Some("!"),
        _ => None,
    }
}
