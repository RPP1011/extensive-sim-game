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
