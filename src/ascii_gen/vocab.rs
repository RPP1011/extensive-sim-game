use std::collections::HashMap;

/// The set of characters valid for ASCII art output, matching the glyph atlas
/// in `src/ascii_viewport/glyph_atlas.rs`.
pub struct GlyphVocab {
    /// Ordered list of valid characters.
    pub chars: Vec<char>,
    /// Reverse lookup: char → index.
    pub char_to_idx: HashMap<char, usize>,
}

impl GlyphVocab {
    /// Build the default game vocabulary:
    /// - Printable ASCII (0x20–0x7E)
    /// - Box-drawing characters (0x2500–0x257F)
    /// - Block elements (0x2580–0x259F)
    pub fn game_default() -> Self {
        let mut chars = Vec::with_capacity(256);

        // Printable ASCII
        for code in 0x20u32..=0x7E {
            if let Some(c) = char::from_u32(code) {
                chars.push(c);
            }
        }

        // Box-drawing characters
        for code in 0x2500u32..=0x257F {
            if let Some(c) = char::from_u32(code) {
                chars.push(c);
            }
        }

        // Block elements
        for code in 0x2580u32..=0x259F {
            if let Some(c) = char::from_u32(code) {
                chars.push(c);
            }
        }

        let char_to_idx: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        Self { chars, char_to_idx }
    }

    /// Check if a character is in the vocabulary.
    pub fn contains(&self, ch: char) -> bool {
        self.char_to_idx.contains_key(&ch)
    }

    /// Vocabulary size.
    pub fn len(&self) -> usize {
        self.chars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Clamp a character to the nearest valid character (by code point).
    /// Returns space (' ') if completely out of range.
    pub fn clamp(&self, ch: char) -> char {
        if self.contains(ch) {
            return ch;
        }
        // Find nearest by code point distance.
        let code = ch as u32;
        self.chars
            .iter()
            .min_by_key(|&&c| (c as u32 as i64 - code as i64).unsigned_abs())
            .copied()
            .unwrap_or(' ')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_vocab_includes_ascii() {
        let v = GlyphVocab::game_default();
        assert!(v.contains(' '));
        assert!(v.contains('A'));
        assert!(v.contains('~'));
        assert!(v.contains('█')); // 0x2588 block element
        assert!(v.contains('┌')); // 0x250C box-drawing
    }

    #[test]
    fn vocab_clamp_returns_self_if_valid() {
        let v = GlyphVocab::game_default();
        assert_eq!(v.clamp('A'), 'A');
    }

    #[test]
    fn vocab_size_is_reasonable() {
        let v = GlyphVocab::game_default();
        // 95 printable ASCII + 128 box-drawing + 32 block elements = 255
        assert!(v.len() >= 200);
    }
}
