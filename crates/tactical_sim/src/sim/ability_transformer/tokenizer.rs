//! Rust port of `training/tokenizer.py`.
//!
//! Converts ability DSL text into integer token IDs using a fixed 252-token
//! vocabulary.  Numbers are bucketed into semantic ranges, ability names and
//! string literals are replaced with placeholder tokens.
//!
//! Token IDs match the Python tokenizer exactly so that weights exported from
//! the Python training pipeline can be used directly.

use std::collections::HashMap;

use super::tokenizer_vocab::*;

// Re-export constants that downstream code uses directly.
pub use super::tokenizer_vocab::{CLS_ID, PAD_ID};

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Ability DSL tokenizer with a fixed 252-token vocabulary.
///
/// Token IDs are compatible with the Python `training/tokenizer.py`.
#[derive(Clone)]
pub struct AbilityTokenizer {
    tok2id: HashMap<String, u32>,
}

impl AbilityTokenizer {
    pub fn new() -> Self {
        let mut tok2id = HashMap::with_capacity(VOCAB.len());
        for (i, &tok) in VOCAB.iter().enumerate() {
            tok2id.insert(tok.to_string(), i as u32);
        }
        Self { tok2id }
    }

    /// Tokenize DSL text and return token IDs with [CLS] prepended.
    pub fn encode_with_cls(&self, text: &str) -> Vec<u32> {
        let token_strs = self.lex(text);
        let mut ids = Vec::with_capacity(token_strs.len() + 1);
        ids.push(CLS_ID);
        for tok in &token_strs {
            ids.push(*self.tok2id.get(tok.as_str()).unwrap_or(&UNK_ID));
        }
        ids.truncate(MAX_LENGTH);
        ids
    }

    /// Tokenize DSL text and return token IDs without [CLS].
    #[allow(dead_code)]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let token_strs = self.lex(text);
        let mut ids = Vec::with_capacity(token_strs.len());
        for tok in &token_strs {
            ids.push(*self.tok2id.get(tok.as_str()).unwrap_or(&UNK_ID));
        }
        ids.truncate(MAX_LENGTH);
        ids
    }

    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        VOCAB.len()
    }

    #[allow(dead_code)]
    pub fn pad_id(&self) -> u32 {
        PAD_ID
    }

    // -----------------------------------------------------------------------
    // Lexer — matches Python _lex() logic
    // -----------------------------------------------------------------------

    fn lex(&self, text: &str) -> Vec<String> {
        // Strip comments (// or # to end of line)
        let text = strip_comments(text);
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut pos = 0;
        let mut tokens = Vec::new();
        let mut expect_name = false;
        let mut bracket_depth: i32 = 0;

        while pos < len {
            // Skip whitespace
            if bytes[pos].is_ascii_whitespace() {
                pos += 1;
                continue;
            }

            // String literal: "..."
            if bytes[pos] == b'"' {
                if let Some(end) = text[pos + 1..].find('"') {
                    pos = pos + 1 + end + 1;
                    tokens.push("[STR]".to_string());
                    expect_name = false;
                    continue;
                }
            }

            // Duration: digits followed by "ms" or "s"
            if let Some((ms, end)) = try_parse_duration(&text, pos) {
                tokens.push(bucket_duration_ms(ms).to_string());
                pos = end;
                expect_name = false;
                continue;
            }

            // Percentage: digits followed by %
            if let Some((val, end)) = try_parse_percent(&text, pos) {
                tokens.push(bucket_number(val / 100.0).to_string());
                tokens.push("%".to_string());
                pos = end;
                expect_name = false;
                continue;
            }

            // Multiplier: x followed by digits (at word boundary)
            if bytes[pos] == b'x' && pos + 1 < len && bytes[pos + 1].is_ascii_digit() {
                // Check it's at a word boundary (start or preceded by non-alnum)
                let at_boundary = pos == 0 || !bytes[pos - 1].is_ascii_alphanumeric();
                if at_boundary {
                    if let Some((n, end)) = try_parse_int(&text, pos + 1) {
                        // Check end is a word boundary
                        let end_boundary = end >= len || !bytes[end].is_ascii_alphanumeric();
                        if end_boundary {
                            tokens.push("x".to_string());
                            tokens.push(bucket_number(n as f64).to_string());
                            pos = end;
                            expect_name = false;
                            continue;
                        }
                    }
                }
            }

            // Punctuation
            if b"{}()[]:,+%".contains(&bytes[pos]) {
                let ch = bytes[pos] as char;
                if ch == '[' { bracket_depth += 1; }
                if ch == ']' { bracket_depth = (bracket_depth - 1).max(0); }
                tokens.push(ch.to_string());
                pos += 1;
                continue;
            }

            // Number (plain, after duration/percent checks)
            if bytes[pos] == b'-' || bytes[pos].is_ascii_digit() {
                if let Some((val, end)) = try_parse_number(&text, pos) {
                    tokens.push(bucket_number(val).to_string());
                    pos = end;
                    expect_name = false;
                    continue;
                }
            }

            // Identifier or keyword
            if bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_' {
                let start = pos;
                while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                    pos += 1;
                }
                let word = &text[start..pos];

                if expect_name {
                    tokens.push("[NAME]".to_string());
                    expect_name = false;
                } else if bracket_depth > 0 && !self.tok2id.contains_key(word) {
                    tokens.push("[TAG]".to_string());
                } else if self.tok2id.contains_key(word) {
                    tokens.push(word.to_string());
                    if word == "ability" || word == "passive" {
                        expect_name = true;
                    }
                } else {
                    tokens.push("[UNK]".to_string());
                }
                continue;
            }

            // Skip unknown character
            pos += 1;
        }

        tokens
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size() {
        let tok = AbilityTokenizer::new();
        assert_eq!(tok.vocab_size(), 252);
    }

    #[test]
    fn test_special_token_ids() {
        let tok = AbilityTokenizer::new();
        assert_eq!(*tok.tok2id.get("[PAD]").unwrap(), 0);
        assert_eq!(*tok.tok2id.get("[CLS]").unwrap(), 1);
        assert_eq!(*tok.tok2id.get("[MASK]").unwrap(), 2);
        assert_eq!(*tok.tok2id.get("[UNK]").unwrap(), 4);
    }

    #[test]
    fn test_number_bucketing() {
        assert_eq!(bucket_number(0.0), "NUM_0");
        assert_eq!(bucket_number(5.0), "NUM_5");
        assert_eq!(bucket_number(10.0), "NUM_10");
        assert_eq!(bucket_number(25.0), "NUM_SMALL");
        assert_eq!(bucket_number(100.0), "NUM_MED");
        assert_eq!(bucket_number(500.0), "NUM_LARGE");
        assert_eq!(bucket_number(5000.0), "NUM_HUGE");
        assert_eq!(bucket_number(0.1), "FRAC_TINY");
        assert_eq!(bucket_number(0.3), "FRAC_LOW");
        assert_eq!(bucket_number(0.5), "FRAC_MID");
        assert_eq!(bucket_number(0.7), "FRAC_HIGH");
        assert_eq!(bucket_number(0.9), "FRAC_MAX");
    }

    #[test]
    fn test_duration_bucketing() {
        assert_eq!(bucket_duration_ms(50.0), "DUR_INSTANT");
        assert_eq!(bucket_duration_ms(500.0), "DUR_SHORT");
        assert_eq!(bucket_duration_ms(5000.0), "DUR_MED");
        assert_eq!(bucket_duration_ms(15000.0), "DUR_LONG");
        assert_eq!(bucket_duration_ms(60000.0), "DUR_VLONG");
    }

    #[test]
    fn test_encode_fireball() {
        let tok = AbilityTokenizer::new();
        let dsl = r#"ability Fireball {
            target: enemy, range: 5.0
            cooldown: 5s, cast: 300ms
            hint: damage
            deliver projectile { speed: 8.0, width: 0.3 } {
                on_hit { damage 55 [FIRE: 60] }
            }
        }"#;
        let ids = tok.encode_with_cls(dsl);
        // Should start with [CLS]
        assert_eq!(ids[0], CLS_ID);
        // "ability" is token 18
        assert_eq!(ids[1], 18);
        // [NAME] is token 5
        assert_eq!(ids[2], NAME_ID);
        // "{" is token 8
        assert_eq!(ids[3], 8);
        assert!(ids.len() > 10);
        println!("Fireball tokens: {:?}", ids);
    }

    #[test]
    fn test_matches_python_fireball() {
        // Known Python output for a specific fireball encoding
        let tok = AbilityTokenizer::new();
        let dsl = r#"ability Fireball {
            target: enemy, range: 5.0
            cooldown: 5s, cast: 300ms
            hint: damage
            deliver projectile { speed: 8.0, width: 0.3 } {
                on_hit { damage 55 [FIRE: 60] }
            }
        }"#;
        let ids = tok.encode_with_cls(dsl);
        // From Python tokenizer:
        let expected: Vec<u32> = vec![
            1, 18, 5, 8, 185, 14, 83, 15, 155, 14, 232,
            63, 14, 249, 15, 38, 14, 248, 96, 14, 68,
            75, 151, 8, 174, 14, 235, 15, 224, 14, 243, 9, 8,
            136, 8, 68, 239, 12, 7, 14, 239, 13, 9, 9, 9,
        ];
        assert_eq!(ids, expected, "Token IDs don't match Python output.\nGot:      {:?}\nExpected: {:?}", ids, expected);
    }
}
