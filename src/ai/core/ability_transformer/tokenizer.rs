//! Rust port of `training/tokenizer.py`.
//!
//! Converts ability DSL text into integer token IDs using a fixed 252-token
//! vocabulary.  Numbers are bucketed into semantic ranges, ability names and
//! string literals are replaced with placeholder tokens.
//!
//! Token IDs match the Python tokenizer exactly so that weights exported from
//! the Python training pipeline can be used directly.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Vocabulary constants (must match training/tokenizer.py exactly)
// ---------------------------------------------------------------------------

/// Full vocabulary in ID order.  Index == token ID.
const VOCAB: &[&str] = &[
    // 0-7: special tokens
    "[PAD]", "[CLS]", "[MASK]", "[SEP]", "[UNK]", "[NAME]", "[STR]", "[TAG]",
    // 8-17: punctuation
    "{", "}", "(", ")", "[", "]", ":", ",", "+", "%",
    // 18-226: keywords (sorted)
    "ability", "absorb_to_heal", "ally", "ally_count_below", "and",
    "angle_deg", "apply_stacks", "arm_time", "armor", "attach",
    "attack_speed", "banish", "blind", "blink", "bounce_range",
    "bounces", "break_on_ability", "break_on_damage", "buff", "cap",
    "cast", "caster_attack_damage", "caster_buff_count", "caster_current_hp",
    "caster_has_status", "caster_hp_above", "caster_hp_below", "caster_max_hp",
    "caster_missing_hp", "caster_resource_above", "caster_resource_below",
    "caster_stacks", "cc", "chain", "chance", "channel", "charges", "charm",
    "circle", "clone", "clone_damage_percent", "command_summons", "cone",
    "confuse", "consume", "cooldown", "cooldown_modify", "cooldown_reduction",
    "cost", "crowd_control", "damage", "damage_modify", "damage_output",
    "dash", "death_mark", "debuff", "defense", "deliver", "detonate",
    "directed", "direction", "dispel", "duel", "duration", "else", "enemy",
    "enemy_count_below", "evolve_ability", "execute", "falloff", "fear",
    "for", "form", "global", "ground", "ground_target", "grounded", "heal",
    "hint", "hit_count_above", "hp_percent", "immunity", "in", "inner_radius",
    "instant", "is_blink", "knockback", "leash", "length", "lifesteal",
    "line", "link", "lookback_ms", "magic", "magic_resist", "max",
    "max_range", "max_stacks", "max_targets", "move_speed", "not",
    "obstacle", "on_ability_hit", "on_ally_death", "on_ally_low_hp",
    "on_arrival", "on_buff_applied", "on_cc_applied", "on_combat_end",
    "on_combat_start", "on_complete", "on_cooldown_end", "on_damage_dealt",
    "on_damage_taken", "on_death", "on_debuff_applied", "on_enter_zone",
    "on_heal", "on_hit", "on_hit_buff", "on_hp_above", "on_hp_below",
    "on_interval", "on_kill", "on_shield_break", "on_stack_reached", "or",
    "outer_radius", "overheal_shield", "passive", "physical", "pierce",
    "polymorph", "projectile", "projectile_block", "pull", "radius", "range",
    "recast", "recast_window", "recharge", "redirect", "reflect", "resurrect",
    "rewind", "ring", "root", "self", "self_aoe", "self_cast", "self_damage",
    "share_percent", "shield", "shield_steal", "silence", "slow", "speed",
    "spread", "stacking", "status_clone", "status_transfer", "stealth",
    "stun", "summon", "suppress", "swap", "swap_form", "target",
    "target_ally", "target_current_hp", "target_debuff_count",
    "target_distance_above", "target_distance_below", "target_enemy",
    "target_has_status", "target_has_tag", "target_hp_above",
    "target_hp_below", "target_is_banished", "target_is_charmed",
    "target_is_feared", "target_is_polymorphed", "target_is_rooted",
    "target_is_silenced", "target_is_slowed", "target_is_stealthed",
    "target_is_stunned", "target_is_taunted", "target_max_hp",
    "target_missing_hp", "target_stack_count", "target_stacks", "taunt",
    "template", "tether", "tick", "to_position", "to_target", "toggle",
    "trap", "trigger_radius", "true", "unstoppable", "utility", "vector",
    "when", "width", "x", "zone",
    // 227-246: number bucket tokens
    "NUM_0", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5",
    "NUM_6", "NUM_7", "NUM_8", "NUM_9", "NUM_10",
    "NUM_SMALL", "NUM_MED", "NUM_LARGE", "NUM_HUGE",
    "FRAC_TINY", "FRAC_LOW", "FRAC_MID", "FRAC_HIGH", "FRAC_MAX",
    // 247-251: duration bucket tokens
    "DUR_INSTANT", "DUR_SHORT", "DUR_MED", "DUR_LONG", "DUR_VLONG",
];

const PAD_ID: u32 = 0;
const CLS_ID: u32 = 1;
const UNK_ID: u32 = 4;
const NAME_ID: u32 = 5;
const STR_ID: u32 = 6;
const TAG_ID: u32 = 7;

const MAX_LENGTH: usize = 256;

// ---------------------------------------------------------------------------
// Number bucketing
// ---------------------------------------------------------------------------

fn bucket_number(value: f64) -> &'static str {
    // Exact integers 0-10
    if value == (value as i64) as f64 && value >= 0.0 && value <= 10.0 {
        return match value as i64 {
            0 => "NUM_0", 1 => "NUM_1", 2 => "NUM_2", 3 => "NUM_3",
            4 => "NUM_4", 5 => "NUM_5", 6 => "NUM_6", 7 => "NUM_7",
            8 => "NUM_8", 9 => "NUM_9", 10 => "NUM_10",
            _ => unreachable!(),
        };
    }
    // Fractions (0, 1) exclusive
    if value > 0.0 && value < 1.0 {
        if value <= 0.2 { return "FRAC_TINY"; }
        if value <= 0.4 { return "FRAC_LOW"; }
        if value <= 0.6 { return "FRAC_MID"; }
        if value <= 0.8 { return "FRAC_HIGH"; }
        return "FRAC_MAX";
    }
    let v = value.abs();
    if v <= 10.0 {
        return match v.round() as i64 {
            0 => "NUM_0", 1 => "NUM_1", 2 => "NUM_2", 3 => "NUM_3",
            4 => "NUM_4", 5 => "NUM_5", 6 => "NUM_6", 7 => "NUM_7",
            8 => "NUM_8", 9 => "NUM_9", 10 => "NUM_10",
            _ => "NUM_10",
        };
    }
    if v <= 50.0 { return "NUM_SMALL"; }
    if v <= 200.0 { return "NUM_MED"; }
    if v <= 1000.0 { return "NUM_LARGE"; }
    "NUM_HUGE"
}

fn bucket_duration_ms(ms: f64) -> &'static str {
    if ms < 200.0 { "DUR_INSTANT" }
    else if ms <= 2000.0 { "DUR_SHORT" }
    else if ms <= 8000.0 { "DUR_MED" }
    else if ms <= 30000.0 { "DUR_LONG" }
    else { "DUR_VLONG" }
}

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
// Parsing helpers
// ---------------------------------------------------------------------------

fn strip_comments(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for line in text.lines() {
        // Strip // and # comments
        let line = if let Some(idx) = line.find("//") {
            &line[..idx]
        } else {
            line
        };
        let line = if let Some(idx) = line.find('#') {
            &line[..idx]
        } else {
            line
        };
        result.push_str(line);
        result.push('\n');
    }
    result
}

/// Try to parse a number (integer or float) starting at `pos`.
/// Returns (value, end_pos) or None.
fn try_parse_number(text: &str, pos: usize) -> Option<(f64, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut end = pos;

    // Optional minus sign
    if end < len && bytes[end] == b'-' {
        end += 1;
    }

    // Must have at least one digit
    if end >= len || !bytes[end].is_ascii_digit() {
        return None;
    }

    // Integer part
    while end < len && bytes[end].is_ascii_digit() {
        end += 1;
    }

    // Optional decimal part
    if end < len && bytes[end] == b'.' && end + 1 < len && bytes[end + 1].is_ascii_digit() {
        end += 1; // skip dot
        while end < len && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }

    let val: f64 = text[pos..end].parse().ok()?;
    Some((val, end))
}

/// Try to parse a duration like "5s" or "1000ms" starting at `pos`.
/// Returns (milliseconds, end_pos) or None.
fn try_parse_duration(text: &str, pos: usize) -> Option<(f64, usize)> {
    let (val, num_end) = try_parse_number(text, pos)?;
    let rest = &text[num_end..];
    if rest.starts_with("ms") {
        // Check word boundary after "ms"
        let end = num_end + 2;
        if end >= text.len() || !text.as_bytes()[end].is_ascii_alphanumeric() {
            return Some((val, end));
        }
    }
    if rest.starts_with('s') {
        let end = num_end + 1;
        if end >= text.len() || !text.as_bytes()[end].is_ascii_alphanumeric() {
            return Some((val * 1000.0, end));
        }
    }
    None
}

/// Try to parse "N%" starting at `pos`.
/// Returns (N, end_pos including the %) or None.
fn try_parse_percent(text: &str, pos: usize) -> Option<(f64, usize)> {
    let (val, num_end) = try_parse_number(text, pos)?;
    if num_end < text.len() && text.as_bytes()[num_end] == b'%' {
        Some((val, num_end + 1))
    } else {
        None
    }
}

/// Try to parse a plain integer starting at `pos`.
fn try_parse_int(text: &str, pos: usize) -> Option<(i64, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut end = pos;
    if end >= len || !bytes[end].is_ascii_digit() {
        return None;
    }
    while end < len && bytes[end].is_ascii_digit() {
        end += 1;
    }
    let val: i64 = text[pos..end].parse().ok()?;
    Some((val, end))
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
