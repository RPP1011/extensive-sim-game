//! Vocabulary constants and number bucketing for the ability tokenizer.
//!
//! Extracted from `tokenizer.rs` to keep files under 500 lines.
//! Must match `training/tokenizer.py` exactly.

// ---------------------------------------------------------------------------
// Vocabulary constants (must match training/tokenizer.py exactly)
// ---------------------------------------------------------------------------

/// Full vocabulary in ID order.  Index == token ID.
pub const VOCAB: &[&str] = &[
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

pub const PAD_ID: u32 = 0;
pub const CLS_ID: u32 = 1;
pub const UNK_ID: u32 = 4;
pub const NAME_ID: u32 = 5;
pub const STR_ID: u32 = 6;
pub const TAG_ID: u32 = 7;

pub const MAX_LENGTH: usize = 256;

// ---------------------------------------------------------------------------
// Number bucketing
// ---------------------------------------------------------------------------

pub fn bucket_number(value: f64) -> &'static str {
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

pub fn bucket_duration_ms(ms: f64) -> &'static str {
    if ms < 200.0 { "DUR_INSTANT" }
    else if ms <= 2000.0 { "DUR_SHORT" }
    else if ms <= 8000.0 { "DUR_MED" }
    else if ms <= 30000.0 { "DUR_LONG" }
    else { "DUR_VLONG" }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Strip `//` and `#` comments from DSL text.
pub fn strip_comments(text: &str) -> String {
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
pub fn try_parse_number(text: &str, pos: usize) -> Option<(f64, usize)> {
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
pub fn try_parse_duration(text: &str, pos: usize) -> Option<(f64, usize)> {
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
pub fn try_parse_percent(text: &str, pos: usize) -> Option<(f64, usize)> {
    let (val, num_end) = try_parse_number(text, pos)?;
    if num_end < text.len() && text.as_bytes()[num_end] == b'%' {
        Some((val, num_end + 1))
    } else {
        None
    }
}

/// Try to parse a plain integer starting at `pos`.
pub fn try_parse_int(text: &str, pos: usize) -> Option<(i64, usize)> {
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
