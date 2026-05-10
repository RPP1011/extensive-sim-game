//! Plan I — ability-grammar tree walker (slice I1).
//!
//! Per-axis sweep of the `.ability` grammar: for each header variant
//! and each verb, synthesize a minimal one-effect ability and assert
//! it parses + lowers. The 172-file LoL corpus covers historical
//! shapes; this walker covers the GRAMMAR — so a regression that
//! breaks `cooldown: 5s @ resolve` (Plan G CooldownPhase) won't slip
//! through merely because no corpus file uses that exact combo.
//!
//! Slice I1 covers two axes (header + verb) at default values for the
//! other axis; modifier slots (`in <shape>`, `for <duration>`, `[TAG:N]`,
//! `chance N%`, nested `{}`) are out of scope here and exercised by the
//! per-modifier tests in `ability_lower_wave_1_5.rs`. Slice I2 (deferred)
//! adds a coverage-matrix report.

use dsl_ast::ability_parser::parse_ability_file;
use dsl_compiler::ability_lower::lower_ability_decl;

/// One verb-axis sample. `args` is appended after the verb keyword to
/// build a single effect statement.
struct VerbProbe {
    name: &'static str,
    args: &'static str,
}

/// Verbs the lowering recognizes today (1490..2123 in `ability_lower.rs`).
/// Each entry uses the smallest legal arg shape so the synthesis stays
/// minimal — modifier-slot exhaustion lives elsewhere.
const VERBS: &[VerbProbe] = &[
    VerbProbe { name: "damage", args: "10" },
    VerbProbe { name: "heal", args: "10" },
    VerbProbe { name: "shield", args: "10 for 5s" },
    VerbProbe { name: "stun", args: "1500ms" },
    VerbProbe { name: "root", args: "1500ms" },
    VerbProbe { name: "silence", args: "1500ms" },
    VerbProbe { name: "fear", args: "1500ms" },
    VerbProbe { name: "taunt", args: "1500ms" },
    VerbProbe { name: "stealth", args: "for 2s" },
    VerbProbe { name: "charm", args: "1500ms" },
    VerbProbe { name: "grounded", args: "1500ms" },
    VerbProbe { name: "suppress", args: "1500ms" },
    VerbProbe { name: "reflect", args: "0.5 for 2s" },
    VerbProbe { name: "dash", args: "to_target" },
    VerbProbe { name: "blink", args: "5.0" },
    VerbProbe { name: "knockback", args: "2.0" },
    VerbProbe { name: "pull", args: "2.0" },
    VerbProbe { name: "execute", args: "0.25" },
    VerbProbe { name: "self_damage", args: "5" },
    VerbProbe { name: "lifesteal", args: "0.3 for 2s" },
    VerbProbe { name: "damage_modify", args: "0.5 for 2s" },
    VerbProbe { name: "slow", args: "0.3 for 2s" },
    VerbProbe { name: "buff", args: "move_speed 0.3 for 2s" },
    VerbProbe { name: "transfer_gold", args: "10" },
    VerbProbe { name: "modify_standing", args: "10" },
    VerbProbe { name: "disguise", args: "3 for 5s" },
    VerbProbe { name: "plant_belief", args: "0 bit 0" },
    VerbProbe { name: "observe", args: "0" },
    VerbProbe { name: "scry", args: "0 0" },
    VerbProbe { name: "reveal", args: "0" },
    VerbProbe { name: "decoy", args: "0 0" },
    VerbProbe { name: "erase_belief", args: "0 0" },
];

/// One header-axis sample. The body always uses a trivial `damage 10`
/// effect so failures point at the header arm, not the verb arm.
struct HeaderProbe {
    label: &'static str,
    line: &'static str,
}

const HEADERS: &[HeaderProbe] = &[
    HeaderProbe { label: "target_self", line: "target: self" },
    HeaderProbe { label: "target_direction", line: "target: direction" },
    HeaderProbe { label: "target_ground", line: "target: ground" },
    HeaderProbe { label: "range_8", line: "range: 8.0" },
    HeaderProbe { label: "cooldown_5s", line: "cooldown: 5s" },
    HeaderProbe { label: "cooldown_5s_at_cast", line: "cooldown: 5s @ cast" },
    HeaderProbe { label: "cooldown_5s_at_resolve", line: "cooldown: 5s @ resolve" },
    HeaderProbe { label: "cast_250ms", line: "cast: 250ms" },
    HeaderProbe { label: "hint_damage", line: "hint: damage" },
    HeaderProbe { label: "hint_crowd_control", line: "hint: crowd_control" },
    HeaderProbe { label: "hint_utility", line: "hint: utility" },
    HeaderProbe { label: "cost_50", line: "cost: 50" },
    HeaderProbe { label: "cost_50_mana", line: "cost: 50 mana" },
    HeaderProbe { label: "cost_25pct_hp", line: "cost: 25% hp" },
    HeaderProbe { label: "charges_3", line: "charges: 3" },
    HeaderProbe { label: "recharge_10s", line: "recharge: 10s" },
    HeaderProbe { label: "toggle", line: "toggle" },
    HeaderProbe { label: "recast_count_1", line: "recast: 1" },
    HeaderProbe { label: "recast_dur_4s", line: "recast: 4s" },
    HeaderProbe { label: "recast_window_4s", line: "recast_window: 4s" },
];

fn synthesize_verb_ability(verb: &VerbProbe) -> String {
    format!(
        "ability TreeWalkerVerbProbe_{name} {{\n    target: self\n    {name} {args}\n}}\n",
        name = verb.name,
        args = verb.args,
    )
}

fn synthesize_header_ability(header: &HeaderProbe) -> String {
    format!(
        "ability TreeWalkerHeaderProbe_{label} {{\n    {line}\n    damage 10\n}}\n",
        label = header.label,
        line = header.line,
    )
}

/// Run text through parse + lower. Returns the failing stage on error
/// so the test can group results by where the grammar broke down.
fn try_pipeline(text: &str) -> Result<(), String> {
    let parsed = parse_ability_file(text).map_err(|e| format!("parse: {e:?}"))?;
    let decl = parsed
        .abilities
        .first()
        .ok_or_else(|| "parse: no abilities in file".to_string())?;
    lower_ability_decl(decl).map_err(|e| format!("lower: {e:?}"))?;
    Ok(())
}

#[test]
fn walker_each_verb_parses_and_lowers() {
    let mut failures: Vec<(String, String)> = Vec::new();
    for verb in VERBS {
        let text = synthesize_verb_ability(verb);
        if let Err(stage) = try_pipeline(&text) {
            failures.push((verb.name.to_string(), format!("{stage}\n--- source ---\n{text}")));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} verbs failed parse/lower:\n\n{}",
        failures.len(),
        VERBS.len(),
        failures
            .iter()
            .map(|(v, msg)| format!("[{v}] {msg}"))
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
}

#[test]
fn walker_each_header_parses_and_lowers() {
    let mut failures: Vec<(String, String)> = Vec::new();
    for header in HEADERS {
        let text = synthesize_header_ability(header);
        if let Err(stage) = try_pipeline(&text) {
            failures.push((header.label.to_string(), format!("{stage}\n--- source ---\n{text}")));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} headers failed parse/lower:\n\n{}",
        failures.len(),
        HEADERS.len(),
        failures
            .iter()
            .map(|(h, msg)| format!("[{h}] {msg}"))
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
}

/// Negative pin: a synthesized ability that uses an unknown verb
/// keyword must surface a `LowerError` (NOT panic, NOT silently
/// succeed). Guards against future regressions where adding parser
/// generality drops the lowering catch-all.
#[test]
fn walker_unknown_verb_surfaces_lower_error() {
    let text = "ability TreeWalkerUnknownVerbProbe {\n    target: self\n    \
                noBoDyExpEctsThiS_verb_to_exist 10\n}\n";
    let parsed = parse_ability_file(text).expect("unknown verb parses (parser is generic)");
    let result = lower_ability_decl(&parsed.abilities[0]);
    assert!(
        result.is_err(),
        "unknown verb must fail lowering, got Ok({:?})",
        result.ok(),
    );
}
