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

/// One modifier-axis sample. `body` is a complete effect statement
/// (verb + args + modifier under test). The body is paired with the
/// verb that legally accepts the modifier — `chance` works on most,
/// `stacking` requires a duration-bearing verb, `+ N% stat_ref` needs
/// a damage-bearing verb, etc. Sourced from real corpus shapes (see
/// assets/ability_test/duel_abilities/* and dsl_coverage/Aegis.ability).
struct ModifierProbe {
    label: &'static str,
    body: &'static str,
}

const MODIFIERS: &[ModifierProbe] = &[
    // Slot 1 — area shape.
    ModifierProbe { label: "area_circle", body: "damage 10 in circle(2.5)" },
    ModifierProbe { label: "area_cone", body: "damage 10 in cone(5.0, 45)" },
    ModifierProbe { label: "area_line", body: "damage 10 in line(5.0, 1.0)" },
    // Slot 2 — power tags.
    ModifierProbe { label: "tag_physical", body: "damage 10 [PHYSICAL: 50]" },
    ModifierProbe { label: "tag_magical", body: "damage 10 [MAGICAL: 50]" },
    ModifierProbe { label: "tag_utility", body: "damage 10 [UTILITY: 50]" },
    ModifierProbe { label: "tag_crowd_control", body: "stun 1s [CROWD_CONTROL: 100]" },
    ModifierProbe { label: "tag_heal", body: "heal 10 [HEAL: 50]" },
    ModifierProbe { label: "tag_defense", body: "shield 10 for 5s [DEFENSE: 100]" },
    // Slot 3 — chance.
    ModifierProbe { label: "chance_50", body: "stun 1s chance 50%" },
    ModifierProbe { label: "chance_100", body: "damage 10 chance 100%" },
    // Slot 4 — stacking.
    ModifierProbe { label: "stacking_refresh", body: "damage_modify 0.5 for 5s stacking refresh" },
    ModifierProbe { label: "stacking_stack", body: "slow 0.3 for 2s stacking stack" },
    ModifierProbe { label: "stacking_extend", body: "lifesteal 0.4 for 6s stacking extend" },
    // Slot 5 — scaling (`+ N% stat_ref`).
    ModifierProbe { label: "scaling_AP", body: "damage 10 + 30% AP" },
    ModifierProbe { label: "scaling_AD", body: "damage 10 + 50% AD" },
    ModifierProbe { label: "scaling_hp", body: "damage 10 + 25% hp" },
    ModifierProbe { label: "scaling_max_hp", body: "damage 10 + 10% max_hp" },
    // Slot 6 — lifetime.
    ModifierProbe { label: "lifetime_break_on_damage", body: "stealth for 3s break_on_damage" },
    // Slot 7 — `when (predicate)`.
    ModifierProbe { label: "when_simple", body: "damage 10 when (target.hp > 0.5)" },
    // Slot 8 — combinations (mirrors corpus Aegis.ability shape).
    ModifierProbe { label: "combo_stack_tag_scaling",
        body: "damage_modify 0.6 for 8s stacking refresh [DEFENSE: 100] + 15% armor" },
];

fn synthesize_modifier_ability(probe: &ModifierProbe) -> String {
    format!(
        "ability TreeWalkerModifierProbe_{label} {{\n    target: self\n    {body}\n}}\n",
        label = probe.label,
        body = probe.body,
    )
}

#[test]
fn walker_each_modifier_parses_and_lowers() {
    let mut failures: Vec<(String, String)> = Vec::new();
    for probe in MODIFIERS {
        let text = synthesize_modifier_ability(probe);
        if let Err(stage) = try_pipeline(&text) {
            failures.push((probe.label.to_string(), format!("{stage}\n--- source ---\n{text}")));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} modifiers failed parse/lower:\n\n{}",
        failures.len(),
        MODIFIERS.len(),
        failures
            .iter()
            .map(|(m, msg)| format!("[{m}] {msg}"))
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
}

/// Multi-effect bodies: an ability with N stacked top-level effect
/// statements. Real-corpus pattern (e.g. Aatrox.WorldEnder pairs heal
/// + fear + slow + buff under one ability). Catches lowering arms
/// that assume single-effect ability bodies.
#[test]
fn walker_multi_effect_body_parses_and_lowers() {
    let text = "ability TreeWalkerMultiEffect {\n\
        \x20   target: self\n\
        \x20   heal 10\n\
        \x20   damage 5\n\
        \x20   shield 10 for 5s\n\
        \x20   stun 1s\n\
        \x20   slow 0.3 for 2s\n\
    }\n";
    try_pipeline(text).unwrap_or_else(|e| panic!("multi-effect body must lower: {e}"));
}

/// One program-shape sample. `body` is everything between the
/// outer `ability X { … }` braces. Covers Plan G's `cast { } effect { }`
/// shape and `deliver <method>` blocks — structural surfaces that
/// the verb / header / modifier sweeps don't reach.
struct ProgramShapeProbe {
    label: &'static str,
    body: &'static str,
}

const PROGRAM_SHAPES: &[ProgramShapeProbe] = &[
    // Plan G — cast { duration } effect { ... } minimal shape.
    ProgramShapeProbe {
        label: "cast_effect_minimal",
        body: "target: enemy\n    range: 30\n    cast { duration: 3t }\n    effect { damage 25 }",
    },
    // Plan G — cast block with telegraph + interrupts.
    ProgramShapeProbe {
        label: "cast_with_telegraph_and_interrupts",
        body: "target: enemy\n    \
               cast { duration: 3t; telegraph: line(self.pos, target.pos, width: 2); interrupts: standard }\n    \
               effect { damage 25 }",
    },
    // Plan G — explicit interrupt subset.
    ProgramShapeProbe {
        label: "cast_interrupts_subset",
        body: "target: enemy\n    \
               cast { duration: 1t; interrupts: { damage, stun } }\n    \
               effect { damage 1 }",
    },
    // deliver projectile — corpus-canonical (Aatrox.InfernalChains shape).
    ProgramShapeProbe {
        label: "deliver_projectile_with_hook",
        body: "target: direction\n    range: 8.0\n    \
               deliver projectile { speed: 18.0 } {\n        \
                   on_hit {\n            damage 7\n            slow 0.3 for 2s\n        }\n    }",
    },
    // deliver channel — multi-tick channel (Akshan shape).
    ProgramShapeProbe {
        label: "deliver_channel_with_on_tick",
        body: "target: ground\n    range: 10.0\n    \
               deliver channel { duration: 2s, tick: 500ms } {\n        \
                   on_tick {\n            heal 5\n        }\n    }",
    },
];

fn synthesize_program_shape_ability(probe: &ProgramShapeProbe) -> String {
    format!(
        "ability TreeWalkerProgramShape_{label} {{\n    {body}\n}}\n",
        label = probe.label,
        body = probe.body,
    )
}

#[test]
fn walker_each_program_shape_parses_and_lowers() {
    let mut failures: Vec<(String, String)> = Vec::new();
    for probe in PROGRAM_SHAPES {
        let text = synthesize_program_shape_ability(probe);
        if let Err(stage) = try_pipeline(&text) {
            failures.push((probe.label.to_string(), format!("{stage}\n--- source ---\n{text}")));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} program shapes failed parse/lower:\n\n{}",
        failures.len(),
        PROGRAM_SHAPES.len(),
        failures
            .iter()
            .map(|(p, msg)| format!("[{p}] {msg}"))
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
}

/// Negative pin: `morph { } into <Other>` parses (the surface is in
/// the AST since Wave 1.4 for spec coverage) but lowering surfaces
/// `MorphBlockNotImplemented`. Documents the parse-vs-lower boundary
/// — when morph lowering lands, this test will start failing and
/// signal the author to flip the assertion to `is_ok()` and add a
/// positive-shape probe.
#[test]
fn walker_morph_block_surfaces_not_implemented() {
    let text = "ability TreeWalkerMorphBlock {\n    \
                target: self\n    \
                morph { damage 5 } into OtherForm\n}\n";
    let parsed = parse_ability_file(text).expect("morph block parses");
    let err = lower_ability_decl(&parsed.abilities[0])
        .expect_err("morph block must surface NotImplemented today");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("MorphBlockNotImplemented"),
        "expected MorphBlockNotImplemented, got: {msg}",
    );
}

/// One Plan G interrupt-set sample. Each shape goes inside an
/// otherwise-minimal cast block; all should parse cleanly. Sourced
/// from the parser-side test fixtures in
/// `crates/dsl_ast/tests/ability_parser_plan_g.rs`.
struct InterruptSetProbe {
    label: &'static str,
    interrupts: &'static str,
}

const INTERRUPT_SETS: &[InterruptSetProbe] = &[
    InterruptSetProbe { label: "standard", interrupts: "interrupts: standard" },
    InterruptSetProbe { label: "none", interrupts: "interrupts: none" },
    InterruptSetProbe { label: "explicit_subset", interrupts: "interrupts: { damage, stun }" },
    InterruptSetProbe { label: "standard_plus_set", interrupts: "interrupts: standard + { movement }" },
    InterruptSetProbe { label: "standard_minus_set", interrupts: "interrupts: standard - { damage }" },
];

#[test]
fn walker_each_interrupt_set_parses_and_lowers() {
    let mut failures: Vec<(String, String)> = Vec::new();
    for probe in INTERRUPT_SETS {
        let text = format!(
            "ability TreeWalkerInterruptSet_{label} {{\n    \
             target: enemy\n    \
             cast {{ duration: 2t; {interrupts} }}\n    \
             effect {{ damage 5 }}\n}}\n",
            label = probe.label,
            interrupts = probe.interrupts,
        );
        if let Err(stage) = try_pipeline(&text) {
            failures.push((probe.label.to_string(), format!("{stage}\n--- source ---\n{text}")));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} interrupt-set shapes failed parse/lower:\n\n{}",
        failures.len(),
        INTERRUPT_SETS.len(),
        failures
            .iter()
            .map(|(p, msg)| format!("[{p}] {msg}"))
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
}

/// Multi-step program: `cast { } effect { } cast { } effect { }`
/// chains. Plan G's program shape allows any interleaving; the
/// minimal verb sweep covered the single cast+effect pair, this
/// covers the multi-stage shape (two casts back-to-back, separated
/// by a per-stage effect). Catches lowering arms that assume
/// at most one Cast step per program.
#[test]
fn walker_multi_step_cast_program_parses_and_lowers() {
    let text = "ability TreeWalkerMultiCastProgram {\n    \
                target: enemy\n    \
                cast { duration: 2t }\n    \
                effect { damage 10 }\n    \
                cast { duration: 1t }\n    \
                effect { heal 5 }\n}\n";
    try_pipeline(text)
        .unwrap_or_else(|e| panic!("multi-step cast/effect program must lower: {e}\n--- source ---\n{text}"));
}

/// Negative pin: top-level `passive <Name> { ... }` blocks parse
/// (the parser has had this since Wave 1.1 for spec coverage of the
/// 172-file LoL corpus) but the lowerer surfaces
/// `PassiveBlockNotImplemented`. Same kind of grammar-vs-lowering
/// boundary documentation as the morph pin: when passive-trigger
/// dispatch lands, this test trips and the author flips it to a
/// positive shape probe.
#[test]
fn walker_passive_block_surfaces_not_implemented() {
    let text = "passive TreeWalkerPassiveBlock {\n    \
                trigger: periodic(5s)\n    \
                cooldown: 5s\n\n    \
                heal 5\n}\n";
    let parsed = parse_ability_file(text).expect("passive block parses");
    // The passive ends up in `parsed.passives`, not `parsed.abilities`.
    // Lowering returns Ok with the passive in `skipped` (soft fail —
    // the file is otherwise valid; lowering just skips passives).
    let outcome = dsl_compiler::ability_lower::lower_ability_file(&parsed)
        .expect("file lowering returns Ok with passives in skipped");
    assert!(
        outcome.programs.is_empty(),
        "passive-only file should produce zero ability programs; got {} programs",
        outcome.programs.len(),
    );
    let msg = format!("{:?}", outcome.skipped);
    assert!(
        msg.contains("PassiveBlockNotImplemented"),
        "expected PassiveBlockNotImplemented in skipped, got: {msg}",
    );
}

/// Negative pin: `self.hp` in scaling position parse
/// (the `parse_stat_ref` cursor accepts dotted segments) but lower
/// rejects them — `ScalingStatRef::parse` only knows flat names like
/// `AP`, `AD`, `hp`. Documents the grammar-vs-lowering boundary so a
/// future fix to either side is a deliberate decision rather than a
/// silent surface-area shift.
#[test]
fn walker_dotted_scaling_ref_surfaces_lower_error() {
    let text = "ability TreeWalkerDottedScaling {\n    target: self\n    damage 10 + 25% self.hp\n}\n";
    let parsed = parse_ability_file(text).expect("dotted ref parses");
    let err = lower_ability_decl(&parsed.abilities[0])
        .expect_err("dotted ref must fail lowering today");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("UnknownStatRef"),
        "expected UnknownStatRef, got: {msg}",
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
