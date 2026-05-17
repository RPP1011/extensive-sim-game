//! Fuzz-scale random ability generator + parse/lower validator.
//!
//! ## Purpose
//!
//! The per-variant walkers (`ability_grammar_walker.rs` ×2,
//! `grammar_tree_walker.rs`) hand-enumerate every enum variant in
//! isolation. They prove "each variant works alone". They do NOT
//! prove that random combinations of variants stacked together
//! across headers + modifier slots + nested effects + scalings + tags
//! survive emit → parse → lower.
//!
//! This generator closes that gap. A deterministic xorshift64 PRNG
//! walks the grammar tree top-down, picking each axis randomly from
//! its valid pool, and synthesises 1024 random `AbilityDecl` values
//! per run. Each one goes through:
//!
//!   1. `emit_ability_file_single` (AST → source)
//!   2. `parse_ability_file`       (source → AST round-trip)
//!   3. `lower_ability_decl`       (AST → AbilityProgram)
//!
//! Any failure surfaces the offending random ability's source +
//! the failure step. Coverage counters track how many times each
//! variant got exercised, and the test prints a histogram so the
//! report visibly proves combinatorial reach.
//!
//! ## Determinism
//!
//! Single fixed seed `0xA811_17_FEED_5EED` — the test is the same
//! every run, so any future regression points at a specific source.
//!
//! ## Scope
//!
//! Generates only the grammar surface that both the parser and the
//! lowering accept. The deferred-block fields (`deliver`, `morph`,
//! `program`, `instantiates`) stay at `None` per the walker
//! contract.

use dsl_ast::ability_emit::emit_ability_file_single;
use dsl_ast::ability_parser::parse_ability_file;
use dsl_ast::ast::*;
use dsl_compiler::ability_lower::lower_ability_decl;
use std::collections::BTreeMap;

const SEED: u64 = 0xA811_17_FEED_5EED;
const N_ABILITIES: usize = 1024;
/// Probability (out of 100) each modifier slot is enabled per effect.
const MODIFIER_P: u8 = 50;
/// Max number of effect statements per ability (1..=MAX_EFFECTS).
const MAX_EFFECTS: u8 = 3;
/// Max number of nested effects per parent (0..=MAX_NESTED).
const MAX_NESTED: u8 = 2;
/// Max number of scalings per effect (lowering caps at
/// MAX_SCALINGS_PER_EFFECT = 2 — see `LowerError::ScalingBudgetExceeded`).
const MAX_SCALINGS: u8 = 2;
/// Max number of tags per effect.
const MAX_TAGS: u8 = 2;

// ---------- xorshift64 PRNG ----------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_F00D } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    fn pct(&mut self) -> u8 {
        (self.next_u32() % 100) as u8
    }
    fn range(&mut self, hi_exclusive: u32) -> u32 {
        self.next_u32() % hi_exclusive
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.range(items.len() as u32) as usize]
    }
    fn pick_str(&mut self, items: &[&'static str]) -> &'static str {
        items[self.range(items.len() as u32) as usize]
    }
    fn pick_bool(&mut self, p_yes: u8) -> bool {
        self.pct() < p_yes
    }
    fn f32_unit(&mut self) -> f32 {
        // 0.0..1.0
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

fn span() -> Span {
    Span { start: 0, end: 0 }
}

fn dur(ms: u32) -> Duration {
    Duration { millis: ms }
}

// ---------- Variant pools ----------

const TARGETS: &[TargetMode] = &[
    TargetMode::Enemy,
    TargetMode::Self_,
    TargetMode::Ally,
    TargetMode::SelfAoe,
    TargetMode::Ground,
    TargetMode::Direction,
    TargetMode::Vector,
    TargetMode::Global,
];

// HintName::Economic is grammar-valid but lowering-reserved
// (`LowerError::HintReserved` at `ability_lower.rs:2283`). Omit so
// the generator doesn't fail lowering on a known reserved variant.
const HINTS: &[HintName] = &[
    HintName::Damage,
    HintName::Defense,
    HintName::CrowdControl,
    HintName::Utility,
    HintName::Heal,
    HintName::Buff,
];

const COST_RESOURCES: &[CostResource] = &[
    CostResource::Mana,
    CostResource::Stamina,
    CostResource::Hp,
    CostResource::Gold,
];

const COOLDOWN_PHASES: &[Option<CooldownPhase>] = &[
    None,
    Some(CooldownPhase::Cast),
    Some(CooldownPhase::Resolve),
    Some(CooldownPhase::Interrupt),
];

const STACKINGS: &[StackingMode] = &[
    StackingMode::Refresh,
    StackingMode::Stack,
    StackingMode::Extend,
];

/// Verb pools per arg shape. Each entry: (verb name, arg-builder).
/// Pulled from `crates/dsl_compiler/tests/grammar_tree_walker.rs`'s
/// VERBS list — the canonical "lowering accepts this verb with this
/// minimal arg shape" set. Listed by arg shape so the generator can
/// produce a syntactically-valid effect for each verb without
/// inventing arg permutations the lowering would reject.
const VERBS_NUMBER_ONLY: &[&str] = &[
    "damage",
    "heal",
    "blink",
    "knockback",
    "pull",
    "execute",
    "self_damage",
    "transfer_gold",
    "modify_standing",
];

const VERBS_DURATION_ONLY: &[&str] = &[
    "stun",
    "root",
    "silence",
    "fear",
    "taunt",
    "charm",
    "grounded",
    "suppress",
];

const VERBS_SCALAR_FOR_DURATION: &[&str] = &[
    "shield",     // shield N for Ds
    "reflect",    // reflect F for Ds (F = 0..1 fraction)
    "lifesteal",  // lifesteal F for Ds
    "damage_modify",
    "slow",
];

// Power-tag whitelist matching `engine::ability::AbilityTag` —
// non-listed names fail the lowering UnknownTag gate.
const TAG_NAMES: &[&str] = &[
    "PHYSICAL",
    "MAGICAL",
    "CROWD_CONTROL",
    "HEAL",
    "DEFENSE",
    "UTILITY",
];

// Scaling stat refs accepted by the lowering. `LowerError::
// UnknownStatRef` lists the valid set:
// `attack_damage/AD, ability_power/AP, max_hp/MaxHP, hp/HP, armor,
// magic_resist/MR, move_speed, mana`. The `self.*` form parses but
// is not in the registered set.
const STAT_REFS: &[&str] = &["AP", "AD", "MaxHP", "HP", "armor", "MR", "move_speed", "mana"];

const AREA_SHAPES: &[(&str, u8)] = &[
    // (shape name, arg count). Lowering accepts a few; circle is the
    // workhorse. Others tested in the per-shape walker.
    ("circle", 1),
];

// ---------- Coverage tracking ----------

#[derive(Default)]
struct Coverage {
    /// "axis:label" → count of generated abilities that hit it.
    counts: BTreeMap<String, u32>,
}

impl Coverage {
    fn hit<S: Into<String>>(&mut self, key: S) {
        *self.counts.entry(key.into()).or_insert(0) += 1;
    }
    fn print_report(&self) {
        println!("---- random ability generator coverage report ----");
        let mut entries: Vec<_> = self.counts.iter().collect();
        entries.sort_by_key(|(k, _)| (*k).clone());
        for (k, v) in entries {
            println!("  {k:40}  {v:5}");
        }
        println!("--------------------------------------------------");
    }
}

// ---------- Generators ----------

fn gen_headers(rng: &mut Rng, cov: &mut Coverage) -> Vec<AbilityHeader> {
    let mut out = Vec::new();
    // Target + range are always present so lowering doesn't fail on
    // a missing dispatch shape. The optional headers below are gated.
    let target = rng.pick(TARGETS);
    out.push(AbilityHeader::Target(target));
    cov.hit(format!("target:{target:?}"));

    let range = 100.0 + rng.f32_unit() * 700.0;
    out.push(AbilityHeader::Range(range));
    cov.hit("header:range");

    if rng.pick_bool(70) {
        // Cooldown — always with a phase variant so coverage stays
        // balanced across all 4 phase options.
        let phase = rng.pick(COOLDOWN_PHASES);
        let ms = 500 + rng.range(20_000);
        out.push(AbilityHeader::Cooldown(dur(ms), phase));
        cov.hit(format!("cooldown_phase:{phase:?}"));
    }
    if rng.pick_bool(40) {
        out.push(AbilityHeader::Cast(dur(100 + rng.range(2000))));
        cov.hit("header:cast");
    }
    if rng.pick_bool(50) {
        let h = rng.pick(HINTS);
        out.push(AbilityHeader::Hint(h));
        cov.hit(format!("hint:{h:?}"));
    }
    if rng.pick_bool(40) {
        let resource = rng.pick(COST_RESOURCES);
        let amount = if rng.pick_bool(50) {
            CostAmount::Flat(10.0 + rng.f32_unit() * 90.0)
        } else {
            CostAmount::PercentOfMax(5.0 + rng.f32_unit() * 25.0)
        };
        cov.hit(format!("cost_resource:{resource:?}"));
        cov.hit(match amount {
            CostAmount::Flat(_) => "cost_amount:Flat",
            CostAmount::PercentOfMax(_) => "cost_amount:PercentOfMax",
        });
        out.push(AbilityHeader::Cost(CostSpec { resource, amount, span: span() }));
    }
    if rng.pick_bool(20) {
        out.push(AbilityHeader::Charges(1 + rng.range(5)));
        cov.hit("header:charges");
    }
    if rng.pick_bool(15) {
        out.push(AbilityHeader::Recharge(dur(2000 + rng.range(15_000))));
        cov.hit("header:recharge");
    }
    if rng.pick_bool(10) {
        out.push(AbilityHeader::Toggle);
        cov.hit("header:toggle");
    }
    if rng.pick_bool(15) {
        let recast = if rng.pick_bool(50) {
            let v = RecastValue::Count(1 + rng.range(4));
            cov.hit("recast:Count");
            v
        } else {
            let v = RecastValue::Duration(dur(1000 + rng.range(8000)));
            cov.hit("recast:Duration");
            v
        };
        out.push(AbilityHeader::Recast(recast));
        // Recast pairs naturally with a window — emit it sometimes.
        if rng.pick_bool(60) {
            out.push(AbilityHeader::RecastWindow(dur(2000 + rng.range(8000))));
            cov.hit("header:recast_window");
        }
    }
    out
}

fn gen_effect(rng: &mut Rng, cov: &mut Coverage, depth: u8) -> EffectStmt {
    // Pick a verb shape + an arg shape that the lowering will accept.
    // `duration_modifier_state` records what the modifier slot should
    // do for this verb:
    //   Set    → the `for <duration>` slot MUST be set (and the
    //            positional duration MUST NOT be present)
    //   Skip   → the `for <duration>` slot MUST NOT be set (the
    //            positional duration is already in args)
    //   Free   → either is fine (verb doesn't consume duration)
    // The mismatch case `extract_duration` rejects is "both set".
    //
    // **Nested-effect constraint**: per `LowerError::
    // NestedModifierDropped`, nested stmts CANNOT carry any modifier
    // slot (chance/stacking/lifetime/scaling/in-shape/tags/
    // for-duration/when/recursive-nested). So at depth > 0 we force
    // the positional-duration form (DurMod::Skip) for the CC/scalar
    // verbs so they don't need the `for` modifier; and we restrict
    // the modifier-slot generation below.
    #[derive(Copy, Clone)]
    enum DurMod { Set, Skip, Free }
    let allow_modifier_set = depth == 0;
    let bucket = rng.range(3);
    let (verb, args, dur_state): (&str, Vec<EffectArg>, DurMod) = match bucket {
        0 => {
            let v = rng.pick_str(VERBS_NUMBER_ONLY);
            cov.hit(format!("verb:{v}"));
            (v, vec![EffectArg::Number(5.0 + rng.f32_unit() * 95.0)], DurMod::Free)
        }
        1 => {
            // CC verbs accept EITHER `stun 1500ms` (positional dur) OR
            // `stun for 1500ms` (modifier dur), never both. At depth 0
            // pick randomly; at depth > 0 force positional to avoid
            // the `for` modifier that nested rejects.
            let v = rng.pick_str(VERBS_DURATION_ONLY);
            cov.hit(format!("verb:{v}"));
            if allow_modifier_set && rng.pick_bool(50) {
                (v, vec![], DurMod::Set)
            } else {
                (
                    v,
                    vec![EffectArg::Duration(dur(500 + rng.range(3000)))],
                    DurMod::Skip,
                )
            }
        }
        _ => {
            // VERBS_SCALAR_FOR_DURATION need the `for <duration>`
            // modifier; nested effects can't carry it, so at depth > 0
            // fall back to a VERBS_NUMBER_ONLY shape instead.
            if !allow_modifier_set {
                let v = rng.pick_str(VERBS_NUMBER_ONLY);
                cov.hit(format!("verb:{v}"));
                (v, vec![EffectArg::Number(5.0 + rng.f32_unit() * 95.0)], DurMod::Free)
            } else {
                let v = rng.pick_str(VERBS_SCALAR_FOR_DURATION);
                cov.hit(format!("verb:{v}"));
                let scalar = if v == "reflect" || v == "lifesteal" || v == "slow" || v == "damage_modify" {
                    EffectArg::Number(rng.f32_unit().clamp(0.05, 0.95))
                } else {
                    EffectArg::Number(20.0 + rng.f32_unit() * 180.0)
                };
                (v, vec![scalar], DurMod::Set)
            }
        }
    };

    let mut e = EffectStmt {
        verb: verb.to_string(),
        args,
        span: span(),
        area: None,
        tags: Vec::new(),
        duration: None,
        condition: None,
        chance: None,
        stacking: None,
        scalings: Vec::new(),
        lifetime: None,
        nested: Vec::new(),
    };

    // For verbs that consume the `for <duration>` modifier, set it
    // unconditionally so the lowering has a duration source.
    if matches!(dur_state, DurMod::Set) {
        e.duration = Some(EffectDuration {
            duration: dur(1000 + rng.range(4000)),
            span: span(),
        });
        cov.hit("mod:duration");
    }

    // Area — only meaningful for damage-style verbs targeting
    // ground/aoe; emit at low rate to avoid invalid combinations.
    // Nested effects can't carry area shapes (NestedModifierDropped).
    if depth == 0 && (verb == "damage" || verb == "heal") && rng.pick_bool(30) {
        let (shape_name, _argc) = rng.pick(AREA_SHAPES);
        e.area = Some(EffectArea {
            shape: (*shape_name).to_string(),
            args: vec![100.0 + rng.f32_unit() * 400.0],
            span: span(),
        });
        cov.hit(format!("area:{shape_name}"));
    }

    // Tags — Wave 1.6 power-tag list; multi-tag emit valid. Lowering
    // drops tags on nested effects (NestedModifierDropped), so only
    // emit tags at the top level.
    if depth == 0 {
        let n_tags = rng.range(MAX_TAGS as u32 + 1);
        for _ in 0..n_tags {
            let name = rng.pick_str(TAG_NAMES);
            e.tags.push(EffectTag {
                name: name.to_string(),
                value: 20.0 + rng.f32_unit() * 80.0,
                span: span(),
            });
            cov.hit(format!("tag:{name}"));
        }
    }

    // Duration modifier — only for verbs that accept `for` per the
    // lowering's `ModifierNotImplemented{for}` gate. Non-duration-
    // bearing verbs (knockback/pull/blink/execute/self_damage/
    // transfer_gold/modify_standing) reject it. `damage`/`heal` accept
    // it as a buff-style duration; CC verbs already had it set above.
    let verb_accepts_for_modifier =
        matches!(verb, "damage" | "heal" | "shield");
    if matches!(dur_state, DurMod::Free)
        && verb_accepts_for_modifier
        && rng.pick_bool(MODIFIER_P)
    {
        e.duration = Some(EffectDuration {
            duration: dur(500 + rng.range(5000)),
            span: span(),
        });
        cov.hit("mod:duration");
    }

    // Condition — when-only or when+else; bodies are simple
    // comparisons the parser's `capture_cond_text` accepts.
    // Nested effects can't carry conditions (NestedModifierDropped).
    if depth == 0 && rng.pick_bool(MODIFIER_P) {
        let with_else = rng.pick_bool(40);
        e.condition = Some(EffectCondition {
            when_cond: "target.hp < 50".to_string(),
            else_cond: if with_else {
                Some("target.hp >= 50".to_string())
            } else {
                None
            },
            span: span(),
        });
        cov.hit(if with_else { "mod:condition_when_else" } else { "mod:condition_when_only" });
    }

    // Chance — nested effects can't carry it (NestedModifierDropped).
    if depth == 0 && rng.pick_bool(MODIFIER_P) {
        e.chance = Some(EffectChance {
            p: 0.05 + rng.f32_unit() * 0.9,
            span: span(),
        });
        cov.hit("mod:chance");
    }

    // Stacking — nested rejects.
    if depth == 0 && rng.pick_bool(MODIFIER_P) {
        let mode = rng.pick(STACKINGS);
        e.stacking = Some(mode);
        cov.hit(format!("stacking:{mode:?}"));
    }

    // Scalings — nested rejects.
    if depth == 0 {
        let n_scale = rng.range(MAX_SCALINGS as u32 + 1);
        for _ in 0..n_scale {
            let stat = rng.pick_str(STAT_REFS);
            e.scalings.push(EffectScaling {
                percent: 5.0 + rng.f32_unit() * 95.0,
                stat_ref: stat.to_string(),
                span: span(),
            });
            cov.hit(format!("scaling:{stat}"));
        }
    }

    // Lifetime — only for `shield` at top level. Nested rejects.
    if depth == 0 && verb == "shield" && rng.pick_bool(30) {
        let v = match rng.range(3) {
            0 => {
                cov.hit("lifetime:UntilCasterDies");
                EffectLifetime::UntilCasterDies { span: span() }
            }
            1 => {
                cov.hit("lifetime:DamageableHp");
                EffectLifetime::DamageableHp {
                    hp: 50.0 + rng.f32_unit() * 250.0,
                    span: span(),
                }
            }
            _ => {
                cov.hit("lifetime:BreakOnDamage");
                EffectLifetime::BreakOnDamage { span: span() }
            }
        };
        e.lifetime = Some(v);
    }

    // Nested effects — shallow only (depth 0 → at most 1 layer of
    // children, then stop) to avoid combinatorial explosion.
    if depth == 0 && rng.pick_bool(20) {
        let n_nested = 1 + rng.range(MAX_NESTED as u32);
        for _ in 0..n_nested {
            e.nested.push(gen_effect(rng, cov, depth + 1));
        }
        cov.hit("mod:nested");
    }

    e
}

fn gen_ability(rng: &mut Rng, cov: &mut Coverage, name: &str) -> AbilityDecl {
    let headers = gen_headers(rng, cov);
    let n_effects = 1 + rng.range(MAX_EFFECTS as u32);
    let effects: Vec<EffectStmt> = (0..n_effects)
        .map(|_| gen_effect(rng, cov, 0))
        .collect();
    cov.hit(format!("effects_count:{n_effects}"));
    AbilityDecl {
        name: name.to_string(),
        headers,
        effects,
        deliver: None,
        morph: None,
        instantiates: None,
        program: None,
        span: span(),
    }
}

// ---------- Per-ability pipeline ----------

#[derive(Default)]
struct Stats {
    generated: u32,
    parse_ok: u32,
    lower_ok: u32,
}

fn run_one(
    rng: &mut Rng,
    cov: &mut Coverage,
    stats: &mut Stats,
    name: &str,
) -> Result<(), String> {
    let d = gen_ability(rng, cov, name);
    stats.generated += 1;
    let src = emit_ability_file_single(&d);
    let parsed = match parse_ability_file(&src) {
        Ok(f) => f,
        Err(e) => {
            return Err(format!(
                "[{name}] PARSE failed:\n--- source ---\n{src}\n--- error ---\n{e}"
            ));
        }
    };
    if parsed.abilities.is_empty() {
        return Err(format!(
            "[{name}] PARSE returned 0 abilities; src:\n{src}"
        ));
    }
    stats.parse_ok += 1;
    let ad = &parsed.abilities[0];
    if let Err(e) = lower_ability_decl(ad) {
        return Err(format!(
            "[{name}] LOWER failed:\n--- source ---\n{src}\n--- error ---\n{e:?}"
        ));
    }
    stats.lower_ok += 1;
    Ok(())
}

#[test]
fn random_ability_corpus_parses_and_lowers() {
    let mut rng = Rng::new(SEED);
    let mut cov = Coverage::default();
    let mut stats = Stats::default();
    let mut failures: Vec<String> = Vec::new();

    for i in 0..N_ABILITIES {
        let name = format!("R{i:04}");
        if let Err(e) = run_one(&mut rng, &mut cov, &mut stats, &name) {
            failures.push(e);
            // Cap failure noise — keep enough to root-cause without
            // exploding the test output.
            if failures.len() >= 10 {
                break;
            }
        }
    }

    cov.print_report();
    println!(
        "[random_ability_generator] generated={} parsed={} lowered={} failures={}",
        stats.generated,
        stats.parse_ok,
        stats.lower_ok,
        failures.len(),
    );

    // Surface failures FIRST. Without this the coverage-gap panic
    // fires before the failure detail, hiding the actual root cause
    // when the generator emits a shape lowering rejects.
    if !failures.is_empty() {
        let report = failures.join("\n\n===\n\n");
        panic!(
            "{n} of {gen} random abilities failed parse or lower (showing first {n}):\n\n{report}",
            n = failures.len(),
            gen = stats.generated,
        );
    }

    // Coverage pin: every header axis + verb bucket should have been
    // hit at least once across 1024 abilities. If a key never appears
    // the generator missed the axis (regression in the generator
    // logic, not the parser).
    let must_hit = [
        "header:range",
        "verb:damage",
        "verb:heal",
        "verb:shield",
        "verb:stun",
        "tag:PHYSICAL",
        "mod:condition_when_only",
        "mod:chance",
        "mod:duration",
        "scaling:AP",
        "stacking:Refresh",
        "stacking:Stack",
        "stacking:Extend",
        "cost_resource:Mana",
        "cost_amount:Flat",
        "cost_amount:PercentOfMax",
        "header:toggle",
        "header:recharge",
        "header:charges",
        "header:cast",
        "recast:Count",
        "recast:Duration",
    ];
    let mut missing: Vec<&str> = Vec::new();
    for k in must_hit {
        if !cov.counts.contains_key(k) {
            missing.push(k);
        }
    }
    assert!(
        missing.is_empty(),
        "coverage gap — these axes were never exercised across {N_ABILITIES} random abilities: {missing:?}"
    );

    assert_eq!(
        stats.lower_ok as usize, N_ABILITIES,
        "every generated ability must parse and lower",
    );
}
