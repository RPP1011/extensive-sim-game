//! Ability grammar fuzz harness.
//!
//! Builds on the per-variant grammar walker
//! (`dsl_ast/tests/ability_grammar_walker.rs` and
//! `dsl_compiler/tests/ability_grammar_walker_lower.rs`) by generating
//! a large corpus of RANDOM valid ability decls — varying header
//! lists, effect counts, modifier combinations, nesting depth, arg
//! types — and asserting that each one parses cleanly AND lowers
//! cleanly. Coverage signal:
//!
//!   * The systematic walker proves "every variant works in isolation".
//!   * This fuzzer proves "valid combinations across the variant matrix
//!     don't have hidden interaction bugs".
//!
//! Runs deterministically off `per_agent_u32` to satisfy P5 — the
//! corpus is reproducible across runs and CI.
//!
//! Failure mode: the test reports the FIRST emitted source that failed
//! to round-trip, so a regression points at the exact ability shape
//! that broke. Subsequent iterations are pruned to keep the report
//! focused.

use dsl_ast::ability_emit::emit_ability_file_single;
use dsl_ast::ast::*;
use dsl_compiler::ability_lower::lower_ability_decl;

fn span() -> Span {
    Span { start: 0, end: 0 }
}

/// Reproducible PCG state. Counter-style: each call returns a fresh
/// u32 derived from (seed, counter). Matches the engine's
/// `per_agent_u32` style — pure function of (seed, step).
struct Pcg {
    seed: u64,
    counter: u32,
}

impl Pcg {
    fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }
    fn next_u32(&mut self) -> u32 {
        // Splitmix-style mixer, deterministic.
        let mut x = self.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= (self.counter as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        self.counter = self.counter.wrapping_add(1);
        (x & 0xFFFF_FFFF) as u32
    }
    fn pick<T: Copy>(&mut self, options: &[T]) -> T {
        let idx = (self.next_u32() as usize) % options.len();
        options[idx]
    }
    fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u32() % (hi - lo + 1))
    }
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let span = hi - lo;
        let r = (self.next_u32() as f32) / (u32::MAX as f32);
        lo + r * span
    }
    /// Bernoulli with probability `p` in [0.0, 1.0].
    fn bool_with_prob(&mut self, p: f32) -> bool {
        let r = (self.next_u32() as f32) / (u32::MAX as f32);
        r < p
    }
}

/// Generate a random lowering-valid `TargetMode`.
fn gen_target(rng: &mut Pcg) -> TargetMode {
    // SelfAoe and Self_ map to the same dispatch in some lowering paths;
    // include the full subset that lowering supports today.
    rng.pick(&[
        TargetMode::Enemy,
        TargetMode::Self_,
        TargetMode::Ally,
        TargetMode::SelfAoe,
        TargetMode::Ground,
    ])
}

/// Generate a random lowering-valid `HintName` (excludes Economic).
fn gen_hint(rng: &mut Pcg) -> HintName {
    rng.pick(&[
        HintName::Damage,
        HintName::Defense,
        HintName::CrowdControl,
        HintName::Utility,
        HintName::Heal,
        HintName::Buff,
    ])
}

/// Generate a random `CostSpec`.
fn gen_cost(rng: &mut Pcg) -> CostSpec {
    let resource = rng.pick(&[
        CostResource::Mana,
        CostResource::Stamina,
        CostResource::Hp,
        CostResource::Gold,
    ]);
    let amount = if rng.bool_with_prob(0.7) {
        CostAmount::Flat(rng.range_f32(10.0, 100.0).round())
    } else {
        CostAmount::PercentOfMax(rng.range_f32(5.0, 20.0).round())
    };
    CostSpec { resource, amount, span: span() }
}

/// Generate a random `Duration` in 100ms..15s.
fn gen_duration(rng: &mut Pcg) -> Duration {
    Duration {
        millis: rng.range_u32(100, 15_000),
    }
}

/// Generate a random valid header list. Always includes target +
/// range; adds optional cooldown / cast / hint / cost / charges /
/// toggle independently with realistic probabilities.
fn gen_headers(rng: &mut Pcg) -> Vec<AbilityHeader> {
    let mut h = Vec::new();
    h.push(AbilityHeader::Target(gen_target(rng)));
    h.push(AbilityHeader::Range(rng.range_f32(100.0, 1200.0).round()));
    if rng.bool_with_prob(0.8) {
        let phase = if rng.bool_with_prob(0.3) {
            Some(CooldownPhase::Cast)
        } else {
            None
        };
        h.push(AbilityHeader::Cooldown(gen_duration(rng), phase));
    }
    if rng.bool_with_prob(0.3) {
        h.push(AbilityHeader::Cast(gen_duration(rng)));
    }
    if rng.bool_with_prob(0.6) {
        h.push(AbilityHeader::Hint(gen_hint(rng)));
    }
    if rng.bool_with_prob(0.6) {
        h.push(AbilityHeader::Cost(gen_cost(rng)));
    }
    if rng.bool_with_prob(0.15) {
        h.push(AbilityHeader::Charges(rng.range_u32(1, 5)));
        h.push(AbilityHeader::Recharge(gen_duration(rng)));
    }
    h
}

/// Generate a random `EffectArg` of the appropriate kind for the
/// verb. Keeps the verb→args contract realistic so the lowering pass
/// doesn't reject a number-where-duration shape.
fn gen_args_for_verb(rng: &mut Pcg, verb: &str) -> Vec<EffectArg> {
    match verb {
        "damage" | "heal" | "shield" => {
            vec![EffectArg::Number(rng.range_f32(20.0, 250.0).round())]
        }
        "stun" | "root" | "silence" | "slow" => {
            // Body takes a duration arg.
            vec![EffectArg::Duration(gen_duration(rng))]
        }
        _ => Vec::new(),
    }
}

/// Generate a random `EffectStmt`. `depth` bounds nested-effect
/// recursion; we cap at 2 to keep the output readable.
fn gen_effect(rng: &mut Pcg, depth: u32) -> EffectStmt {
    let verb = rng.pick(&["damage", "heal", "shield", "stun"]);
    let mut e = EffectStmt {
        verb: verb.to_string(),
        args: gen_args_for_verb(rng, verb),
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
    // Optional area for damage/heal verbs (in <shape>).
    if (verb == "damage" || verb == "heal") && rng.bool_with_prob(0.25) {
        e.area = Some(EffectArea {
            shape: rng.pick(&["circle", "sphere", "cone"]).to_string(),
            args: vec![rng.range_f32(150.0, 500.0).round()],
            span: span(),
        });
    }
    // Optional duration (for non-stun/root/silence — those took it
    // as their primary arg).
    if !matches!(verb, "stun" | "root" | "silence" | "slow") && rng.bool_with_prob(0.15) {
        e.duration = Some(EffectDuration {
            duration: gen_duration(rng),
            span: span(),
        });
    }
    // Optional chance gate.
    if rng.bool_with_prob(0.15) {
        e.chance = Some(EffectChance {
            p: (rng.range_f32(10.0, 80.0).round()) / 100.0,
            span: span(),
        });
    }
    // Optional stacking (only with duration).
    if e.duration.is_some() && rng.bool_with_prob(0.4) {
        e.stacking = Some(rng.pick(&[
            StackingMode::Refresh,
            StackingMode::Stack,
            StackingMode::Extend,
        ]));
    }
    // Optional power tags (registered AbilityTag names only — the
    // lowering registry rejects other names).
    if rng.bool_with_prob(0.3) {
        let tag_name = rng.pick(&["PHYSICAL", "MAGICAL", "CROWD_CONTROL"]);
        e.tags.push(EffectTag {
            name: tag_name.to_string(),
            value: rng.range_f32(20.0, 80.0).round(),
            span: span(),
        });
    }
    // Optional scaling terms.
    if rng.bool_with_prob(0.3) {
        let stat = rng.pick(&["AP", "AD"]);
        e.scalings.push(EffectScaling {
            percent: rng.range_f32(20.0, 60.0).round(),
            stat_ref: stat.to_string(),
            span: span(),
        });
    }
    // Optional nested follow-up effect (depth-bounded).
    // CAVEAT: nested effects can only carry `for <duration>`; every
    // other modifier (area, tags, chance, stacking, scalings,
    // lifetime, condition, deeper nesting) is rejected by the
    // lowering's `NestedModifierDropped` guard — recursive aggregator
    // capture is a future architectural lift. So we generate a BARE
    // nested effect: verb + args + optional duration.
    if depth < 2 && rng.bool_with_prob(0.15) {
        e.nested.push(gen_bare_effect(rng));
    }
    e
}

/// Bare nested effect — verb + args + optional duration only. No
/// area / tags / chance / stacking / scalings / lifetime / condition.
fn gen_bare_effect(rng: &mut Pcg) -> EffectStmt {
    let verb = rng.pick(&["damage", "heal", "shield"]);
    let mut e = EffectStmt {
        verb: verb.to_string(),
        args: gen_args_for_verb(rng, verb),
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
    if rng.bool_with_prob(0.25) {
        e.duration = Some(EffectDuration {
            duration: gen_duration(rng),
            span: span(),
        });
    }
    e
}

/// Generate a random complete `AbilityDecl`.
fn gen_ability(rng: &mut Pcg, idx: usize) -> AbilityDecl {
    let n_effects = rng.range_u32(1, 4) as usize;
    let effects: Vec<EffectStmt> = (0..n_effects).map(|_| gen_effect(rng, 0)).collect();
    AbilityDecl {
        name: format!("Fuzz{idx}"),
        headers: gen_headers(rng),
        effects,
        deliver: None,
        morph: None,
        instantiates: None,
        program: None,
        span: span(),
    }
}

#[test]
fn fuzz_1000_random_abilities_parse_and_lower_clean() {
    const N: usize = 1000;
    const SEED: u64 = 0xC0DE_BEEF_2026_0516;

    let mut rng = Pcg::new(SEED);
    let mut total_parsed = 0usize;
    let mut total_lowered = 0usize;
    let mut total_effects = 0usize;
    let mut total_modifiers = 0usize;
    let mut first_failure: Option<(usize, String, String, String)> = None;

    for i in 0..N {
        let ability = gen_ability(&mut rng, i);
        let src = emit_ability_file_single(&ability);
        let parsed = match dsl_ast::parse_ability_file(&src) {
            Ok(f) => f,
            Err(e) => {
                if first_failure.is_none() {
                    first_failure = Some((i, src.clone(), "parse".to_string(), format!("{e}")));
                }
                continue;
            }
        };
        total_parsed += 1;
        if parsed.abilities.len() != 1 {
            continue;
        }
        let ad = &parsed.abilities[0];
        total_effects += ad.effects.len();
        total_modifiers += ad
            .effects
            .iter()
            .map(|e| {
                let mut m = 0usize;
                if e.area.is_some() { m += 1; }
                m += e.tags.len();
                if e.duration.is_some() { m += 1; }
                if e.chance.is_some() { m += 1; }
                if e.stacking.is_some() { m += 1; }
                m += e.scalings.len();
                if e.lifetime.is_some() { m += 1; }
                m += e.nested.len();
                m
            })
            .sum::<usize>();
        match lower_ability_decl(ad) {
            Ok(_) => total_lowered += 1,
            Err(e) => {
                if first_failure.is_none() {
                    first_failure =
                        Some((i, src.clone(), "lower".to_string(), format!("{e:?}")));
                }
            }
        }
    }

    println!("==== ability fuzz harness — {N} random valid abilities ====");
    println!(
        "  parsed={total_parsed}/{N} ({pct:.1}%)",
        pct = (total_parsed as f64) / (N as f64) * 100.0
    );
    println!(
        "  lowered={total_lowered}/{N} ({pct:.1}%)",
        pct = (total_lowered as f64) / (N as f64) * 100.0
    );
    println!("  total effects emitted: {total_effects}");
    println!("  total modifier-slot occurrences: {total_modifiers}");
    println!("==========================================================");

    if let Some((i, src, stage, err)) = first_failure {
        panic!(
            "[fuzz iter {i}] {stage} failure:\n--- emitted ---\n{src}\n--- error ---\n{err}"
        );
    }

    // Sanity: the generator must have actually produced a meaningful
    // mix of variants — not just trivial empty effects. ≥1 effect per
    // ability minimum; expect ≥2× modifier-slot uses on top.
    assert_eq!(total_parsed, N, "every generated ability must parse");
    assert_eq!(total_lowered, N, "every generated ability must lower");
    assert!(
        total_effects >= N,
        "expected ≥{N} total effects ({}/iter avg); got {total_effects}",
        1.0
    );
    assert!(
        total_modifiers >= N,
        "expected ≥{N} modifier-slot uses on top (probabilistic gen); got {total_modifiers}"
    );
}
