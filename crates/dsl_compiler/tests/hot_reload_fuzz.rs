//! Fuzz-scale hot-reload pin — bridges the random ability generator
//! (`tests/ability_random_generator.rs`) with the hot-reload primitive
//! (`tests/hot_reload_end_to_end.rs`).
//!
//! ## Why
//!
//! The existing single-ability hot-reload pins prove the swap contract
//! for `damage 10 → damage 25` against a hand-crafted source. They
//! don't prove the swap contract holds for the breadth of the grammar
//! — what if a particular header combo, modifier slot, or nested
//! effect shape silently aliased through the registry's
//! `with_program_replaced` path?
//!
//! This pin runs N random source abilities through the FULL cycle:
//!
//!   1. Build a random `AbilityDecl` v1 + a perturbed v2 (same shape,
//!      one literal nudged by a delta).
//!   2. Emit v1 → parse → lower → register in an `AbilityRegistry`.
//!   3. Emit v2 → parse → lower → `with_program_replaced(id, v2)`.
//!   4. Confirm:
//!        a. The new registry returns the v2 program at the swapped id.
//!        b. The original registry STILL returns the v1 program (the
//!           immutable-snapshot contract any in-flight Arc<Registry>
//!           consumer depends on).
//!        c. The packed-registry bytes change after the swap (proves
//!           a GPU re-upload would observe the diff — not a no-op).
//!        d. Slot id stays stable across the swap.
//!
//! Determinism: single fixed seed (`0xC0FF_EE15_5_E5ED`), 64 abilities.
//! Any per-shape regression points at the exact emitted source via
//! the assert message.

use dsl_ast::ability_emit::emit_ability_file_single;
use dsl_ast::ability_parser::parse_ability_file;
use dsl_ast::ast::*;
use dsl_compiler::ability_lower::lower_ability_decl;
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, PackedAbilityRegistry,
};

const SEED: u64 = 0xC0FF_EE15_5_E5ED;
const N_PAIRS: usize = 64;
const PERTURB_DELTA: f32 = 17.0;

// ---------- xorshift64 (same shape as random_ability_generator) ----------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_F00D } else { seed },
        }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
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
    fn pct(&mut self) -> u8 {
        (self.next_u32() % 100) as u8
    }
    fn f32_unit(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

fn span() -> Span {
    Span { start: 0, end: 0 }
}

fn dur(ms: u32) -> Duration {
    Duration { millis: ms }
}

// Small-surface generator — narrower than ability_random_generator.rs
// because the goal here is *coverage of the swap path*, not coverage
// of every grammar combo. Each ability has one effect with one numeric
// arg, a stable header set, and an optional modifier; the v2 perturbs
// the numeric arg by `PERTURB_DELTA` so the swap's byte-level diff
// lands at a known offset in the packed columns.
fn gen_pair(rng: &mut Rng, name: &str) -> (AbilityDecl, AbilityDecl, f32, f32) {
    let target = rng.pick(&[
        TargetMode::Enemy,
        TargetMode::Self_,
        TargetMode::Ally,
    ]);
    let hint = rng.pick(&[
        HintName::Damage,
        HintName::Defense,
        HintName::Heal,
        HintName::Utility,
    ]);
    let cooldown_phase = rng.pick(&[None, Some(CooldownPhase::Cast)]);
    let verb = rng.pick_str(&["damage", "heal"]);
    let v1_amount: f32 = 10.0 + rng.f32_unit() * 90.0;
    let v2_amount: f32 = v1_amount + PERTURB_DELTA;

    let with_chance = rng.pct() < 50;
    let with_when = rng.pct() < 30;

    let make_decl = |amount: f32, name: &str| AbilityDecl {
        name: name.to_string(),
        headers: vec![
            AbilityHeader::Target(target),
            AbilityHeader::Range(150.0 + rng_seeded_offset(amount) * 200.0),
            AbilityHeader::Cooldown(dur(2_000 + (amount as u32 * 100)), cooldown_phase),
            AbilityHeader::Hint(hint),
        ],
        effects: vec![EffectStmt {
            verb: verb.to_string(),
            args: vec![EffectArg::Number(amount)],
            span: span(),
            area: None,
            tags: Vec::new(),
            duration: None,
            condition: if with_when {
                Some(EffectCondition {
                    when_cond: "target.hp < 50".to_string(),
                    else_cond: None,
                    span: span(),
                })
            } else {
                None
            },
            chance: if with_chance {
                Some(EffectChance {
                    p: 0.25,
                    span: span(),
                })
            } else {
                None
            },
            stacking: None,
            scalings: Vec::new(),
            lifetime: None,
            nested: Vec::new(),
        }],
        deliver: None,
        morph: None,
        instantiates: None,
        program: None,
        span: span(),
    };

    (
        make_decl(v1_amount, name),
        make_decl(v2_amount, name),
        v1_amount,
        v2_amount,
    )
}

/// Deterministic offset hash so the `range` value is a pure function of
/// `amount` — keeps the decl shape stable between v1/v2 generations
/// even though the rng would otherwise step. (The rng is consumed only
/// at decl-shape decision points; per-amount nudges feed into the
/// header values via this helper.)
fn rng_seeded_offset(amount: f32) -> f32 {
    let bits = amount.to_bits();
    ((bits & 0xFF) as f32) / 255.0
}

fn parse_and_lower(decl: &AbilityDecl) -> AbilityProgram {
    let src = emit_ability_file_single(decl);
    let file = parse_ability_file(&src)
        .unwrap_or_else(|e| panic!("parse failed for source:\n{src}\n--- error ---\n{e}"));
    let parsed = file
        .abilities
        .into_iter()
        .next()
        .expect("at least one ability");
    lower_ability_decl(&parsed)
        .unwrap_or_else(|e| panic!("lower failed for source:\n{src}\n--- error ---\n{e:?}"))
}

fn extract_first_damage(prog: &AbilityProgram) -> Option<f32> {
    prog.effects.iter().find_map(|e| match e {
        EffectOp::Damage { amount } => Some(*amount),
        _ => None,
    })
}

fn extract_first_heal(prog: &AbilityProgram) -> Option<f32> {
    prog.effects.iter().find_map(|e| match e {
        EffectOp::Heal { amount } => Some(*amount),
        _ => None,
    })
}

#[test]
fn random_ability_pairs_hot_swap_cleanly() {
    let mut rng = Rng::new(SEED);
    let mut diffs_observed: u32 = 0;

    for i in 0..N_PAIRS {
        let name = format!("HotR{i:03}");
        let (v1_decl, v2_decl, v1_amount, v2_amount) = gen_pair(&mut rng, &name);

        let prog_v1 = parse_and_lower(&v1_decl);
        let prog_v2 = parse_and_lower(&v2_decl);

        // (a) Register v1 + build the immutable v1 registry.
        let mut builder = AbilityRegistryBuilder::new();
        let id = builder.register(prog_v1.clone());
        let registry_v1 = builder.build();

        let v1_lookup_amount = match v1_decl.effects[0].verb.as_str() {
            "damage" => extract_first_damage(registry_v1.get(id).expect("v1 registered")),
            "heal" => extract_first_heal(registry_v1.get(id).expect("v1 registered")),
            other => panic!("[{name}] unexpected verb {other}"),
        }
        .unwrap_or_else(|| panic!("[{name}] v1 program missing the expected effect"));
        assert!(
            (v1_lookup_amount - v1_amount).abs() < 0.01,
            "[{name}] v1 lookup mismatch: expected {v1_amount}, got {v1_lookup_amount}",
        );

        // (b) Hot swap. `with_program_replaced` returns
        // `Option<AbilityRegistry>` — `None` only if the id is
        // out-of-bounds (impossible here since we just registered).
        let registry_v2 = registry_v1
            .with_program_replaced(id, prog_v2.clone())
            .unwrap_or_else(|| panic!("[{name}] with_program_replaced returned None for a freshly-registered id"));

        // (c) New registry returns v2.
        let v2_lookup_amount = match v2_decl.effects[0].verb.as_str() {
            "damage" => extract_first_damage(registry_v2.get(id).expect("v2 present at id")),
            "heal" => extract_first_heal(registry_v2.get(id).expect("v2 present at id")),
            _ => unreachable!(),
        }
        .unwrap_or_else(|| panic!("[{name}] v2 program missing the expected effect"));
        assert!(
            (v2_lookup_amount - v2_amount).abs() < 0.01,
            "[{name}] v2 lookup mismatch: expected {v2_amount}, got {v2_lookup_amount}",
        );

        // (d) Original registry STILL has v1 (immutable snapshot).
        let still_v1 = match v1_decl.effects[0].verb.as_str() {
            "damage" => extract_first_damage(
                registry_v1.get(id).expect("v1 still present after swap"),
            ),
            "heal" => extract_first_heal(
                registry_v1.get(id).expect("v1 still present after swap"),
            ),
            _ => unreachable!(),
        }
        .unwrap_or_else(|| panic!("[{name}] v1 program missing post-swap"));
        assert!(
            (still_v1 - v1_amount).abs() < 0.01,
            "[{name}] v1 registry mutated after swap: expected immutable {v1_amount}, got {still_v1}",
        );

        // (e) Slot ID + length contracts.
        assert_eq!(
            registry_v1.len(),
            registry_v2.len(),
            "[{name}] swap must not resize the registry",
        );

        // (f) Packed-bytes diff lands. The Damage / Heal verbs both
        // store `bitcast<u32>(amount)` in `effect_payload_a[0]`, so
        // post-swap that word MUST differ.
        let packed_v1 = PackedAbilityRegistry::pack(&registry_v1);
        let packed_v2 = PackedAbilityRegistry::pack(&registry_v2);
        let v1_word = packed_v1.effect_payload_a[0];
        let v2_word = packed_v2.effect_payload_a[0];
        assert_eq!(
            v1_word,
            f32::to_bits(v1_amount),
            "[{name}] v1 packed bytes != bitcast<u32>({v1_amount}); got 0x{v1_word:08X}",
        );
        assert_eq!(
            v2_word,
            f32::to_bits(v2_amount),
            "[{name}] v2 packed bytes != bitcast<u32>({v2_amount}); got 0x{v2_word:08X}",
        );
        assert_ne!(
            v1_word, v2_word,
            "[{name}] packed payload MUST diverge after swap — GPU re-upload would be a no-op otherwise",
        );
        diffs_observed += 1;
    }

    println!(
        "[hot_reload_fuzz] {diffs_observed}/{N_PAIRS} random ability pairs hot-swapped cleanly",
    );
    assert_eq!(
        diffs_observed as usize, N_PAIRS,
        "every pair must complete the full swap-and-verify cycle",
    );
}

/// Repeated swaps on the same slot — proves the swap path is stable
/// when an author rapidly saves a file over and over (the realistic
/// file-watch hot-reload pattern). After N rounds we have N+1
/// independent registries (each one produced by
/// `with_program_replaced` from the previous round); the
/// immutable-snapshot contract requires each one to still hold the
/// program it was built with, regardless of how many subsequent
/// swaps fired off the same parent.
///
/// `AbilityRegistry` isn't `Clone`, but `with_program_replaced` takes
/// `&self` and returns a fresh owned `AbilityRegistry`, so chaining
/// keeps every previous snapshot alive at its own value.
#[test]
fn repeated_swaps_preserve_immutable_snapshots_across_rounds() {
    let mut rng = Rng::new(SEED ^ 0x5A5A_5A5A_5A5A_5A5A);
    let (mut current_decl, _v2_unused, _v1_a, _v2_a) = gen_pair(&mut rng, "Repeated");

    // Base amount = the decl's only positional arg.
    let base_amount: f32 = match current_decl.effects[0].args[0] {
        EffectArg::Number(n) => n,
        _ => unreachable!(),
    };
    let verb = current_decl.effects[0].verb.clone();

    let mut builder = AbilityRegistryBuilder::new();
    let id = builder.register(parse_and_lower(&current_decl));
    let mut snapshots: Vec<(f32, engine::ability::AbilityRegistry)> = Vec::new();
    snapshots.push((base_amount, builder.build()));

    const N_ROUNDS: usize = 8;
    for round in 0..N_ROUNDS {
        let next_amount = base_amount + ((round + 1) as f32) * PERTURB_DELTA;
        current_decl.effects[0].args[0] = EffectArg::Number(next_amount);
        let new_prog = parse_and_lower(&current_decl);
        let prev_reg = &snapshots.last().expect("at least one snapshot").1;
        let new_reg = prev_reg
            .with_program_replaced(id, new_prog)
            .expect("swap at known id");
        snapshots.push((next_amount, new_reg));
    }

    // Walk every snapshot and confirm it still holds its own amount —
    // proves no swap mutated a previous registry by reference.
    for (i, (expected_amount, snap)) in snapshots.iter().enumerate() {
        let prog = snap.get(id).expect("snapshot has id");
        let actual = match verb.as_str() {
            "damage" => extract_first_damage(prog),
            "heal" => extract_first_heal(prog),
            other => panic!("unexpected verb {other}"),
        }
        .unwrap_or_else(|| panic!("snapshot {i} missing expected effect"));
        assert!(
            (actual - expected_amount).abs() < 0.01,
            "snapshot {i} mutated by later swap: expected {expected_amount}, got {actual}",
        );
    }

    println!(
        "[hot_reload_fuzz] {N_ROUNDS}-round repeated-swap chain preserved every intermediate snapshot",
    );
}
