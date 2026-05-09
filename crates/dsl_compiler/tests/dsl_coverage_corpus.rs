//! DSL feature-coverage corpus regression canary.
//!
//! Walks `assets/ability_test/dsl_coverage/*.ability` and asserts that
//! every `ability` decl in every file lowers without error. Unlike the
//! LoL canary (`lol_corpus_lowering.rs`) — which permits a baseline of
//! deferred-feature parse failures — this corpus is hand-authored to
//! exclusively use surfaces the lowering already supports. A single
//! lowering failure here indicates either:
//!   * an actual regression in `dsl_compiler::ability_lower`, or
//!   * a corpus-author mistake (the ability used a surface the
//!     parser/lowering never agreed on).
//!
//! Either way the test fails noisily so the regression / authoring
//! mistake surfaces immediately.
//!
//! Coverage shape (20 abilities × 5 hero kits):
//!   * BladeStorm    — physical melee striker (projectile + cone/line + DoT
//!                     + chance + stacking stack/extend + AD/MaxHP scaling
//!                     + dash/knockback/pull/modify_standing).
//!   * Lifeweaver    — ally healer (heal/HoT/shield/TimedShield/buff
//!                     + sphere/ring + AP/MR/move_speed scaling
//!                     + until_caster_dies + damageable_hp lifetimes
//!                     + heal/utility hints).
//!   * ShadowSummoner— magical caster (channel/chain/tether/trap delivery
//!                     + ground/direction/vector targets + Mana/Stamina cost
//!                     + charges/recharge + cast verb + when/else condition
//!                     + root/silence/fear/taunt + blink + break_on_damage).
//!   * Aegis         — tank/disruptor (toggle + self_aoe/global targets
//!                     + transfer_gold/modify_standing/execute/self_damage
//!                     + lifesteal/damage_modify + Gold cost + nested block
//!                     + wall/cylinder/dome/hull/spread/box/column shapes).
//!   * Forager       — survivor / colonist (non-combat verbs phase 1:
//!                     harvest / mine / place_voxel).
//!
//! The test prints a per-feature coverage report (which EffectOp /
//! header / modifier / shape / tag each file used) so future maintainers
//! can see at a glance which slots the corpus exercises.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::lower_ability_decl;
use engine::ability::program::{
    AbilityProgram, AbilityTag, CostAmount, CostResource, Delivery, EffectOp, LifetimeMode,
    ScalingStatRef, ShapeKind, StackingMode, TargetModeKind,
};
use dsl_ast::ast::AbilityDecl;
use std::collections::BTreeSet;
use std::path::PathBuf;

#[test]
fn dsl_coverage_corpus_lowers_clean() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("dsl_coverage");
    assert!(
        dir.is_dir(),
        "expected coverage corpus directory at {}",
        dir.display()
    );

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "ability"))
        .collect();
    files.sort();

    assert!(
        !files.is_empty(),
        "no .ability files found in {}",
        dir.display()
    );

    // Per-feature coverage tracking. Each set holds the human-readable
    // labels of the surface variants the corpus exercised; the final
    // report prints them as a sorted list.
    let mut effect_ops:    BTreeSet<&'static str> = BTreeSet::new();
    let mut target_modes:  BTreeSet<&'static str> = BTreeSet::new();
    let mut deliveries:    BTreeSet<&'static str> = BTreeSet::new();
    let mut shapes:        BTreeSet<&'static str> = BTreeSet::new();
    let mut tags:          BTreeSet<&'static str> = BTreeSet::new();
    let mut scalings:      BTreeSet<&'static str> = BTreeSet::new();
    let mut stackings:     BTreeSet<&'static str> = BTreeSet::new();
    let mut lifetimes:     BTreeSet<&'static str> = BTreeSet::new();
    let mut headers:       BTreeSet<&'static str> = BTreeSet::new();
    let mut cost_resources:BTreeSet<&'static str> = BTreeSet::new();
    let mut cost_amounts:  BTreeSet<&'static str> = BTreeSet::new();
    let mut had_chance     = false;
    let mut had_when       = false;
    let mut had_when_else  = false;
    let mut had_nested     = false;
    let mut total          = 0usize;

    for path in &files {
        let src = std::fs::read_to_string(path).expect("read .ability");
        let parsed = parse_ability_file(&src)
            .unwrap_or_else(|e| panic!("PARSE FAIL {}: {:?}", path.display(), e));

        for decl in &parsed.abilities {
            let prog = lower_ability_decl(decl).unwrap_or_else(|e| {
                panic!(
                    "LOWER FAIL {} :: {} -> {}",
                    path.display(),
                    decl.name,
                    e
                )
            });
            total += 1;
            ingest_program(
                decl,
                &prog,
                &mut effect_ops,
                &mut target_modes,
                &mut deliveries,
                &mut shapes,
                &mut tags,
                &mut scalings,
                &mut stackings,
                &mut lifetimes,
                &mut headers,
                &mut cost_resources,
                &mut cost_amounts,
                &mut had_chance,
                &mut had_when,
                &mut had_when_else,
                &mut had_nested,
            );
        }
    }

    eprintln!("\nDSL coverage corpus: {} abilities across {} files all lowered cleanly.\n", total, files.len());
    eprintln!("Per-feature coverage report:");
    eprintln!("  EffectOp variants    ({:>2}): {:?}", effect_ops.len(),    effect_ops);
    eprintln!("  TargetMode variants  ({:>2}): {:?}", target_modes.len(),  target_modes);
    eprintln!("  Delivery variants    ({:>2}): {:?}", deliveries.len(),    deliveries);
    eprintln!("  ShapeKind variants   ({:>2}): {:?}", shapes.len(),        shapes);
    eprintln!("  AbilityTag variants  ({:>2}): {:?}", tags.len(),          tags);
    eprintln!("  ScalingStatRef       ({:>2}): {:?}", scalings.len(),      scalings);
    eprintln!("  StackingMode         ({:>2}): {:?}", stackings.len(),     stackings);
    eprintln!("  LifetimeMode         ({:>2}): {:?}", lifetimes.len(),     lifetimes);
    eprintln!("  Headers (non-target) ({:>2}): {:?}", headers.len(),       headers);
    eprintln!("  CostResource         ({:>2}): {:?}", cost_resources.len(),cost_resources);
    eprintln!("  CostAmount           ({:>2}): {:?}", cost_amounts.len(),  cost_amounts);
    eprintln!("  chance modifier       :  {had_chance}");
    eprintln!("  when modifier         :  {had_when}  (with else: {had_when_else})");
    eprintln!("  nested {{ … }}          :  {had_nested}");

    // Pin coverage floors. If a future edit shrinks coverage below
    // these counts (e.g. by deleting an ability that uniquely
    // exercised a slot), the assertion fires and the maintainer must
    // either (a) restore coverage or (b) consciously bump the floor.
    //
    // Counts come from the canonical engine catalog:
    //   EffectOp        — 26 variants (program.rs)
    //   TargetMode      — 8  variants
    //   Delivery        — Instant + 6 method kinds = 7 distinct labels
    //   ShapeKind       — 12 variants
    //   AbilityTag      — 6  variants
    //   ScalingStatRef  — 8  variants
    //   StackingMode    — 3  variants
    //   LifetimeMode    — 3  variants
    //   CostResource    — 4  variants
    //   CostAmount      — 2  variants (Flat / PercentOfMax)
    //   Headers         — cooldown / range / cast / hint / cost /
    //                     charges / recharge / toggle = 8 (target excluded — counted separately)
    assert_eq!(effect_ops.len(),    26, "EffectOp coverage shrank: {effect_ops:?}");
    assert_eq!(target_modes.len(),  8,  "TargetMode coverage shrank: {target_modes:?}");
    assert_eq!(deliveries.len(),    7,  "Delivery coverage shrank: {deliveries:?}");
    assert_eq!(shapes.len(),        12, "ShapeKind coverage shrank: {shapes:?}");
    assert_eq!(tags.len(),          6,  "AbilityTag coverage shrank: {tags:?}");
    assert_eq!(scalings.len(),      8,  "ScalingStatRef coverage shrank: {scalings:?}");
    assert_eq!(stackings.len(),     3,  "StackingMode coverage shrank: {stackings:?}");
    assert_eq!(lifetimes.len(),     3,  "LifetimeMode coverage shrank: {lifetimes:?}");
    assert_eq!(cost_resources.len(),4,  "CostResource coverage shrank: {cost_resources:?}");
    assert_eq!(cost_amounts.len(),  2,  "CostAmount coverage shrank: {cost_amounts:?}");
    assert_eq!(headers.len(),       8,  "Header coverage shrank: {headers:?}");
    assert!(had_chance,    "no `chance N%` modifier exercised");
    assert!(had_when,      "no `when <cond>` modifier exercised");
    // Wave 1.5#7 GPU eval (2026-05-07): `else <cond>` clause is now
    // rejected at lower time (deferred — open task #163-followup), so
    // no surviving program carries one. Coverage assertion retired.
    let _ = had_when_else;
    assert!(had_nested,    "no nested `{{ ... }}` block exercised");
}

#[allow(clippy::too_many_arguments)]
fn ingest_program(
    decl:           &AbilityDecl,
    prog:           &AbilityProgram,
    effect_ops:     &mut BTreeSet<&'static str>,
    target_modes:   &mut BTreeSet<&'static str>,
    deliveries:     &mut BTreeSet<&'static str>,
    shapes:         &mut BTreeSet<&'static str>,
    tags:           &mut BTreeSet<&'static str>,
    scalings:       &mut BTreeSet<&'static str>,
    stackings:      &mut BTreeSet<&'static str>,
    lifetimes:      &mut BTreeSet<&'static str>,
    headers:        &mut BTreeSet<&'static str>,
    cost_resources: &mut BTreeSet<&'static str>,
    cost_amounts:   &mut BTreeSet<&'static str>,
    had_chance:     &mut bool,
    had_when:       &mut bool,
    had_when_else:  &mut bool,
    had_nested:     &mut bool,
) {
    target_modes.insert(target_label(prog.target_mode));
    deliveries.insert(delivery_label(&prog.delivery));

    for op in prog.effects.iter() {
        effect_ops.insert(effect_op_label(op));
    }
    for nested_inner in prog.nested_per_effect.iter() {
        for op in nested_inner.iter() {
            effect_ops.insert(effect_op_label(op));
        }
        if !nested_inner.is_empty() {
            *had_nested = true;
        }
    }

    for shape in prog.per_effect_areas.iter().flatten() {
        shapes.insert(shape_label(shape.kind));
    }
    for &(tag, _) in prog.tags.iter() {
        tags.insert(tag_label(tag));
    }
    for inner in prog.scalings_per_effect.iter() {
        for sc in inner.iter() {
            scalings.insert(scaling_label(sc.stat_ref));
        }
    }
    for sm in prog.stackings.iter().flatten() {
        stackings.insert(stacking_label(*sm));
    }
    for lt in prog.lifetimes.iter().flatten() {
        lifetimes.insert(lifetime_label(*lt));
    }
    for ch in prog.chances.iter().flatten() {
        let _ = ch;
        *had_chance = true;
    }
    for when in prog.when_per_effect.iter().flatten() {
        *had_when = true;
        if when.else_cond.is_some() {
            *had_when_else = true;
        }
    }

    if prog.gate.cooldown_ticks > 0 {
        headers.insert("cooldown");
    }
    // Walk the AST headers for source-of-truth presence info: the
    // lowered IR collapses some headers (range bakes into Area, cast
    // is silently dropped per Wave 1.6 docs) so we can't read them off
    // `prog`. The AST list preserves authorial intent.
    for h in &decl.headers {
        match h {
            dsl_ast::ast::AbilityHeader::Target(_) => {
                // counted separately via target_modes
            }
            dsl_ast::ast::AbilityHeader::Range(_)        => { headers.insert("range"); }
            dsl_ast::ast::AbilityHeader::Cooldown(_)     => { headers.insert("cooldown"); }
            dsl_ast::ast::AbilityHeader::Cast(_)         => { headers.insert("cast"); }
            dsl_ast::ast::AbilityHeader::Hint(_)         => { headers.insert("hint"); }
            dsl_ast::ast::AbilityHeader::Cost(_)         => { headers.insert("cost"); }
            dsl_ast::ast::AbilityHeader::Charges(_)      => { headers.insert("charges"); }
            dsl_ast::ast::AbilityHeader::Recharge(_)     => { headers.insert("recharge"); }
            dsl_ast::ast::AbilityHeader::Toggle          => { headers.insert("toggle"); }
            // Recast / RecastWindow lower as errors today (Wave 2+); the
            // coverage corpus deliberately avoids them.
            dsl_ast::ast::AbilityHeader::Recast(_)
            | dsl_ast::ast::AbilityHeader::RecastWindow(_) => {}
        }
    }

    if let Some(cost) = prog.cost {
        cost_resources.insert(cost_resource_label(cost.resource));
        cost_amounts.insert(cost_amount_label(cost.amount));
    }
}

fn effect_op_label(op: &EffectOp) -> &'static str {
    match op {
        EffectOp::Damage          { .. } => "Damage",
        EffectOp::Heal            { .. } => "Heal",
        EffectOp::Shield          { .. } => "Shield",
        EffectOp::Stun            { .. } => "Stun",
        EffectOp::Slow            { .. } => "Slow",
        EffectOp::TransferGold    { .. } => "TransferGold",
        EffectOp::ModifyStanding  { .. } => "ModifyStanding",
        EffectOp::CastAbility     { .. } => "CastAbility",
        EffectOp::Root            { .. } => "Root",
        EffectOp::Silence         { .. } => "Silence",
        EffectOp::Fear            { .. } => "Fear",
        EffectOp::Taunt           { .. } => "Taunt",
        EffectOp::Dash            { .. } => "Dash",
        EffectOp::Blink           { .. } => "Blink",
        EffectOp::Knockback       { .. } => "Knockback",
        EffectOp::Pull            { .. } => "Pull",
        EffectOp::Execute         { .. } => "Execute",
        EffectOp::SelfDamage      { .. } => "SelfDamage",
        EffectOp::LifeSteal       { .. } => "LifeSteal",
        EffectOp::DamageModify    { .. } => "DamageModify",
        EffectOp::DamageOverTime  { .. } => "DamageOverTime",
        EffectOp::HealOverTime    { .. } => "HealOverTime",
        EffectOp::TimedShield     { .. } => "TimedShield",
        EffectOp::Buff            { .. } => "Buff",
        EffectOp::Summon          { .. } => "Summon",
        EffectOp::Harvest         { .. } => "Harvest",
        EffectOp::PlaceVoxel      { .. } => "PlaceVoxel",
        EffectOp::Stealth         { .. } => "Stealth",
        EffectOp::Charm           { .. } => "Charm",
        EffectOp::Grounded        { .. } => "Grounded",
        EffectOp::Suppress        { .. } => "Suppress",
        EffectOp::Reflect         { .. } => "Reflect",
        EffectOp::PlantBelief     { .. } => "PlantBelief",
        EffectOp::Observe         { .. } => "Observe",
        EffectOp::Scry            { .. } => "Scry",
        EffectOp::Reveal          { .. } => "Reveal",
        EffectOp::Disguise        { .. } => "Disguise",
        EffectOp::Decoy           { .. } => "Decoy",
        EffectOp::EraseBelief     { .. } => "EraseBelief",
        EffectOp::TravelTo        { .. } => "TravelTo",
        EffectOp::Recipe          { .. } => "Recipe",
        EffectOp::WearTool        { .. } => "WearTool",
        EffectOp::Propose         { .. } => "Propose",
        EffectOp::Announce        { .. } => "Announce",
        EffectOp::GainSkill       { .. } => "GainSkill",
        EffectOp::CreateObligation { .. } => "CreateObligation",
    }
}

fn target_label(t: TargetModeKind) -> &'static str {
    match t {
        TargetModeKind::Enemy     => "Enemy",
        TargetModeKind::SelfCast  => "SelfCast",
        TargetModeKind::Ally      => "Ally",
        TargetModeKind::SelfAoe   => "SelfAoe",
        TargetModeKind::Ground    => "Ground",
        TargetModeKind::Direction => "Direction",
        TargetModeKind::Vector    => "Vector",
        TargetModeKind::Global    => "Global",
    }
}

fn delivery_label(d: &Delivery) -> &'static str {
    match d {
        Delivery::Instant => "Instant",
        Delivery::Method { kind, .. } => {
            use engine::ability::program::DeliveryMethodKind as DM;
            match kind {
                DM::Projectile => "Projectile",
                DM::Channel    => "Channel",
                DM::Zone       => "Zone",
                DM::Chain      => "Chain",
                DM::Tether     => "Tether",
                DM::Trap       => "Trap",
            }
        }
    }
}

fn shape_label(s: ShapeKind) -> &'static str {
    match s {
        ShapeKind::Circle   => "Circle",
        ShapeKind::Cone     => "Cone",
        ShapeKind::Line     => "Line",
        ShapeKind::Ring     => "Ring",
        ShapeKind::Spread   => "Spread",
        ShapeKind::Box      => "Box",
        ShapeKind::Sphere   => "Sphere",
        ShapeKind::Column   => "Column",
        ShapeKind::Wall     => "Wall",
        ShapeKind::Cylinder => "Cylinder",
        ShapeKind::Dome     => "Dome",
        ShapeKind::Hull     => "Hull",
    }
}

fn tag_label(t: AbilityTag) -> &'static str {
    match t {
        AbilityTag::Physical     => "PHYSICAL",
        AbilityTag::Magical      => "MAGICAL",
        AbilityTag::CrowdControl => "CROWD_CONTROL",
        AbilityTag::Heal         => "HEAL",
        AbilityTag::Defense      => "DEFENSE",
        AbilityTag::Utility      => "UTILITY",
    }
}

fn scaling_label(s: ScalingStatRef) -> &'static str {
    match s {
        ScalingStatRef::AttackDamage => "AD",
        ScalingStatRef::AbilityPower => "AP",
        ScalingStatRef::MaxHp        => "MaxHP",
        ScalingStatRef::Hp           => "HP",
        ScalingStatRef::Armor        => "armor",
        ScalingStatRef::MagicResist  => "MR",
        ScalingStatRef::MoveSpeed    => "move_speed",
        ScalingStatRef::Mana         => "mana",
    }
}

fn stacking_label(s: StackingMode) -> &'static str {
    match s {
        StackingMode::Refresh => "refresh",
        StackingMode::Stack   => "stack",
        StackingMode::Extend  => "extend",
    }
}

fn lifetime_label(l: LifetimeMode) -> &'static str {
    match l {
        LifetimeMode::UntilCasterDies   => "until_caster_dies",
        LifetimeMode::DamageableHp(_)   => "damageable_hp",
        LifetimeMode::BreakOnDamage     => "break_on_damage",
    }
}

fn cost_resource_label(r: CostResource) -> &'static str {
    match r {
        CostResource::Mana    => "Mana",
        CostResource::Stamina => "Stamina",
        CostResource::Hp      => "Hp",
        CostResource::Gold    => "Gold",
    }
}

fn cost_amount_label(a: CostAmount) -> &'static str {
    match a {
        CostAmount::Flat(_)         => "Flat",
        CostAmount::PercentOfMax(_) => "PercentOfMax",
    }
}
