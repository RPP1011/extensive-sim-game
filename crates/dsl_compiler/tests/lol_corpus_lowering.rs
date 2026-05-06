//! LoL hero corpus lowering canary.
//!
//! Walks `dataset/abilities/lol_heroes/*.ability` and reports the
//! parse + lower pass rate. Establishes a regression-detection floor
//! for cumulative lowering work. After Wave 1.5#7 (when-condition,
//! 9e9e866a), Wave 1.5#9 (nested, c423460b), and Wave 2 piece 5/6
//! (Delivery::Method, 2f80b6d6) the corpus should lower at a much
//! higher rate than before — this test pins the baseline so
//! accidental regressions surface.
//!
//! Failure modes intentionally NOT asserted-against (still expected
//! Wave 2+ work):
//!   * `HeaderNotImplemented{recast | recast_window}` (Wave 1.4 left)
//!   * `MorphBlockNotImplemented` (Wave 1.4 left)
//!   * `TemplateInstantiationNotImplemented` (Wave 1.2 left)
//!
//! The test prints a per-error-category breakdown so any regression
//! lands with diagnostic detail.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, LowerError};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[test]
fn lol_corpus_lowering_baseline() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("dataset")
        .join("abilities")
        .join("lol_heroes");
    if !dir.is_dir() {
        eprintln!("dataset/abilities/lol_heroes not found at {}", dir.display());
        return;
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "ability"))
        .collect();
    files.sort();

    let mut ok = 0usize;
    let mut parse_err = 0usize;
    let mut lower_err: BTreeMap<String, usize> = BTreeMap::new();

    for path in &files {
        let src = std::fs::read_to_string(path).expect("read .ability");
        let file = match parse_ability_file(&src) {
            Ok(f) => f,
            Err(_) => {
                parse_err += 1;
                continue;
            }
        };
        let mut all_ok = true;
        for decl in &file.abilities {
            if let Err(e) = lower_ability_decl(decl) {
                all_ok = false;
                let key = match e {
                    LowerError::HeaderNotImplemented { header, .. } => {
                        format!("HeaderNotImplemented:{header}")
                    }
                    LowerError::PassiveBlockNotImplemented { .. } => {
                        "PassiveBlockNotImplemented".to_string()
                    }
                    LowerError::ModifierNotImplemented { modifier, .. } => {
                        format!("ModifierNotImplemented:{modifier}")
                    }
                    LowerError::MorphBlockNotImplemented { .. } => {
                        "MorphBlockNotImplemented".to_string()
                    }
                    LowerError::TemplateBlockNotImplemented { .. } => {
                        "TemplateBlockNotImplemented".to_string()
                    }
                    LowerError::TemplateInstantiationNotImplemented { .. } => {
                        "TemplateInstantiationNotImplemented".to_string()
                    }
                    LowerError::StructureBlockNotImplemented { .. } => {
                        "StructureBlockNotImplemented".to_string()
                    }
                    LowerError::UnknownDeliveryMethod { method, .. } => {
                        format!("UnknownDeliveryMethod:{method}")
                    }
                    LowerError::UnknownTag { tag, .. } => format!("UnknownTag:{tag}"),
                    LowerError::TagBudgetExceeded { .. } => "TagBudgetExceeded".to_string(),
                    LowerError::UnknownShape { shape, .. } => format!("UnknownShape:{shape}"),
                    LowerError::UnknownStatRef { stat, .. } => format!("UnknownStatRef:{stat}"),
                    LowerError::ScalingBudgetExceeded { .. } => "ScalingBudgetExceeded".to_string(),
                    LowerError::NestedBudgetExceeded { .. } => "NestedBudgetExceeded".to_string(),
                    other => format!("{other:?}").chars().take(60).collect::<String>(),
                };
                *lower_err.entry(key).or_insert(0) += 1;
                break;
            }
        }
        if all_ok {
            ok += 1;
        }
    }

    eprintln!("LoL corpus baseline ({} files):", files.len());
    eprintln!("  ok           : {ok}");
    eprintln!("  parse_err    : {parse_err}");
    eprintln!("  lower errors :");
    for (e, c) in &lower_err {
        eprintln!("    {c:>3}  {e}");
    }

    // Baseline assertion — frozen at the post-TargetMode-lift number.
    // If a later commit lowers MORE files, bump this floor; if fewer,
    // investigate the regression. With TargetMode direction/ground/etc.
    // now lowering (#127), the next biggest unblocks are
    // EffectArgMismatch:dash (32 files), HeaderNotImplemented:recast
    // (34 files — Wave 1.4 deferred multi-stage cast state), and
    // MixedBody (parser-permitted spec violations).
    //
    // Bumped 81 → 89 when EffectOp::Summon landed — 8 LoL .ability
    // files (Annie, Azir, Janna, Jhin, Malzahar, Swain, Yorick, Zyra
    // and a handful more) used the previously-unrecognised `summon`
    // verb that now lowers cleanly.
    let baseline = 89usize;
    assert!(
        ok >= baseline,
        "LoL lowering regression: ok={ok} fell below baseline={baseline}",
    );
}

/// Honesty audit for the canary "ok" count. The baseline test above
/// asserts "lowering returns Ok(_)" but says nothing about whether the
/// resulting AbilityProgram carries actionable IR. This test breaks the
/// 89 ok files down by IR shape so we can see how much of "lowered"
/// is "lowered into something the apply path can do anything with".
///
/// Categories:
///   * `instant_with_effects`     — Delivery::Instant + non-empty effects
///                                 (the normal shape, real coverage)
///   * `composite_deliver`        — Delivery::Method + non-empty effects
///                                 (post-MixedBody-relax shape: projectile
///                                  + caster self-effect like dash)
///   * `deliver_only_empty`       — Delivery::Method + EMPTY effects
///                                 (the gap #139 shape: on_hit content
///                                  is silently dropped; program looks
///                                  cosmetically "ok" but does nothing)
///   * `instant_empty`            — Delivery::Instant + EMPTY effects
///                                 (shouldn't happen — would mean the
///                                  ability declared nothing actionable)
///
/// Prints the breakdown via --nocapture; asserts only that
/// `instant_with_effects + composite_deliver` (the genuinely-actionable
/// shapes) is at least 1 (so a future regression that drops it to 0
/// surfaces loudly).
#[test]
fn lol_corpus_lowering_honesty_audit() {
    use engine::ability::program::Delivery;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("dataset")
        .join("abilities")
        .join("lol_heroes");
    if !dir.is_dir() {
        eprintln!("dataset/abilities/lol_heroes not found at {}", dir.display());
        return;
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "ability"))
        .collect();
    files.sort();

    let mut instant_with_effects = 0usize;
    let mut composite_deliver    = 0usize;
    let mut deliver_only_empty   = 0usize;
    let mut instant_empty        = 0usize;
    // Inventory the on-hit-only files for spot-checking.
    let mut deliver_only_examples: Vec<String> = Vec::new();

    for path in &files {
        let src = std::fs::read_to_string(path).expect("read .ability");
        let file = match parse_ability_file(&src) { Ok(f) => f, Err(_) => continue };
        for decl in &file.abilities {
            let prog = match lower_ability_decl(decl) { Ok(p) => p, Err(_) => continue };
            let has_effects = !prog.effects.is_empty();
            let is_deliver = matches!(prog.delivery, Delivery::Method { .. });
            match (is_deliver, has_effects) {
                (false, true)  => instant_with_effects += 1,
                (true,  true)  => composite_deliver += 1,
                (true,  false) => {
                    deliver_only_empty += 1;
                    if deliver_only_examples.len() < 6 {
                        deliver_only_examples.push(decl.name.clone());
                    }
                }
                (false, false) => instant_empty += 1,
            }
        }
    }

    let total = instant_with_effects + composite_deliver + deliver_only_empty + instant_empty;
    let actionable = instant_with_effects + composite_deliver;

    eprintln!("LoL canary honesty audit ({total} ability decls in {} files):", files.len());
    eprintln!("  instant_with_effects : {instant_with_effects:>4}   (real coverage)");
    eprintln!("  composite_deliver    : {composite_deliver:>4}   (deliver+trailing self-effect)");
    eprintln!("  deliver_only_empty   : {deliver_only_empty:>4}   (gap #139 — on_hit silently dropped)");
    eprintln!("  instant_empty        : {instant_empty:>4}   (declares nothing actionable)");
    eprintln!("  ─────────────────────────────");
    eprintln!("  actionable           : {actionable:>4}   ({:.1}%)", 100.0 * actionable as f32 / total.max(1) as f32);
    eprintln!("  empty (any shape)    : {:>4}   ({:.1}%)",
              deliver_only_empty + instant_empty,
              100.0 * (deliver_only_empty + instant_empty) as f32 / total.max(1) as f32);
    if !deliver_only_examples.is_empty() {
        eprintln!("\n  Sample deliver-only-empty decls (first 6):");
        for name in &deliver_only_examples { eprintln!("    - {name}"); }
    }

    assert!(actionable >= 1, "canary regression: zero actionable abilities");
}
