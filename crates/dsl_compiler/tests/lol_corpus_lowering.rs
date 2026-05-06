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
