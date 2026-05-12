//! Symbolic ability-name surface — `apply_ability <Name> …` (2026-05-12).
//!
//! Closes the silent-mis-dispatch footgun from commit `08cc223e`
//! (squad_skirmish): authors who wrote `apply_ability 1` thinking it
//! was Strike instead got Daze (alphabetically first). The new surface
//! lets authors write `apply_ability Strike` and have the lowerer bind
//! the correct slot against the fixture's `.ability` corpus.
//!
//! This file pins three behaviours:
//!   1. **Resolution** — `apply_ability Strike` rewrites the IR's
//!      ability operand to `LitInt(N)` where `N` is Strike's 1-based
//!      slot in the registry (sorted filenames).
//!   2. **Unknown name** — `apply_ability Smite` (no matching
//!      `.ability` file) surfaces as
//!      `LoweringError::UnknownAbilityName` with the sorted available
//!      names so the user sees the typo at lower time, not as an
//!      obscure WGSL failure downstream.
//!   3. **Backwards compatibility** — `apply_ability 3` (numeric) still
//!      lowers without `ability_names` configured; the legacy fixtures
//!      keep working.

use dsl_ast::ir::{IrExpr, IrStmt};
use dsl_compiler::cg::lower::{lower_compilation_to_cg_with_opts, LoweringError, LowerOpts};

/// Helper: parse + resolve the source. Returns the resolved Compilation.
fn parse_and_resolve(src: &str) -> dsl_ast::ir::Compilation {
    let program = dsl_compiler::parse(src).expect("parse must succeed");
    dsl_ast::resolve::resolve(program).expect("resolve must succeed")
}

fn build_opts(names: &[(&str, u32)]) -> LowerOpts {
    LowerOpts {
        ability_names: names
            .iter()
            .map(|(n, id)| (n.to_string(), *id))
            .collect(),
        ..LowerOpts::default()
    }
}

/// Pin 1A: parser captures `apply_ability Strike` as a symbolic name on
/// the AST. The resolver propagates it onto `IrStmt::ApplyAbility::ability_name`.
#[test]
fn parser_captures_bare_identifier_as_ability_name() {
    let src = r#"
event Tick { }
entity Hero : Agent { }

physics Dispatch @phase(per_agent) {
  on Tick {} where (self.alive) {
    apply_ability Strike by self target self
  }
}
"#;
    let comp = parse_and_resolve(src);
    let handler = &comp.physics[0].handlers[0];
    match &handler.body[0] {
        IrStmt::ApplyAbility { ability_name, .. } => {
            assert_eq!(
                ability_name.as_deref(),
                Some("Strike"),
                "parser should have captured `Strike` as the symbolic ability name"
            );
        }
        other => panic!("expected ApplyAbility, got {other:?}"),
    }
}

/// Pin 1B: with `ability_names = { Daze: 1, Rally: 2, Strike: 3, Volley: 4 }`
/// (the same alphabetical 1-based ordering `PackedAbilityRegistry`
/// builds from sorted filenames), `apply_ability Strike` resolves to
/// `LitInt(3)`. This is the canonical squad_skirmish case from commit
/// `08cc223e` — Strike is slot 3, not slot 1.
#[test]
fn apply_ability_strike_resolves_to_alphabetical_slot() {
    let src = r#"
event Tick { }
entity Hero : Agent { }

physics Dispatch @phase(per_agent) {
  on Tick {} where (self.alive) {
    apply_ability Strike by self target self
  }
}
"#;
    let comp = parse_and_resolve(src);
    let opts = build_opts(&[
        ("Daze", 1),
        ("Rally", 2),
        ("Strike", 3),
        ("Volley", 4),
    ]);
    let cg = lower_compilation_to_cg_with_opts(&comp, opts)
        .expect("named-ability lower must succeed");

    // The CgProgram doesn't expose the IR directly, but we can re-walk
    // the resolved Compilation to confirm the substitution happened —
    // run the same resolve helper twice and lower both, checking the
    // emitted dispatcher kernel's body. Instead of CgProgram poking,
    // assert via a separate path: directly invoke the substitution
    // helper by re-running lower and pinning that NO
    // UnknownAbilityName diagnostic fired. That's the actual
    // user-visible contract.
    //
    // Note: the lowering driver clones the Compilation, mutates the
    // clone, and runs verb_expand on the mutation. We assert (a) lower
    // succeeded (the smoke), (b) the program isn't empty, (c) the
    // ability operand reached the dispatcher (kernel body check).
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit succeeds");
    let dispatcher = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("Dispatch"))
        .map(|(_, b)| b.as_str())
        .expect("Dispatch kernel must emit");
    // The dispatcher's pre-loop block contains the ability_id pulled
    // from the resolved expression. For our LitInt(3) substitution
    // the emit lowers it as `3u` (CG promotes i64 → u32 for the
    // ability slot). Pin the literal so a regression that drops the
    // name→id substitution surfaces here.
    assert!(
        dispatcher.contains("3u") || dispatcher.contains("3i"),
        "Dispatch kernel must carry the resolved AbilityId(3) for `Strike`;\n\
         body excerpt: {}",
        &dispatcher[..dispatcher.len().min(800)],
    );
}

/// Pin 2: `apply_ability Smite` when no `Smite.ability` is registered
/// surfaces as `LoweringError::UnknownAbilityName` with the sorted
/// available list. The lowering still produces a (best-effort) program
/// — the diagnostic is the user-visible signal.
#[test]
fn unknown_ability_name_surfaces_typed_lower_error() {
    let src = r#"
event Tick { }
entity Hero : Agent { }

physics Dispatch @phase(per_agent) {
  on Tick {} where (self.alive) {
    apply_ability Smite by self target self
  }
}
"#;
    let comp = parse_and_resolve(src);
    let opts = build_opts(&[
        ("Daze", 1),
        ("Rally", 2),
        ("Strike", 3),
        ("Volley", 4),
    ]);
    let outcome = lower_compilation_to_cg_with_opts(&comp, opts);
    let diagnostics = match outcome {
        Ok(_) => panic!("expected lower failure with UnknownAbilityName"),
        Err(o) => o.diagnostics,
    };
    let unknown = diagnostics
        .iter()
        .find(|d| matches!(d, LoweringError::UnknownAbilityName { .. }))
        .expect("UnknownAbilityName must appear in diagnostics");
    match unknown {
        LoweringError::UnknownAbilityName { name, available, .. } => {
            assert_eq!(name, "Smite");
            assert_eq!(
                available,
                &vec![
                    "Daze".to_string(),
                    "Rally".to_string(),
                    "Strike".to_string(),
                    "Volley".to_string(),
                ],
                "available list must be the sorted name set so the user can spot the typo"
            );
        }
        _ => unreachable!(),
    }
    // Display rendering pin — the user-facing message must spell
    // out the missing name + available list. Regression coverage so
    // a future refactor that drops the available list silently
    // doesn't slip through.
    let rendered = format!("{unknown}");
    assert!(
        rendered.contains("Smite") && rendered.contains("Strike"),
        "Display of UnknownAbilityName must surface `Smite` + the available list;\n\
         got: {rendered}"
    );
}

/// Pin 3: numeric `apply_ability N` keeps lowering even without
/// `ability_names` populated. The legacy fixtures (squad_skirmish,
/// duel_abilities, etc.) ship with numeric ids; this surface must
/// stay byte-compatible until they migrate.
#[test]
fn numeric_apply_ability_still_lowers_without_ability_names() {
    let src = r#"
event Tick { }
entity Hero : Agent { }

physics Dispatch @phase(per_agent) {
  on Tick {} where (self.alive) {
    apply_ability 3 by self target self
  }
}
"#;
    let comp = parse_and_resolve(src);
    // Parser must NOT have captured `3` as a name — it's a literal.
    match &comp.physics[0].handlers[0].body[0] {
        IrStmt::ApplyAbility { ability_name, ability, .. } => {
            assert!(
                ability_name.is_none(),
                "numeric ability operand must not be captured as a symbolic name"
            );
            // The resolver must have lowered the literal directly,
            // not stubbed it as LitInt(0).
            match &ability.kind {
                IrExpr::LitInt(3) => {}
                other => panic!("expected LitInt(3), got {other:?}"),
            }
        }
        other => panic!("expected ApplyAbility, got {other:?}"),
    }
    // Lower with NO ability_names — should still succeed.
    let _ = lower_compilation_to_cg_with_opts(&comp, LowerOpts::default())
        .expect("numeric lower must succeed without ability_names");
}

/// Pin 4 (companion to 3): reserved DSL identifiers in the ability
/// operand position keep their existing expression-only resolution
/// path. `apply_ability self` is an error today (CgTy mismatch) but
/// the failure shape must NOT mutate into `UnknownAbilityName` — that
/// would be a worse diagnostic.
#[test]
fn reserved_identifier_does_not_become_ability_name() {
    let src = r#"
event Tick { }
entity Hero : Agent { }

physics Dispatch @phase(per_agent) {
  on Tick {} where (self.alive) {
    apply_ability self by self target self
  }
}
"#;
    let comp = parse_and_resolve(src);
    match &comp.physics[0].handlers[0].body[0] {
        IrStmt::ApplyAbility { ability_name, .. } => {
            assert!(
                ability_name.is_none(),
                "`self` must NOT be captured as a symbolic ability name — it's a reserved keyword"
            );
        }
        other => panic!("expected ApplyAbility, got {other:?}"),
    }
}
