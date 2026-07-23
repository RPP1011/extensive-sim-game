//! Regression pins for user-event ↔ engine-alias kind-id collisions
//! (S5b, 2026-07-22).
//!
//! The engine aliases its chronicle events to hardcoded `EventKindId`
//! discriminants (26..=80 — `dsl_ast::engine_events::ENGINE_EVENT_KIND_IDS`)
//! and the `apply_ability` dispatcher stamps records with those ids.
//! User event kind ids used to be a bare source-order index, so a `.sim`
//! with more than 26 non-aliased events had one silently aliased onto a
//! dispatcher tag — payload words and all. The defect was found by a game
//! fixture with 60 user events whose 27th WAS kind 26, i.e. the dispatcher's
//! `EffectDamageApplied` tag, which is why that game could not express combat
//! through `.ability` programs. That game moved to its own repository on
//! 2026-07-23 (see `docs/superpowers/plans/webband-port.md`); the subject
//! here is now `many_events_ability.sim`, the synthetic built to reproduce
//! it and kept for that purpose.
//!
//! These tests pin the fixed allocation at the resolve/lower level:
//! ids skip the reserved discriminants, stay distinct, and — the
//! compatibility property that keeps the fix cheap — are UNCHANGED for
//! the first 26 user events of every fixture.

use dsl_ast::engine_events::{
    event_kind_ids, is_reserved_engine_kind_id, ENGINE_EVENT_KIND_IDS,
};

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

fn resolve_sim(rel: &str) -> dsl_ast::ir::Compilation {
    let path = workspace_path(rel);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let program = dsl_compiler::parse(&src).unwrap_or_else(|e| panic!("parse {rel}: {e:?}"));
    // Custom `field` decls must be interned BEFORE resolve or every
    // rule touching one is silently dropped (S5's sched_probe note in
    // the webband-port plan). Event declarations are unaffected either
    // way, but the lower-and-inspect test below wants a real program.
    let _ = dsl_compiler::custom_agent_fields::populate(&program);
    dsl_ast::resolve::resolve(program).unwrap_or_else(|e| panic!("resolve {rel}: {e:?}"))
}

/// Shared assertion body: no allocated id may collide with a reserved
/// engine discriminant, and no two events may share an id (a duplicate
/// trips `BuilderError::DuplicateInternEntry` at lowering).
fn assert_no_collisions(label: &str, comp: &dsl_ast::ir::Compilation) -> Vec<u32> {
    let ids = event_kind_ids(&comp.events);
    assert_eq!(ids.len(), comp.events.len());
    for (ev, id) in comp.events.iter().zip(&ids) {
        if ev.engine_kind_id.is_some() {
            // Aliased events legitimately hold a reserved id — that IS
            // the alias.
            continue;
        }
        assert!(
            !is_reserved_engine_kind_id(*id),
            "{label}: user event `{}` was allocated kind {id}, which the engine reserves \
             for `{}` — the apply_ability dispatcher writes records with that tag",
            ev.name,
            ENGINE_EVENT_KIND_IDS
                .iter()
                .find(|(_, k)| k == id)
                .map(|(n, _)| *n)
                .unwrap_or("?"),
        );
    }
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        ids.len(),
        "{label}: duplicate event kind ids allocated",
    );
    ids
}

/// The subject: a fixture past the reserved floor.
///
/// This pin was originally taken on `webband_colony.sim` (60 user events),
/// the fixture that FOUND the defect. The game left this repo on 2026-07-23
/// (see docs/superpowers/plans/webband-port.md), so the pin now runs on
/// `many_events_ability.sim` — the synthetic that was purpose-built for
/// exactly this and kept for exactly this reason. Its 27th event is named
/// `PadWouldCollide` because under the old policy it WAS kind 26, the
/// dispatcher's `EffectDamageApplied` tag.
#[test]
fn many_events_user_events_avoid_reserved_kinds() {
    let comp = resolve_sim("assets/sim/many_events_ability.sim");
    assert!(
        comp.events.len() > 25,
        "this pin is only meaningful past the reserved floor; \
         many_events_ability declares {} events",
        comp.events.len(),
    );
    let ids = assert_no_collisions("many_events_ability", &comp);

    // The exact pre-fix symptom, named: source index 26 used to be
    // allocated kind 26 == EffectDamageApplied.
    let (idx, ev) = comp
        .events
        .iter()
        .enumerate()
        .find(|(_, e)| e.name == "PadWouldCollide")
        .expect("many_events_ability declares PadWouldCollide");
    assert_eq!(idx, 26, "PadWouldCollide is the fixture's 27th event");
    assert_ne!(
        ids[idx], 26,
        "PadWouldCollide is back on the dispatcher's EffectDamageApplied tag",
    );
    assert!(ev.engine_kind_id.is_none());

    // Compatibility: the first 26 user events keep the ids the old
    // sequential policy gave them, which is why no existing fixture's
    // kernels move.
    for (i, id) in ids.iter().take(26).enumerate() {
        assert_eq!(*id as usize, i, "event {i} moved off its historic id");
    }
}

/// Every allowlisted fixture in the tree, swept. Catches a fixture that
/// grows past the floor later without anyone re-reading this note.
#[test]
fn every_fixture_avoids_reserved_kinds() {
    let dir = workspace_path("assets/sim");
    let mut checked = 0usize;
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("read assets/sim")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sim"))
        .collect();
    entries.sort();
    for path in entries {
        let src = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Fixtures with `import` directives need the multi-file driver;
        // skip them here (they are covered by their own suites) rather
        // than half-resolve.
        if src.lines().any(|l| l.trim_start().starts_with("import ")) {
            continue;
        }
        let program = match dsl_compiler::parse(&src) {
            Ok(p) => p,
            Err(_) => continue,
        };
        // Safe to intern across the whole corpus: no two fixtures
        // declare the same custom field name with different types
        // (a conflict would panic in the registry, by design).
        let _ = dsl_compiler::custom_agent_fields::populate(&program);
        let comp = match dsl_ast::resolve::resolve(program) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        assert_no_collisions(&name, &comp);
        checked += 1;
    }
    assert!(checked > 40, "swept only {checked} fixtures — discovery broke");
    println!("  event-kind reservation sweep: {checked} fixtures clean");
}

/// The `many_events_ability` fixture is the executable proof: under the
/// old policy its `PadWouldCollide` (source index 26) and its aliased
/// `EffectDamageApplied` both intern `EventKindId(26)` and lowering
/// fails. Post-fix it lowers, the alias keeps 26, and the pad moves to
/// the first free id above the alias block.
#[test]
fn many_events_ability_alias_and_pad_are_disjoint() {
    let comp = resolve_sim("assets/sim/many_events_ability.sim");
    let ids = assert_no_collisions("many_events_ability", &comp);

    let idx_of = |n: &str| {
        comp.events
            .iter()
            .position(|e| e.name == n)
            .unwrap_or_else(|| panic!("many_events_ability declares {n}"))
    };
    let pad = idx_of("PadWouldCollide");
    let dmg = idx_of("EffectDamageApplied");
    assert_eq!(pad, 26, "PadWouldCollide must sit at the historic collision index");
    assert_eq!(ids[dmg], 26, "the engine alias must keep its discriminant");
    assert_eq!(ids[pad], 33, "the pad takes the first id above the 26..=32 alias run");
    // Six pads later the allocator has exhausted the 33..=38 gap and
    // lands past the whole reserved block.
    assert_eq!(ids[idx_of("Pad32")], 81);

    // And the whole thing must lower cleanly — this is the step that
    // failed with DuplicateInternEntry before the fix.
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| panic!("many_events_ability lower failed: {:?}", o.diagnostics));
    assert_eq!(
        cg.interner.event_kinds.get(&26).map(String::as_str),
        Some("EffectDamageApplied"),
        "kind 26 must intern as the engine alias, not a user event",
    );
    assert_eq!(
        cg.interner.event_kinds.get(&33).map(String::as_str),
        Some("PadWouldCollide"),
    );
}

/// Synthetic minimum: 30 bare user events, no aliases. Independent of
/// any fixture's authoring so a fixture rewrite can't quietly retire the
/// coverage.
#[test]
fn synthetic_thirty_event_sim_skips_the_alias_block() {
    let mut src = String::from("entity Thing : Agent { }\n");
    for i in 0..30 {
        src.push_str(&format!("event Ev{i:02} {{ }}\n"));
    }
    let program = dsl_compiler::parse(&src).expect("synthetic parse");
    let comp = dsl_ast::resolve::resolve(program).expect("synthetic resolve");
    let ids = assert_no_collisions("synthetic-30", &comp);
    assert_eq!(&ids[24..30], &[24, 25, 33, 34, 35, 36]);
}
