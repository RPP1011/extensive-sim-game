//! Plan I — slice I.1 parser tests for the `belief` keyword.
//!
//! Pins the parser surface for the four migration targets called out
//! in `docs/superpowers/plans/2026-05-15-belief-primitive.md` (I.6–I.8).
//! Asserts that:
//!
//!   * The `belief` keyword produces a [`Decl::Belief`] node (not a
//!     [`Decl::View`]) with annotations + name + params + return-type
//!     + body + social-merges preserved.
//!   * Propagation handlers (`on <Event> { ... }`) stay inside the
//!     belief body's [`ViewBody::Fold`] handlers list.
//!   * `merge from <agent>: <op>` clauses split out into the
//!     `social_merges: Vec<SocialMergeClause>` field with the op-name
//!     enum populated correctly.
//!   * All four supported merge ops (`bit_or`, `max`, `min`, `replace`)
//!     parse to the matching [`SocialMergeOpName`] variant.
//!
//! Resolver + lowering + emit are covered in I.2 / I.3 / I.4.

use dsl_ast::ast::{Decl, SocialMergeOpName, ViewBody};
use dsl_ast::parser::parse_program;

fn first_belief(src: &str) -> dsl_ast::ast::BeliefDecl {
    let prog = parse_program(src).unwrap_or_else(|e| panic!("parse failed:\n{src}\nerror: {e}"));
    prog.decls
        .into_iter()
        .find_map(|d| match d {
            Decl::Belief(b) => Some(b),
            _ => None,
        })
        .expect("expected one belief decl")
}

#[test]
fn room_known_propagation_only_parses_into_belief_decl() {
    // I.6 migration target: `belief room_known(observer: Agent, room: u32) -> bool`.
    // Single propagation handler on a hand-rolled event; no social-merge clause.
    let src = "\
        event RoomEntered { observer: Agent, room: u32 }\n\
        belief room_known(observer: Agent, room: u32) -> bool {\n\
          initial: false,\n\
          on RoomEntered { observer: obs, room: r } { true }\n\
        }\n";
    let b = first_belief(src);
    assert_eq!(b.name, "room_known");
    assert_eq!(b.params.len(), 2);
    assert_eq!(b.params[0].name, "observer");
    assert_eq!(b.params[1].name, "room");
    assert!(
        b.social_merges.is_empty(),
        "no `merge from` clauses → social_merges should be empty"
    );
    match &b.body {
        ViewBody::Fold { handlers, .. } => {
            assert_eq!(handlers.len(), 1, "one propagation handler expected");
        }
        other => panic!("expected ViewBody::Fold, got {other:?}"),
    }
}

#[test]
fn detected_subject_social_merge_bit_or_parses() {
    // I.7 migration target: pair-keyed bool with a bit_or social merge.
    let src = "\
        event SubjectSeen { observer: Agent, subject: Agent }\n\
        event AllyDied { dead: Agent }\n\
        belief detected_subject(observer: Agent, subject: Agent) -> bool {\n\
          initial: false,\n\
          on SubjectSeen { observer: obs, subject: subj } { true }\n\
          on AllyDied { dead: d } merge from d: bit_or\n\
        }\n";
    let b = first_belief(src);
    assert_eq!(b.name, "detected_subject");
    assert_eq!(b.social_merges.len(), 1);
    let merge = &b.social_merges[0];
    assert_eq!(merge.source_agent_name, "d");
    assert_eq!(merge.op, SocialMergeOpName::BitOr);
    match &b.body {
        ViewBody::Fold { handlers, .. } => {
            assert_eq!(
                handlers.len(),
                1,
                "social-merge clause must split out — only the propagation handler stays"
            );
        }
        other => panic!("expected ViewBody::Fold, got {other:?}"),
    }
}

#[test]
fn merge_op_names_round_trip_to_enum_variants() {
    // Each of the four supported merge-op spellings produces the
    // matching SocialMergeOpName variant.
    let cases = [
        ("bit_or", SocialMergeOpName::BitOr),
        ("max", SocialMergeOpName::Max),
        ("min", SocialMergeOpName::Min),
        ("replace", SocialMergeOpName::Replace),
    ];
    for (op_text, expected) in cases {
        let src = format!(
            "\
            event Tick {{ giver: Agent }}\n\
            belief flag(observer: Agent) -> u32 {{\n\
              initial: 0,\n\
              on Tick {{ giver: g }} merge from g: {op_text}\n\
            }}\n"
        );
        let b = first_belief(&src);
        assert_eq!(b.social_merges.len(), 1);
        assert_eq!(b.social_merges[0].op, expected, "op text was `{op_text}`");
    }
}

#[test]
fn unknown_merge_op_reports_parse_error() {
    // Anything outside the four-op set should fail at parse time
    // with the typed error from `parse_belief_handler`.
    let src = "\
        event Tick { giver: Agent }\n\
        belief flag(observer: Agent) -> u32 {\n\
          initial: 0,\n\
          on Tick { giver: g } merge from g: average\n\
        }\n";
    let err = parse_program(src).expect_err("expected parse error for unknown merge op");
    let msg = format!("{err}");
    assert!(
        msg.contains("merge op")
            && msg.contains("average")
            && msg.contains("bit_or"),
        "error should name the bad op + list valid ops; got: {msg}",
    );
}

#[test]
fn belief_decl_distinct_from_view_decl() {
    // Sanity: `belief` ≠ `view`. A `view` keyword with the same body
    // shape produces Decl::View; only `belief` produces Decl::Belief.
    let view_src = "\
        event Tick { observer: Agent }\n\
        view counter(observer: Agent) -> u32 {\n\
          initial: 0,\n\
          on Tick { observer: obs } { 1 }\n\
        }\n";
    let prog = parse_program(view_src).expect("parse");
    let kinds: Vec<&'static str> = prog
        .decls
        .iter()
        .map(|d| match d {
            Decl::View(_) => "view",
            Decl::Belief(_) => "belief",
            _ => "other",
        })
        .collect();
    assert!(kinds.iter().any(|k| *k == "view"));
    assert!(!kinds.iter().any(|k| *k == "belief"));
}

#[test]
fn where_clause_on_social_merge_preserved() {
    // Plan I-spec example: belief active_threat carries an
    // `on TargetEliminated { ... } where ... merge from giver: replace`
    // — the where-clause must round-trip to SocialMergeClause::where_clause.
    let src = "\
        event TargetEliminated { giver: Agent, target: Agent }\n\
        belief active_threat(observer: Agent, source: Agent) -> u32 {\n\
          initial: 0,\n\
          on TargetEliminated { giver: g, target: t } where t == source merge from g: replace\n\
        }\n";
    let b = first_belief(src);
    assert_eq!(b.social_merges.len(), 1);
    assert!(
        b.social_merges[0].where_clause.is_some(),
        "where-clause must be preserved on the social-merge clause"
    );
    assert_eq!(b.social_merges[0].op, SocialMergeOpName::Replace);
}
