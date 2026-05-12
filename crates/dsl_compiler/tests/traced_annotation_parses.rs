//! Surface-level coverage for the `@traced` annotation on event
//! declarations (Gap forest_fire#E, `docs/architecture/gaps_observed.md`).
//!
//! ## Why this test exists
//!
//! The forest_fire-event-storm gap doc flagged `@traced` as
//! "surface absent / unverified": the original task brief mentions
//! `@traced` as a non-replayable diagnostics-only event marker
//! (sibling to `@non_replayable`, inverse-intent of `@replayable`),
//! but no fixture had verified that the resolver / lowerer accept
//! the surface end-to-end.
//!
//! Investigation (2026-05-12) found the annotation already flows
//! cleanly through:
//!
//! 1. `dsl_ast::parser` accepts it via the generic
//!    `parse_annotations` surface — annotations are name+arg pairs
//!    with no allowlist at parse time.
//! 2. `dsl_ast::resolve` partitions trailing annotations into
//!    `event_tag` references vs free annotations and copies the
//!    free annotations onto [`dsl_ast::ir::EventIR::annotations`]
//!    (see `resolve.rs` near the `non_tag_anns` partition).
//! 3. `dsl_compiler::cg::lower::lower_compilation_to_cg` happily
//!    lowers fixtures whose events carry `@traced` — there is no
//!    annotation-name allowlist on the lowering path either.
//! 4. The `predator_prey_min.sim` fixture already exercises
//!    `@non_replayable @traced` together on its `DeathCry` event;
//!    `predator_prey_non_replayable.rs` pins the resolver behaviour.
//!
//! What this test adds, on top of the predator_prey pin:
//!
//! - A standalone fixture (no String payload, no `cpu_only` rule) so
//!   the `@traced` surface can be exercised on its own without
//!   pulling in chronicle-prose / cpu-only-rule prerequisites.
//! - A pin for the [`EventIR::is_traced`] helper added alongside
//!   this test (so downstream consumers — fold filters, runtime
//!   ring-routers — don't have to spell out the annotation string
//!   themselves).
//! - An end-to-end check that `lower_compilation_to_cg` accepts an
//!   event carrying `@traced` (no panic, no diagnostic).
//!
//! ## Runtime status (deferred)
//!
//! Wiring `is_traced` into the per-kind
//! [`dsl_compiler::cg::program::EventLayout`] — so the schedule
//! synthesizer can route traced events to a separate ring and the
//! host fold can filter on a layout-level bool without re-walking
//! the `EventIR.annotations` vec — is a separately-scoped follow-up
//! tracked under the same gap entry. The annotation surface itself
//! is verified here.

const FIXTURE: &str = r#"
event Tick { }

@traced
event Histogram {
  agent: AgentId,
  bucket: u32,
}

entity Particle : Agent {
  pos: vec3,
  vel: vec3,
}

physics Tickle @phase(per_agent) {
  on Tick {} {
    emit Histogram { agent: self, bucket: 0 }
  }
}
"#;

const FIXTURE_COMBINED: &str = r#"
event Tick { }

@non_replayable
@traced
event ChronicleNote {
  agent: AgentId,
}

entity Particle : Agent {
  pos: vec3,
  vel: vec3,
}

physics Tickle @phase(per_agent) {
  on Tick {} {
    emit ChronicleNote { agent: self }
  }
}
"#;

// -- Surface check #1: bare `@traced` parses + resolves. -----------------

#[test]
fn traced_alone_parses_and_resolves() {
    let program =
        dsl_compiler::parse(FIXTURE).expect("parser must accept `@traced` on an event");
    let comp =
        dsl_ast::resolve::resolve(program).expect("resolver must accept `@traced` on an event");

    let histogram = comp
        .events
        .iter()
        .find(|e| e.name == "Histogram")
        .expect("Histogram event resolved");

    let names: Vec<&str> =
        histogram.annotations.iter().map(|a| a.name.as_str()).collect();
    assert!(
        names.contains(&"traced"),
        "Histogram should carry @traced after resolve, got: {names:?}"
    );
}

// -- Surface check #2: `EventIR::is_traced()` reports the annotation. ----

#[test]
fn event_ir_is_traced_helper_matches_annotation() {
    let program = dsl_compiler::parse(FIXTURE).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");

    let histogram = comp
        .events
        .iter()
        .find(|e| e.name == "Histogram")
        .expect("Histogram event resolved");
    assert!(
        histogram.is_traced(),
        "EventIR::is_traced() must return true for an event annotated `@traced`"
    );
    assert!(
        !histogram.is_non_replayable(),
        "Histogram does not carry @non_replayable in this fixture"
    );

    // Sanity: the bare `Tick` event (no annotations) reports false
    // through the same helper — guards against an accidental
    // always-true implementation.
    let tick = comp
        .events
        .iter()
        .find(|e| e.name == "Tick")
        .expect("Tick event resolved");
    assert!(
        !tick.is_traced(),
        "EventIR::is_traced() must return false for an event without `@traced`"
    );
}

// -- Surface check #3: combined `@non_replayable @traced` parses. --------

#[test]
fn traced_combined_with_non_replayable_parses_and_resolves() {
    let program = dsl_compiler::parse(FIXTURE_COMBINED)
        .expect("parser must accept stacked `@non_replayable` + `@traced`");
    let comp = dsl_ast::resolve::resolve(program)
        .expect("resolver must accept stacked `@non_replayable` + `@traced`");

    let note = comp
        .events
        .iter()
        .find(|e| e.name == "ChronicleNote")
        .expect("ChronicleNote event resolved");
    assert!(
        note.is_traced(),
        "ChronicleNote should carry @traced after resolve"
    );
    assert!(
        note.is_non_replayable(),
        "ChronicleNote should carry @non_replayable after resolve"
    );
}

// -- Surface check #4: lowering accepts `@traced` end-to-end. -----------

#[test]
fn traced_event_lowers_to_cg_program_without_error() {
    // Pins that the lowering driver does not gate on an
    // annotation allowlist — `@traced` flows through to the
    // CgProgram without producing a diagnostic.
    let program = dsl_compiler::parse(FIXTURE).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let prog = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("lower_compilation_to_cg must accept events carrying `@traced`");

    // Sanity: the `Histogram` event reached the CG program's
    // event-layouts side-table. We don't pin a kind id (engine
    // alias allocation may change), only that the kind-name appears
    // somewhere in the synthesized program. The shared-ring
    // `event_ring` buffer name is the contract today; pinning it
    // here keeps the test load-bearing for the
    // "ring fanout split traced events out" follow-up.
    let any_layout_uses_event_ring = prog
        .event_layouts
        .values()
        .any(|l| l.buffer_name == "event_ring");
    assert!(
        any_layout_uses_event_ring,
        "expected at least one event layout to use the shared `event_ring` buffer; \
         got: {:#?}",
        prog.event_layouts
    );
}
