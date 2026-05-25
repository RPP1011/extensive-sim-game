//! Pin: `@host_callable` codegen emits a typed injector for events
//! whose kind id is fixture-allocated (Gap plague_city#P-B).
//!
//! Pre-fix: `build_helper::synthesize_runtime_core_a2`'s
//! `@host_callable` walk required `EventIR::engine_kind_id` to be
//! `Some(_)` — events whose name had no entry in
//! `dsl_ast::engine_events::engine_event_kind_id_for_name` were dropped
//! with a `cargo:warning=[<fixture> host_callable] event ... has no
//! engine kind id; skipping codegen` and NO method was emitted. The
//! plague_city fixture's `Infected` event (no engine alias) hit this
//! exact gate, forcing the pin to host-side direct write of
//! `agent_hunger_buf` to seed the initial outbreak.
//!
//! Post-fix (this change): the codegen path falls back to the
//! sequential `EventKindId(i)` allocation the lowering driver
//! (`cg::lower::driver::populate_event_kinds`) performs — the
//! dispatcher already handles arbitrary kind ids via the unified event
//! ring, so the codegen gate was overly narrow. The synthesized
//! injector matches the engine-aliased shape: 10-word record, slot 0 =
//! kind id, slot 1 = `self.tick as u32`, slots 2..= = declared fields.
//!
//! This test pins:
//!   1. A fixture-defined `@host_callable event Infected { patient,
//!      source }` declared as the third event in `comp.events` (index
//!      = 2) gets a `pub fn infected(...)` injector method.
//!   2. The method's `record[0]` literal equals `2u32` — the dynamic
//!      kind id matches the event's position in `comp.events`.
//!   3. The method signature carries one `u32` parameter per AgentId
//!      field, in declaration order.
//!   4. An engine-aliased event in the same `events` slice still uses
//!      its hardcoded kind id (engine alias takes precedence over the
//!      sequential fallback).

use dsl_ast::ast::{Annotation, Span};
use dsl_ast::ir::{EventField, EventIR, IrType};
use dsl_compiler::build_helper::synthesize_runtime_core_a2;
use dsl_compiler::cg::emit::EmittedArtifacts;

fn host_callable_event(name: &str, fields: &[(&str, IrType)]) -> EventIR {
    let span = Span::new(0, 0);
    EventIR {
        name: name.into(),
        fields: fields
            .iter()
            .map(|(n, ty)| EventField {
                name: (*n).into(),
                ty: ty.clone(),
                span,
            })
            .collect(),
        tags: Vec::new(),
        annotations: vec![Annotation {
            name: "host_callable".into(),
            args: Vec::new(),
            span,
        }],
        span,
        engine_kind_id: None,
    }
}

#[test]
fn host_callable_fixture_event_emits_typed_injector_with_sequential_kind_id() {
    let span = Span::new(0, 0);

    // Three events in `comp.events`. Indices 0 and 1 are non-host-
    // callable (they pad the kind-id allocation so `Infected` lands at
    // index 2, matching plague_city's actual ordering of
    // Tick(0)/Died(1)/...). Index 2 is the fixture-defined
    // `@host_callable event Infected`.
    let pad_a = EventIR {
        name: "Tick".into(),
        fields: vec![],
        tags: Vec::new(),
        annotations: Vec::new(),
        span,
        engine_kind_id: None,
    };
    let pad_b = EventIR {
        name: "Died".into(),
        fields: vec![EventField {
            name: "victim".into(),
            ty: IrType::AgentId,
            span,
        }],
        tags: Vec::new(),
        annotations: Vec::new(),
        span,
        engine_kind_id: None,
    };
    let infected = host_callable_event(
        "Infected",
        &[("patient", IrType::AgentId), ("source", IrType::AgentId)],
    );

    let events = vec![pad_a, pad_b, infected];

    let artifacts = EmittedArtifacts::default();
    let out = synthesize_runtime_core_a2(
        "plague_city_smoke",
        &artifacts,
        &[],
        &std::collections::BTreeMap::new(),
        &[],
        &events,
        None,
        &[],
        false,
        false, // binds_navgrid
        &[],
        0,
        0,
        false,
        None,
        "{\"bindings\":[]}",
        "{\"arena_radius\":0.0,\"camera\":\"Observer\",\"agents\":[],\"vfx\":[]}",
        "{\"hud\":[],\"screens\":[]}",
    );

    // No `cargo:warning` codepath should fire — the typed method must
    // exist with the snake_case name and `u32` params.
    assert!(
        out.contains("pub fn infected(&mut self, patient: u32, source: u32)"),
        "expected typed `infected` injector for fixture-defined event\n--- source ---\n{out}"
    );

    // The skip stub from the pre-fix path must NOT appear.
    assert!(
        !out.contains("@host_callable on `Infected` skipped"),
        "fixture-defined @host_callable event should not be skipped\n--- source ---\n{out}"
    );

    // Kind id at slot 0 equals the event's position in `comp.events`
    // (Infected is index 2). This mirrors
    // `cg::lower::driver::populate_event_kinds`:
    //     `let kind_id = EventKindId(event.engine_kind_id.unwrap_or(i as u32));`
    assert!(
        out.contains("record[0] = 2;"),
        "expected `record[0] = 2;` (sequential kind id from event index)\n--- source ---\n{out}"
    );

    // Tick at slot 1.
    assert!(
        out.contains("record[1] = self.tick as u32;"),
        "expected tick stamp at slot 1\n--- source ---\n{out}"
    );

    // Per-field slot writes in declaration order.
    for (slot, field) in [(2, "patient"), (3, "source")] {
        let expected = format!("record[{slot}] = {field};");
        assert!(
            out.contains(&expected),
            "expected `{expected}`\n--- source ---\n{out}"
        );
    }

    // Body forwards to the generic injector helper.
    assert!(
        out.contains("self.inject_chronicle_record(&record);"),
        "expected forward to inject_chronicle_record\n--- source ---\n{out}"
    );
}

#[test]
fn host_callable_engine_aliased_event_still_uses_engine_kind_id() {
    // Mixed slice: one fixture-defined @host_callable event at index 0
    // followed by an engine-aliased @host_callable event at index 1.
    // The engine-aliased one must keep its hardcoded kind id (not 1).
    let span = Span::new(0, 0);

    let infected = host_callable_event(
        "Infected",
        &[("patient", IrType::AgentId), ("source", IrType::AgentId)],
    );

    // EffectObserveApplied → engine kind id 64 (per tom_probe.sim).
    let observe = EventIR {
        name: "EffectObserveApplied".into(),
        fields: vec![
            EventField {
                name: "actor".into(),
                ty: IrType::AgentId,
                span,
            },
            EventField {
                name: "target".into(),
                ty: IrType::AgentId,
                span,
            },
            EventField {
                name: "target_observer".into(),
                ty: IrType::U32,
                span,
            },
        ],
        tags: Vec::new(),
        annotations: vec![Annotation {
            name: "host_callable".into(),
            args: Vec::new(),
            span,
        }],
        span,
        engine_kind_id: Some(64),
    };

    let events = vec![infected, observe];

    let artifacts = EmittedArtifacts::default();
    let out = synthesize_runtime_core_a2(
        "mixed_smoke",
        &artifacts,
        &[],
        &std::collections::BTreeMap::new(),
        &[],
        &events,
        None,
        &[],
        false,
        false, // binds_navgrid
        &[],
        0,
        0,
        false,
        None,
        "{\"bindings\":[]}",
        "{\"arena_radius\":0.0,\"camera\":\"Observer\",\"agents\":[],\"vfx\":[]}",
        "{\"hud\":[],\"screens\":[]}",
    );

    // Fixture-defined event uses sequential id (index 0 → 0).
    assert!(
        out.contains("pub fn infected(&mut self, patient: u32, source: u32)"),
        "expected `infected` method\n--- source ---\n{out}"
    );
    assert!(
        out.contains("record[0] = 0;"),
        "expected `record[0] = 0;` for fixture-defined event at index 0\n--- source ---\n{out}"
    );

    // Engine-aliased event keeps its hardcoded kind id (64), NOT its
    // sequential index (1). Engine alias wins.
    assert!(
        out.contains(
            "pub fn effect_observe_applied(&mut self, actor: u32, target: u32, target_observer: u32)"
        ),
        "expected `effect_observe_applied` method\n--- source ---\n{out}"
    );
    assert!(
        out.contains("record[0] = 64;"),
        "expected `record[0] = 64;` for engine-aliased event\n--- source ---\n{out}"
    );
    assert!(
        !out.contains("record[0] = 1;"),
        "engine-aliased event must NOT use sequential index 1\n--- source ---\n{out}"
    );
}
