//! Per-event payload layout regression test for the predator_prey
//! Stage 2 fixture. The runtime today shares a single `EventRingId(0)`
//! across every event kind (per the design comment in
//! `cg/lower/driver.rs::populate_event_kinds`), but each kind owns a
//! distinct [`EventLayout`] with its own field-index registry — that's
//! the meaningful "per-event allocation" surface fixtures rely on.
//!
//! This test pins three properties for `predator_prey_min.sim`:
//!   1. Both `Tick` and `Killed` get registered as event kinds.
//!   2. `Killed` declares its three payload fields (`by`, `prey`, `pos`)
//!      in the field-index registry (Vec3 → 3 words).
//!   3. The compile pipeline (parse → resolve → CG lower → schedule
//!      → emit) succeeds end-to-end with the `event Killed { ... }`
//!      declaration present, even though no rule emits or consumes it
//!      yet — Stage 5 / Stage 7 wire that.

#[test]
fn predator_prey_min_per_event_layout_pins() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read predator_prey_min.sim");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("CG lower");

    // Both Tick and Killed should be registered.
    let event_names: Vec<&str> = cg
        .interner
        .event_kinds
        .values()
        .map(String::as_str)
        .collect();
    assert!(
        event_names.contains(&"Tick"),
        "Tick missing from event_kinds: {:?}",
        event_names
    );
    assert!(
        event_names.contains(&"Killed"),
        "Killed missing from event_kinds: {:?}",
        event_names
    );

    // Killed should have its three GPU-representable fields registered
    // in declaration order. Vec3 occupies 3 words but counts as one
    // declared-field index.
    let killed_kind = cg
        .interner
        .event_kinds
        .iter()
        .find_map(|(k, n)| (n == "Killed").then_some(*k))
        .expect("Killed event kind allocated");
    let layout = cg
        .event_layouts
        .get(&killed_kind)
        .expect("Killed event layout registered");
    let mut field_names: Vec<&str> = layout.fields.keys().map(String::as_str).collect();
    field_names.sort();
    assert_eq!(
        field_names,
        vec!["by", "pos", "prey"],
        "Killed payload fields mismatch: {:?}",
        field_names
    );

    // End-to-end pipeline still succeeds. Emit must not regress.
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit predator_prey_min CG program");
}
