//! Diagnostic: dump the resolved view annotations + decl ordering
//! to find where `@materialized` lands when it disappears off the
//! kill_count view.

#[test]
fn predator_prey_min_dump_decl_annotations() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read");
    let program = dsl_compiler::parse(&src).expect("parse");
    eprintln!("[dump] {} top-level decls", program.decls.len());
    for (i, decl) in program.decls.iter().enumerate() {
        let (kind, name, anns): (&str, String, Vec<String>) = match decl {
            dsl_compiler::ast::Decl::Entity(d) => (
                "entity",
                d.name.clone(),
                d.annotations.iter().map(|a| a.name.clone()).collect(),
            ),
            dsl_compiler::ast::Decl::Event(d) => (
                "event",
                d.name.clone(),
                d.annotations.iter().map(|a| a.name.clone()).collect(),
            ),
            dsl_compiler::ast::Decl::View(d) => (
                "view",
                d.name.clone(),
                d.annotations.iter().map(|a| a.name.clone()).collect(),
            ),
            dsl_compiler::ast::Decl::Query(d) => (
                "query",
                d.name.clone(),
                d.annotations.iter().map(|a| a.name.clone()).collect(),
            ),
            dsl_compiler::ast::Decl::SpatialQuery(d) => (
                "spatial_query",
                d.name.clone(),
                d.annotations.iter().map(|a| a.name.clone()).collect(),
            ),
            dsl_compiler::ast::Decl::Physics(d) => (
                "physics",
                d.name.clone(),
                d.annotations.iter().map(|a| a.name.clone()).collect(),
            ),
            dsl_compiler::ast::Decl::Config(d) => (
                "config",
                d.name.clone(),
                d.annotations.iter().map(|a| a.name.clone()).collect(),
            ),
            dsl_compiler::ast::Decl::Invariant(d) => (
                "invariant",
                d.name.clone(),
                d.annotations.iter().map(|a| a.name.clone()).collect(),
            ),
            other => ("other", format!("{other:?}").chars().take(40).collect(), vec![]),
        };
        eprintln!("  [{i}] {kind} {name} annotations={anns:?}");
    }
}
