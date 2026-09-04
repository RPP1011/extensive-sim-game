//! Regression: a `field` decl must not swallow the NEXT declaration's
//! leading annotation.
//!
//! A field's trailing annotations must stay on the declaration's own source line.
//! Without this guard:
//!
//! ```text
//! field mood: f32
//! @phase(per_agent)
//! physics Drift { ... }
//! ```
//!
//! would parse as "field with a trailing @phase" plus an UNANNOTATED physics rule.
//! The robbed rule would then fail well-formedness checks on every `self` read,
//! with no error pointing back at the field.

use dsl_ast::ast::Decl;

const SRC: &str = "\
entity Worker : Agent { pos: vec3, vel: vec3 }

event Tick { }

field mood: f32
@phase(per_agent)
physics Drift {
  on Tick {} where (self.alive) {
    agents.set_mood(self, self.mood + 1.0);
  }
}
";

/// The annotation belongs to the physics rule, not the field above it.
#[test]
fn leading_annotation_survives_a_preceding_field_decl() {
    let p = dsl_ast::parse(SRC).expect("parse field-then-annotated-physics");

    let field = p
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::AgentField(f) if f.name == "mood" => Some(f),
            _ => None,
        })
        .expect("field mood declared");
    assert!(
        field.annotations.is_empty(),
        "the field stole the next decl's annotation: {:?}",
        field.annotations,
    );

    let physics = p
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::Physics(x) if x.name == "Drift" => Some(x),
            _ => None,
        })
        .expect("physics Drift declared");
    assert!(
        physics.annotations.iter().any(|a| a.name == "phase"),
        "physics Drift lost its @phase annotation: {:?}",
        physics.annotations,
    );
}

/// The same-line form still works — a trailing annotation on the field's own
/// line is genuinely the field's.
#[test]
fn same_line_trailing_annotation_still_binds_to_the_field() {
    const INLINE: &str = "\
entity Worker : Agent { pos: vec3, vel: vec3 }

event Tick { }

field mood: f32 @hot
physics Drift {
  on Tick {} where (self.alive) {
    agents.set_mood(self, self.mood + 1.0);
  }
}
";
    let p = dsl_ast::parse(INLINE).expect("parse inline-annotated field");
    let field = p
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::AgentField(f) if f.name == "mood" => Some(f),
            _ => None,
        })
        .expect("field mood declared");
    assert!(
        field.annotations.iter().any(|a| a.name == "hot"),
        "same-line annotation should bind to the field: {:?}",
        field.annotations,
    );
}

/// The optional trailing semicolon still terminates the decl.
#[test]
fn optional_semicolon_still_accepted() {
    const SEMI: &str = "\
entity Worker : Agent { pos: vec3, vel: vec3 }

event Tick { }

field mood: f32;
@phase(per_agent)
physics Drift {
  on Tick {} where (self.alive) {
    agents.set_mood(self, self.mood + 1.0);
  }
}
";
    let p = dsl_ast::parse(SEMI).expect("parse semicolon-terminated field");
    let physics = p
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::Physics(x) if x.name == "Drift" => Some(x),
            _ => None,
        })
        .expect("physics Drift declared");
    assert!(
        physics.annotations.iter().any(|a| a.name == "phase"),
        "physics Drift lost its @phase after a semicolon-terminated field",
    );
}
