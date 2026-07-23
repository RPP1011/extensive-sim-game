//! Pin: Gap dungeon_stealth#3-vec3 — `field <name>: vec3` top-level
//! declarations register a custom per-agent SoA column whose
//! primitive is `AgentFieldTy::Vec3` (the same shape the built-in
//! `pos` / `vel` / `busy_target_pos` columns use).
//!
//! Pre-fix the registry's `parse_field_ty` only accepted
//! `u32 | f32 | bool`; vec3-typed per-agent values had to be split
//! into per-axis scalar columns (the `patrol_origin_x` /
//! `patrol_origin_y` workaround in `dungeon_stealth.sim`). With
//! `vec3` accepted, fixture authors can declare
//! `field patrol_origin: vec3` directly and read/write it via the
//! standard `agents.<field>(self)` / `agents.set_<field>(target,
//! value)` surface.
//!
//! The pins exercise:
//!   1. `parse_field_ty("vec3")` resolves to `AgentFieldTy::Vec3`
//!      (NOT `None`).
//!   2. `intern_field` round-trips a vec3 entry through the
//!      pointer-identity registry.
//!   3. `populate` accepts `field <name>: vec3` decls without
//!      panicking (proves the type-name allowlist update propagates).
//!   4. `AgentFieldId::from_snake` resolves a registered vec3
//!      custom name through `Custom(_)` with the right ty().
//!   5. The synthesized runtime allocates `agent_<name>_buf` for a
//!      vec3 binding sized `agent_count * 16` bytes (std430 vec3 →
//!      vec4-padded). The `create_buffer` call uses the same loop
//!      as u32/f32 customs; the only path-difference is the
//!      `elem_bytes_for_wgsl_ty("array<vec3<f32>>") == 16`
//!      branch.
//!
//! Mirrors the structure of `custom_agent_field_registry.rs` so
//! later compiler refactors that touch one path naturally touch
//! the vec3 pin too.

use dsl_ast::ast::{Decl, AgentFieldDecl, Span as AstSpan};
use dsl_compiler::build_helper::synthesize_runtime_core_a2;
use dsl_compiler::cg::data_handle::{AgentFieldId, AgentFieldTy};
use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::custom_agent_fields::{intern_field, parse_field_ty, populate};
use dsl_compiler::kernel_binding_ir::{
    AccessMode, BgSource, KernelBinding, KernelKind, KernelSpec,
};

#[test]
fn parse_field_ty_accepts_vec3() {
    // Pin (1): the surface type-name allowlist now includes vec3.
    // Pre-fix this returned None and `populate` panicked with
    // "unknown custom field type `vec3`".
    assert_eq!(parse_field_ty("vec3"), Some(AgentFieldTy::Vec3));
}

#[test]
fn intern_then_resolve_vec3_via_from_snake() {
    // Pin (2 + 4): a vec3-typed custom field interned via
    // `intern_field` resolves through the public
    // `AgentFieldId::from_snake` path used by `lower_field` for
    // `self.<name>` reads, and round-trips its primitive type.
    let id = intern_field("patrol_origin_pin_vec3", AgentFieldTy::Vec3);
    let resolved = AgentFieldId::from_snake("patrol_origin_pin_vec3")
        .expect("vec3 custom field must resolve through from_snake");
    let AgentFieldId::Custom(cid) = resolved else {
        panic!("expected Custom(_), got {resolved:?}");
    };
    assert_eq!(cid, id);
    assert_eq!(cid.ty(), AgentFieldTy::Vec3);
    assert_eq!(cid.name(), "patrol_origin_pin_vec3");
    // Snake-name round-trips.
    assert_eq!(resolved.snake(), "patrol_origin_pin_vec3");
}

#[test]
fn populate_accepts_vec3_decls() {
    // Pin (3): build_helper's `populate(&program)` walks every
    // `Decl::AgentField` and asserts the type name is allow-listed.
    // Pre-fix a `vec3` decl panicked here. Now it interns cleanly.
    use dsl_ast::ast::Program;
    let program = Program {
        imports: vec![],
        imports_resolved: vec![],
        decls: vec![
            Decl::AgentField(AgentFieldDecl {
                annotations: vec![],
                name: "popfield_vec3_a".into(),
                ty_name: "vec3".into(),
                span: AstSpan::new(0, 0),
            }),
            Decl::AgentField(AgentFieldDecl {
                annotations: vec![],
                name: "popfield_vec3_b".into(),
                ty_name: "vec3".into(),
                span: AstSpan::new(0, 0),
            }),
        ],
        terrain: None,
        controls: None,
        render: None,
        ui: None,
    };
    let ids = populate(&program);
    assert_eq!(ids.len(), 2);
    assert_eq!(ids[0].name(), "popfield_vec3_a");
    assert_eq!(ids[0].ty(), AgentFieldTy::Vec3);
    assert_eq!(ids[1].name(), "popfield_vec3_b");
    assert_eq!(ids[1].ty(), AgentFieldTy::Vec3);

    // After populate, each name is resolvable via from_snake.
    for name in ["popfield_vec3_a", "popfield_vec3_b"] {
        let id = AgentFieldId::from_snake(name).expect("registered");
        assert_eq!(id.ty(), AgentFieldTy::Vec3);
    }
}

#[test]
fn parser_accepts_vec3_field_top_level() {
    // The `.sim` parser must accept `field <name>: vec3` as a
    // valid top-level decl; the type name flows through as a bare
    // identifier and is validated by the compiler-side interner.
    let src = r#"
field patrol_origin_for_parse_test: vec3
field patrol_step_for_parse_test:   vec3
"#;
    let program = dsl_ast::parser::parse_program(src).expect("parses cleanly");
    let decls: Vec<&AgentFieldDecl> = program
        .decls
        .iter()
        .filter_map(|d| match d {
            Decl::AgentField(d) => Some(d),
            _ => None,
        })
        .collect();
    assert_eq!(decls.len(), 2, "expected 2 AgentFieldDecls in {decls:?}");
    assert_eq!(decls[0].name, "patrol_origin_for_parse_test");
    assert_eq!(decls[0].ty_name, "vec3");
    assert_eq!(decls[1].name, "patrol_step_for_parse_test");
    assert_eq!(decls[1].ty_name, "vec3");
}

fn binding_for(slot: u32, name: &str, wgsl_ty: &str) -> KernelBinding {
    KernelBinding {
        slot,
        name: name.into(),
        access: AccessMode::ReadStorage,
        wgsl_ty: wgsl_ty.into(),
        bg_source: BgSource::External(name.into()),
    }
}

#[test]
fn synthesize_runtime_allocates_vec3_buf() {
    // Pin (5): a `field <name>: vec3` declared in the registry
    // surfaces as an `array<vec3<f32>>` binding when read by a
    // kernel. The build_helper's owned-binding loop sizes its
    // backing buffer through `elem_bytes_for_wgsl_ty`, which
    // already returns 16 for `array<vec3<f32>>` (std430 vec3 →
    // vec4-padded). The `create_buffer` call must use that
    // size — `(agent_count * 16u64).max(16)`.
    let _ = intern_field("patrol_origin_runtime_alloc_vec3", AgentFieldTy::Vec3);

    let kernel = KernelSpec {
        name: "synthetic_vec3_custom".into(),
        pascal: "SyntheticVec3Custom".into(),
        entry_point: "cs_synthetic_vec3_custom".into(),
        cfg_struct: "SyntheticVec3CustomCfg".into(),
        cfg_build_expr: "SyntheticVec3CustomCfg::default()".into(),
        cfg_struct_decl: "pub struct SyntheticVec3CustomCfg;".into(),
        bindings: vec![
            binding_for(
                0,
                "agent_patrol_origin_runtime_alloc_vec3",
                "array<vec3<f32>>",
            ),
            KernelBinding {
                slot: 1,
                name: "cfg".into(),
                access: AccessMode::Uniform,
                wgsl_ty: "SyntheticVec3CustomCfg".into(),
                bg_source: BgSource::Cfg,
            },
        ],
        kind: KernelKind::Generic,
        y_dim_override: None,
        runtime_cfg_fields: Vec::new(),
    };
    let artifacts = EmittedArtifacts {
        kernel_index: vec![kernel.name.clone()],
        kernel_specs: vec![kernel],
        ..Default::default()
    };
    let core = synthesize_runtime_core_a2(
        "custom_field_vec3_test",
        &artifacts,
        &[],
        &[],
        &std::collections::BTreeMap::new(),
        &[],
        &[],
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
        dsl_compiler::cg::lower::DebugDepth::Off,
    );

    // The runtime struct must own the per-agent buffer for the
    // vec3 column.
    assert!(
        core.contains("agent_patrol_origin_runtime_alloc_vec3_buf"),
        "synthesized runtime must own `agent_patrol_origin_runtime_alloc_vec3_buf`; \
         generated source:\n{core}",
    );

    // Sizing must be `agent_count * 16` bytes (std430 vec3 padded
    // to vec4). Pre-fix vec3 was unreachable through the surface
    // — this is the regression gate for the new acceptor wiring.
    // The emitted form is `(agent_count as u64 * 16u64).max(16)`.
    assert!(
        core.contains("agent_count as u64 * 16u64"),
        "vec3 buffer must be sized agent_count * 16 bytes (std430 \
         vec3 → vec4 padded); generated source:\n{core}",
    );

    // Fixture-prefixed label survives the alloc loop.
    assert!(
        core.contains("custom_field_vec3_test::agent_patrol_origin_runtime_alloc_vec3"),
        "synthesized runtime must label the agent_patrol_origin buffer with \
         its fixture-prefixed name; generated source:\n{core}",
    );
}
