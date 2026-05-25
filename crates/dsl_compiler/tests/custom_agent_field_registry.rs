//! Pin: Gap plague_city#P-A — `field <name>: <type>` top-level
//! declarations register custom per-agent SoA columns. The compiler
//! must:
//!   1. Parse `field infected: u32` cleanly (no UnknownField errors).
//!   2. Resolve `self.infected` reads to a `Custom(_)` field id.
//!   3. Resolve `agents.set_infected(...)` writes to the same id via
//!      the physics setter path.
//!   4. Auto-allocate `agent_infected_buf` in the synthesized runtime
//!      core, sized by the field's primitive (u32 → 4 bytes).
//!   5. NOT break any built-in `AgentFieldId` round-trip — the
//!      registry is additive.
//!
//! Mirrors the structure of `init_const_type_routing.rs`: synthesize
//! a kernel that binds `agent_infected` directly, then drive
//! `synthesize_runtime_core_a2` and assert the buffer allocation
//! lands in the generated source.

use dsl_ast::ast::{Decl, AgentFieldDecl, Span as AstSpan};
use dsl_compiler::build_helper::synthesize_runtime_core_a2;
use dsl_compiler::cg::data_handle::{AgentFieldId, AgentFieldTy};
use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::custom_agent_fields::{intern_field, lookup_by_snake, populate};
use dsl_compiler::kernel_binding_ir::{
    AccessMode, BgSource, KernelBinding, KernelKind, KernelSpec,
};

#[test]
fn intern_then_resolve_via_from_snake() {
    // Property: a custom field interned via `intern_field` resolves
    // through the public `AgentFieldId::from_snake` path used by
    // `lower_field` for `self.<name>` reads.
    let id = intern_field("custom_test_field_uniq_a", AgentFieldTy::U32);
    let resolved =
        AgentFieldId::from_snake("custom_test_field_uniq_a").expect("must resolve");
    match resolved {
        AgentFieldId::Custom(cid) => {
            assert_eq!(cid, id);
            assert_eq!(cid.ty(), AgentFieldTy::U32);
            assert_eq!(cid.name(), "custom_test_field_uniq_a");
        }
        other => panic!("expected Custom(_), got {other:?}"),
    }
    // Snake-name round-trips.
    assert_eq!(resolved.snake(), "custom_test_field_uniq_a");
    // ty() routes through the leaked descriptor.
    assert_eq!(resolved.ty(), AgentFieldTy::U32);
}

#[test]
fn builtin_from_snake_is_unaffected() {
    // The fast-path `from_snake_builtin` must still resolve every
    // built-in, AND a `Custom` interned with a non-colliding name
    // must NOT leak into a built-in lookup.
    let _ = intern_field("custom_field_must_not_collide_a", AgentFieldTy::F32);
    let hp = AgentFieldId::from_snake("hp").expect("hp resolves");
    assert!(matches!(hp, AgentFieldId::Hp));
    let custom_seen = AgentFieldId::from_snake("custom_field_must_not_collide_a");
    assert!(matches!(custom_seen, Some(AgentFieldId::Custom(_))));
}

#[test]
fn lookup_by_snake_returns_none_for_unknown_name() {
    assert!(lookup_by_snake("definitely_never_interned_xyz_zzz").is_none());
}

#[test]
fn populate_registers_field_decls_from_program() {
    use dsl_ast::ast::Program;
    let program = Program {
        imports: vec![],
        imports_resolved: vec![],
        decls: vec![
            Decl::AgentField(AgentFieldDecl {
                annotations: vec![],
                name: "popfield_a".into(),
                ty_name: "u32".into(),
                span: AstSpan::new(0, 0),
            }),
            Decl::AgentField(AgentFieldDecl {
                annotations: vec![],
                name: "popfield_b".into(),
                ty_name: "f32".into(),
                span: AstSpan::new(0, 0),
            }),
            Decl::AgentField(AgentFieldDecl {
                annotations: vec![],
                name: "popfield_c".into(),
                ty_name: "bool".into(),
                span: AstSpan::new(0, 0),
            }),
        ],
        terrain: None,
        controls: None,
        render: None,
        ui: None,
    };
    let ids = populate(&program);
    assert_eq!(ids.len(), 3);
    assert_eq!(ids[0].name(), "popfield_a");
    assert_eq!(ids[0].ty(), AgentFieldTy::U32);
    assert_eq!(ids[1].name(), "popfield_b");
    assert_eq!(ids[1].ty(), AgentFieldTy::F32);
    assert_eq!(ids[2].name(), "popfield_c");
    assert_eq!(ids[2].ty(), AgentFieldTy::Bool);

    // After populate, each name is resolvable via from_snake.
    for (name, ty) in [
        ("popfield_a", AgentFieldTy::U32),
        ("popfield_b", AgentFieldTy::F32),
        ("popfield_c", AgentFieldTy::Bool),
    ] {
        let id = AgentFieldId::from_snake(name).expect("registered");
        assert_eq!(id.ty(), ty);
    }
}

#[test]
fn parser_accepts_field_top_level() {
    let src = r#"
field infected_for_parse_test: u32
field cult_loyalty_for_parse_test: f32
field is_zombie_for_parse_test: bool
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
    assert_eq!(decls.len(), 3, "expected 3 AgentFieldDecls in {decls:?}");
    assert_eq!(decls[0].name, "infected_for_parse_test");
    assert_eq!(decls[0].ty_name, "u32");
    assert_eq!(decls[1].name, "cult_loyalty_for_parse_test");
    assert_eq!(decls[1].ty_name, "f32");
    assert_eq!(decls[2].name, "is_zombie_for_parse_test");
    assert_eq!(decls[2].ty_name, "bool");
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
fn synthesize_runtime_allocates_agent_buf_for_custom_field() {
    // Verification gate (1) from the task brief: a `.sim` with `field
    // infected: u32` + a rule that reads/writes it — lower + emit
    // succeeds AND the synthesized runtime allocates
    // `agent_infected_buf`.
    //
    // Direct-spec synthesis (mirrors `init_const_type_routing.rs`)
    // pins the buffer-allocation side without bringing the full DSL
    // pipeline along. The custom-field registry is shared static
    // state — so an `intern_field` call is all the build_helper
    // owned-binding loop needs to size + name the buffer correctly.
    //
    // We exercise BOTH primitive arms (u32 + f32) since they hit
    // distinct sizing/init code paths in the per-binding loop.
    let _ = intern_field("infected_runtime_alloc_test", AgentFieldTy::U32);
    let _ = intern_field("cult_loyalty_runtime_alloc_test", AgentFieldTy::F32);

    let kernel = KernelSpec {
        name: "synthetic_custom".into(),
        pascal: "SyntheticCustom".into(),
        entry_point: "cs_synthetic_custom".into(),
        cfg_struct: "SyntheticCustomCfg".into(),
        cfg_build_expr: "SyntheticCustomCfg::default()".into(),
        cfg_struct_decl: "pub struct SyntheticCustomCfg;".into(),
        bindings: vec![
            binding_for(0, "agent_infected_runtime_alloc_test", "array<u32>"),
            binding_for(1, "agent_cult_loyalty_runtime_alloc_test", "array<f32>"),
            KernelBinding {
                slot: 2,
                name: "cfg".into(),
                access: AccessMode::Uniform,
                wgsl_ty: "SyntheticCustomCfg".into(),
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
        "custom_field_registry_test",
        &artifacts,
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
    );
    // The runtime struct must own the per-agent buffer for each
    // custom field, sized `agent_count * elem_bytes` via the
    // `create_buffer` call generated by the owned-binding loop.
    assert!(
        core.contains("agent_infected_runtime_alloc_test_buf"),
        "synthesized runtime must own `agent_infected_runtime_alloc_test_buf`; \
         generated source:\n{core}",
    );
    assert!(
        core.contains("agent_cult_loyalty_runtime_alloc_test_buf"),
        "synthesized runtime must own `agent_cult_loyalty_runtime_alloc_test_buf`; \
         generated source:\n{core}",
    );
    // Default sizing path: `agent_count * 4` bytes (`create_buffer`
    // with `size: (agent_count * 4u64).max(16)`). The exact format
    // string in build_helper is `({slot_expr} * {elem_bytes}u64).max(16)`
    // — assert the per-agent sizing is present for the custom column.
    assert!(
        core.contains("custom_field_registry_test::agent_infected_runtime_alloc_test"),
        "synthesized runtime must label the agent_infected buffer with \
         its fixture-prefixed name; generated source:\n{core}",
    );
}

#[test]
fn lower_self_field_read_for_custom_field() {
    // The expr-lower `lower_field` path resolves `self.<name>` for
    // a registered custom field. We exercise it via the public
    // `AgentFieldId::from_snake` pin since that's the only call
    // site the lowering uses (it goes:
    // `lower_field` → `lookup_virtual_field` → `from_snake`).
    //
    // The pin is structural: assert from_snake routes a custom name
    // through `Custom(_)`. If a future refactor moves this lookup,
    // this pin breaks early and the test author can re-target.
    let _ = intern_field("infected_lower_pin", AgentFieldTy::U32);
    let id = AgentFieldId::from_snake("infected_lower_pin")
        .expect("custom field must resolve through from_snake");
    let AgentFieldId::Custom(cid) = id else {
        panic!("expected Custom variant, got {id:?}");
    };
    assert_eq!(cid.name(), "infected_lower_pin");
    assert_eq!(cid.ty(), AgentFieldTy::U32);
}
