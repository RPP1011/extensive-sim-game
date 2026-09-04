//! Pin: every `@materialized` view in a fixture allocates its OWN
//! `view_storage_<view_name>_primary_buf` (not the shared
//! `view_storage_primary_buf` the legacy emitter produced) and each
//! fold/decay kernel's BGL `view_storage_primary` slot routes to its
//! own per-view buffer.
//!
//! ## Why this matters (per-view storage aliasing — 6-fixture gap)
//!
//! Pre-fix, the auto-emitted `GeneratedRuntime` allocated ONE
//! `view_storage_primary_buf` field on `GeneratedRuntime` and every
//! fold kernel's `view_storage_primary` BGL binding bound to it.
//! With N declared views, all N folds atomicAdd'd into the same
//! per-agent slot — aggregating their counts incoherently.
//!
//! Surfaced by 6 adversarial fixtures (forest_fire, squad_skirmish,
//! plague_city, detective_investigation, palace_coup, among_us). See
//! `docs/architecture/gaps_observed.md` Gap A + Gap D, plus the
//! per-fixture gap docs.
//!
//! ## What this exercises
//!
//! Drives a synthetic 2-view fixture (one per-agent + one pair-keyed
//! Agent×Agent) through the auto-emit pipeline and asserts the
//! generated `runtime_core.rs` source:
//!   1. Declares one `view_storage_<view_name>_primary_buf` field per
//!      view (NOT the shared `view_storage_primary_buf`).
//!   2. Wires each fold kernel's dispatch arm to its own per-view
//!      buffer (`view_storage_primary: &self.view_storage_<view>_primary_buf`).
//!   3. Sizes each per-view buffer for THAT view's shape — per-agent
//!      gets `agent_count`; pair-keyed gets `agent_count * agent_count`.

use dsl_ast::ast::{InitExpr, InitStmt, Span};
use dsl_compiler::build_helper::{
    collect_materialized_views, synthesize_runtime_core_a2, MaterializedViewInfo,
    PairKeyedSecondKey,
};
use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::kernel_binding_ir::{
    AccessMode, BgSource, KernelBinding, KernelKind, KernelSpec,
};

/// Build a synthetic ViewFold KernelSpec for view `view_name`, mirroring
/// the BGL shape `build_view_fold_bindings` emits in production.
fn fold_spec(view_name: &str) -> KernelSpec {
    let kname = format!("fold_{view_name}");
    let pascal = format!("Fold{}", upper_camel(view_name));
    let cfg_struct = format!("{pascal}Cfg");
    KernelSpec {
        name: kname.clone(),
        pascal: pascal.clone(),
        entry_point: format!("cs_{kname}"),
        cfg_struct: cfg_struct.clone(),
        cfg_build_expr: format!("{cfg_struct} {{ event_count: 0, tick: state.tick, second_key_pop: 1, _pad0: 0 }}"),
        cfg_struct_decl: format!(
            "#[repr(C)]\n#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\npub struct {cfg_struct} {{ event_count: u32, tick: u32, second_key_pop: u32, _pad0: u32 }}"
        ),
        bindings: vec![
            KernelBinding {
                slot: 0,
                name: "event_ring".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Resident("batch_events_ring".into()),
            },
            KernelBinding {
                slot: 1,
                name: "event_tail".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Resident("batch_events_tail".into()),
            },
            KernelBinding {
                slot: 2,
                name: "view_storage_primary".into(),
                access: AccessMode::AtomicStorage,
                wgsl_ty: "u32".into(),
                bg_source: BgSource::ViewHandle {
                    accessor: format!("fold_view_{view_name}_handles"),
                    tuple_idx: 0,
                },
            },
            KernelBinding {
                slot: 3,
                name: "view_storage_anchor".into(),
                access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::ViewHandle {
                    accessor: format!("fold_view_{view_name}_handles"),
                    tuple_idx: 1,
                },
            },
            KernelBinding {
                slot: 4,
                name: "view_storage_ids".into(),
                access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::ViewHandle {
                    accessor: format!("fold_view_{view_name}_handles"),
                    tuple_idx: 2,
                },
            },
            KernelBinding {
                slot: 5,
                name: "sim_cfg".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::External("sim_cfg".into()),
            },
            KernelBinding {
                slot: 6,
                name: "cfg".into(),
                access: AccessMode::Uniform,
                wgsl_ty: cfg_struct,
                bg_source: BgSource::Cfg,
            },
        ],
        kind: KernelKind::ViewFold,
        y_dim_override: None,
        runtime_cfg_fields: Vec::new(),
    }
}

fn upper_camel(snake: &str) -> String {
    let mut out = String::new();
    let mut up = true;
    for c in snake.chars() {
        if c == '_' {
            up = true;
        } else if up {
            out.extend(c.to_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}

/// Drives the auto-emitter against two synthetic fold kernels (one
/// per-agent, one pair-keyed Agent×Agent) and returns the generated
/// `runtime_core.rs` source. The per-view buffer + per-kernel
/// binding routing assertions run on this output.
fn synthesize_two_view_fixture() -> String {
    let damage_dealt = fold_spec("damage_dealt");
    let threat_taken = fold_spec("threat_taken");
    let artifacts = EmittedArtifacts {
        kernel_index: vec![damage_dealt.name.clone(), threat_taken.name.clone()],
        kernel_specs: vec![damage_dealt, threat_taken],
        ..Default::default()
    };
    let materialized_views = vec![
        MaterializedViewInfo {
            name: "damage_dealt".into(),
            pair_keyed: None,
            ring_k: None,
            cell_stride_u32: 1,
        },
        MaterializedViewInfo {
            name: "threat_taken".into(),
            pair_keyed: Some(PairKeyedSecondKey::Agent),
            ring_k: None,
            cell_stride_u32: 1,
        },
    ];
    let init_stmts: Vec<InitStmt> = Vec::new();
    let _ = (InitExpr::Const(0), Span::new(0, 0)); // silence imports if unused
    synthesize_runtime_core_a2(
        "two_view_fixture",
        &artifacts,
        &init_stmts,
        &[],
        &std::collections::BTreeMap::new(),
        &[],
        &[],
        Some(PairKeyedSecondKey::Agent),
        &materialized_views,
        false,
        false, // binds_navgrid
        &[],
        &[],
        0,
        0,
        false,
        None,
        "{\"bindings\":[]}",
        "{\"arena_radius\":0.0,\"camera\":\"Observer\",\"agents\":[],\"vfx\":[]}",
        "{\"hud\":[],\"screens\":[]}",
        dsl_compiler::cg::lower::DebugDepth::Off,
    )
}

#[test]
fn per_view_buffer_field_per_view() {
    let core = synthesize_two_view_fixture();

    // Each view declares its own backing field in `GeneratedRuntime`.
    for view in ["damage_dealt", "threat_taken"] {
        let field = format!("pub view_storage_{view}_primary_buf: wgpu::Buffer,");
        assert!(
            core.contains(&field),
            "missing per-view field `{field}` — view_storage aliasing gap regressed.\n\
             Generated source:\n{core}",
        );
    }

    // Negative: the legacy shared `view_storage_primary_buf` field
    // must NOT appear (the rename routes every fold/decay kernel's
    // BGL slot to a per-view buffer; nothing left to allocate the
    // legacy shared buffer).
    assert!(
        !core.contains("pub view_storage_primary_buf:"),
        "legacy shared `view_storage_primary_buf` field present — \
         per-view rename failed and the aliasing gap is back.\n\
         Generated source:\n{core}",
    );
}

#[test]
fn per_view_dispatch_arms_route_to_per_view_buffers() {
    let core = synthesize_two_view_fixture();

    // Each fold kernel's dispatch arm routes its `view_storage_primary`
    // BGL slot to THAT view's per-view buffer.
    for view in ["damage_dealt", "threat_taken"] {
        let routing = format!(
            "view_storage_primary: &self.view_storage_{view}_primary_buf,",
        );
        assert!(
            core.contains(&routing),
            "fold_{view} kernel arm does not route view_storage_primary to its \
             own per-view buffer; expected `{routing}`.\nGenerated source:\n{core}",
        );
    }
}

#[test]
fn per_view_buffer_sizing_matches_view_shape() {
    let core = synthesize_two_view_fixture();

    // The per-agent `damage_dealt` buffer sizes as `agent_count * 4`;
    // the pair-keyed `threat_taken` buffer sizes as
    // `agent_count * agent_count * 4`. Find each buffer's create
    // descriptor and verify the size expression contains the right
    // shape.
    let dd_alloc = find_buffer_alloc(&core, "view_storage_damage_dealt_primary");
    assert!(
        dd_alloc.contains("agent_count as u64")
            && !dd_alloc.contains("agent_count as u64) * (agent_count as u64"),
        "per-agent view damage_dealt should size for agent_count, not N²; \
         alloc snippet:\n{dd_alloc}",
    );

    let tt_alloc = find_buffer_alloc(&core, "view_storage_threat_taken_primary");
    assert!(
        tt_alloc.contains("(agent_count as u64) * (agent_count as u64)"),
        "pair-keyed view threat_taken should size for N²; \
         alloc snippet:\n{tt_alloc}",
    );
}

/// Find the `let <name>_buf = ...create_buffer(...)` block in the
/// generated source and return it. Used by sizing assertions to
/// inspect a single buffer's alloc snippet without scanning the whole
/// (large) generated source.
fn find_buffer_alloc<'a>(src: &'a str, name: &str) -> &'a str {
    let needle = format!("let {name}_buf = gpu.device.create_buffer");
    let start = src.find(&needle).unwrap_or_else(|| {
        panic!("missing buffer alloc for {name} in source:\n{src}")
    });
    let after = &src[start..];
    // Read up to the next blank line / let statement (the alloc block
    // is ~6 lines).
    let end = after.find("        let ").map(|i| i + 8).unwrap_or(after.len().min(400));
    &after[..end.min(800)]
}

#[test]
fn collect_materialized_views_matches_synthetic_fixture() {
    // collect_materialized_views walks `comp.views` directly. Pin the
    // shape against a synthetic 2-view .sim source to guard against
    // regressions in the per-view classifier (e.g. dropping
    // pair-keyed second-key detection, mis-counting Item entities).
    let src = r#"
event Damaged { source: AgentId, target: AgentId, amount: f32 }

@materialized(on_event = [Damaged])
view damage_dealt(source: Agent) -> f32 {
  initial: 0.0,
  on Damaged { source: s, target: _, amount: a } where s == source { self += a }
  clamp: [0.0, 100000.0],
}

@materialized(on_event = [Damaged])
view threat_taken(observer: Agent, source: Agent) -> f32 {
  initial: 0.0,
  on Damaged { source: s, target: _, amount: a } where s == source { self += a }
  clamp: [0.0, 100000.0],
}
"#;
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let views = collect_materialized_views(&comp);
    assert_eq!(views.len(), 2, "expected 2 materialized views, got {views:?}");
    assert_eq!(views[0].name, "damage_dealt");
    assert!(views[0].pair_keyed.is_none(), "damage_dealt is per-agent");
    assert_eq!(views[1].name, "threat_taken");
    assert_eq!(
        views[1].pair_keyed,
        Some(PairKeyedSecondKey::Agent),
        "threat_taken is Agent×Agent pair-keyed",
    );
}
