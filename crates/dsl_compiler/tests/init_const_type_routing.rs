//! Pin: `init { hp: 100, mana: 50 }` lowers to the correct primitive
//! type per column, NOT a uniform `vec![Nu32; ...]`.
//!
//! ## Why this matters (Gap G — TYPE CONFUSION)
//!
//! Pre-fix the build_helper init lowering treated every `InitExpr::Const`
//! as `vec![{n}u32; agent_count]` and bytemuck-cast the slice into the
//! target buffer regardless of the column's primitive type. For an f32
//! SoA column (hp, max_hp, mana, ...) that wrote the **u32 bit-pattern**
//! of N into the f32 buffer:
//!   * `init { hp: 100 }` ⇒ buffer slot bits = `0x64` ⇒ read as f32 =
//!     `1.4e-43` ⇒ functionally zero.
//! Every `target.hp` read from the scoring kernel was therefore ~0.0
//! from tick 0; verbs that gate on `target.hp > X` see hp=0 and either
//! fire instantly or never. The squad_skirmish_pin had a host-side
//! workaround that overwrote agent_hp_buf after `try_new`.
//!
//! The fix routes the `InitExpr::Const` lowering by the target column's
//! `AgentFieldTy` (resolved via `AgentFieldId::from_snake(col)`):
//!   * F32 columns ⇒ `vec![{n}.0_f32; agent_count]` + cast_slice on f32.
//!   * U32 / Bool columns ⇒ keep the historical `vec![{n}u32; ...]`.
//!
//! ## What this exercises
//!
//! Constructs a synthetic `KernelSpec` with three fixture-owned
//! bindings (`agent_hp`, `agent_mana`, `agent_alive`) and three matching
//! `InitStmt`s, then drives `synthesize_runtime_core_a2` directly and
//! asserts the generated source has the right per-type init slices.
//! Synthetic-spec keeps the test independent of the full DSL→CG→emit
//! stack; the lower / schedule / emit pipeline is exercised by the
//! sims pin tests downstream.

use dsl_ast::ast::{InitExpr, InitStmt, Span};
use dsl_compiler::build_helper::synthesize_runtime_core_a2;
use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::kernel_binding_ir::{
    AccessMode, BgSource, KernelBinding, KernelKind, KernelSpec,
};

fn binding(slot: u32, name: &str, wgsl_ty: &str) -> KernelBinding {
    KernelBinding {
        slot,
        name: name.into(),
        access: AccessMode::ReadStorage,
        wgsl_ty: wgsl_ty.into(),
        bg_source: BgSource::External(name.into()),
    }
}

fn synthesize_with_init(init_stmts: &[InitStmt]) -> String {
    // One synthetic kernel with three fixture-owned bindings — agent_hp
    // (f32), agent_mana (f32), agent_alive (bool/u32-packed). The
    // build helper's owned-binding alloc loop walks
    // `artifacts.kernel_specs[*].bindings`, so a single kernel with
    // these bindings is enough to exercise the init dispatch.
    let kernel = KernelSpec {
        name: "synthetic".into(),
        pascal: "Synthetic".into(),
        entry_point: "cs_synthetic".into(),
        cfg_struct: "SyntheticCfg".into(),
        cfg_build_expr: "SyntheticCfg::default()".into(),
        cfg_struct_decl: "pub struct SyntheticCfg;".into(),
        bindings: vec![
            binding(0, "agent_hp", "array<f32>"),
            binding(1, "agent_mana", "array<f32>"),
            binding(2, "agent_alive", "array<u32>"),
            // cfg slot last so the spec validates as having exactly one
            // Cfg-source binding (matches every real kernel today).
            KernelBinding {
                slot: 3,
                name: "cfg".into(),
                access: AccessMode::Uniform,
                wgsl_ty: "SyntheticCfg".into(),
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
    synthesize_runtime_core_a2(
        "init_routing_test",
        &artifacts,
        init_stmts,
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
    )
}

fn const_init(field: &str, n: i64) -> InitStmt {
    InitStmt {
        field: field.into(),
        expr: InitExpr::Const(n),
        span: Span::new(0, 0),
    }
}

#[test]
fn init_hp_lowers_to_f32_not_u32_bit_pattern() {
    let core = synthesize_with_init(&[
        const_init("alive", 1),
        const_init("hp", 100),
        const_init("mana", 50),
    ]);

    // The f32 path: hp + mana must be initialised as f32 slices with a
    // float literal (not the u32 bit-pattern).
    assert!(
        core.contains("vec![100.0_f32; agent_count as usize]"),
        "agent_hp_init must lower to `vec![100.0_f32; ...]` for the f32 \
         hp column (Gap G fix). Generated source:\n{core}",
    );
    assert!(
        core.contains("vec![50.0_f32; agent_count as usize]"),
        "agent_mana_init must lower to `vec![50.0_f32; ...]` for the f32 \
         mana column (Gap G fix). Generated source:\n{core}",
    );

    // The init binding must be typed Vec<f32>, not Vec<u32>.
    assert!(
        core.contains("let agent_hp_init: Vec<f32> ="),
        "agent_hp_init must be a Vec<f32> binding. Generated source:\n{core}",
    );
    assert!(
        core.contains("let agent_mana_init: Vec<f32> ="),
        "agent_mana_init must be a Vec<f32> binding. Generated source:\n{core}",
    );

    // Negative assertion: the pre-fix wrong path (`vec![100u32; ...]`
    // for hp / `vec![50u32; ...]` for mana) must NOT appear.
    assert!(
        !core.contains("vec![100u32; agent_count as usize]"),
        "Gap G regression: agent_hp init still using `vec![100u32; ...]` \
         — that writes the u32 bit-pattern (0x64) into the f32 buffer. \
         Generated source:\n{core}",
    );
    assert!(
        !core.contains("vec![50u32; agent_count as usize]"),
        "Gap G regression: agent_mana init still using `vec![50u32; ...]` \
         — that writes the u32 bit-pattern into the f32 buffer. \
         Generated source:\n{core}",
    );
}

#[test]
fn init_alive_keeps_u32_path() {
    let core = synthesize_with_init(&[const_init("alive", 1)]);

    // Bool column: alive flag is u32-packed on GPU, so the historical
    // `vec![1u32; ...]` path is correct.
    assert!(
        core.contains("vec![1u32; agent_count as usize]"),
        "agent_alive_init must keep `vec![1u32; ...]` for the u32-packed \
         bool alive column. Generated source:\n{core}",
    );
    assert!(
        core.contains("let agent_alive_init: Vec<u32> ="),
        "agent_alive_init must be a Vec<u32> binding. Generated source:\n{core}",
    );
}

#[test]
fn init_slot_expr_routes_by_column_type() {
    // `init { hp: slot }` would write per-slot values. For an f32
    // column the slice must be Vec<f32>; for a u32 column Vec<u32>.
    let core = synthesize_with_init(&[
        InitStmt { field: "hp".into(), expr: InitExpr::Slot, span: Span::new(0, 0) },
        InitStmt { field: "alive".into(), expr: InitExpr::Slot, span: Span::new(0, 0) },
    ]);
    // f32 column: cast index to f32.
    assert!(
        core.contains("(0..agent_count).map(|i| i as f32).collect::<Vec<f32>>()"),
        "agent_hp Slot init must produce a Vec<f32>. Generated source:\n{core}",
    );
    // u32 column: keep the historical collect.
    assert!(
        core.contains("(0..agent_count).collect::<Vec<u32>>()"),
        "agent_alive Slot init must produce a Vec<u32>. Generated source:\n{core}",
    );
}
