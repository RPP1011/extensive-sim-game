//! Pin: spatial-grid backing buffers (`spatial_grid_starts` /
//! `spatial_grid_offsets` / `spatial_grid_cells`) are sized by
//! `GRID_DIM³` / `GRID_DIM³ * MAX_PER_CELL`, not by `agent_count`.
//!
//! ## Why this matters (Gap detective#1, hill_raid)
//!
//! Pre-fix, `slot_count_expr` defaulted every fixture-owned binding
//! to `agent_count` slots. The three `spatial_grid_*` buffers index
//! by cell — `[0, GRID_DIM³)` — not by agent. With the boids default
//! `GRID_DIM = 22u`, the per-cell index space is **10 648** but
//! `agent_count` for the detective_investigation fixture is 18:
//! every WGSL read past index 17 returned 0 (silent OOB → zero in
//! storage), collapsing `spatial.nearby_targets(self)` to the empty
//! set. Witnessed events emitted at ~2/tick instead of the expected
//! ~7/tick, and the `view_storage_evidence_primary` fold never
//! accumulated.
//!
//! The hill_raid fixture's "siege didn't animate" failure mode at
//! commit `1c565df9` is the same gap surfaced from a different angle.
//!
//! ## What this exercises
//!
//! Drives a synthetic kernel-spec list that declares the three
//! spatial-grid bindings through `synthesize_runtime_core_a2`
//! directly (no GPU dispatch needed), then asserts:
//!
//! 1. The generated `try_new` allocates each spatial buffer with the
//!    correct cell-keyed size (`GRID_DIM³ + 1` for `_starts` to cover
//!    the `_cell + 1u` lookahead read; `GRID_DIM³` for `_offsets`;
//!    `GRID_DIM³ * MAX_PER_CELL` for `_cells`).
//! 2. None of the spatial buffers fall through to the per-agent
//!    default (`(agent_count as u64) * 4u64.max(16)`).

use dsl_compiler::build_helper::synthesize_runtime_core_a2;
use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::kernel_binding_ir::{
    AccessMode, BgSource, KernelBinding, KernelKind, KernelSpec,
};

/// Mirrors the spatial-grid constants surfaced from
/// `dsl_compiler::cg::emit::spatial`. Hard-coded here so the pin
/// catches accidental constant drift (e.g. someone bumps
/// `WORLD_HALF_EXTENT` / `CELL_SIZE` and `grid_dim()` changes from
/// 22 to 21 — every spatial allocation downstream needs to follow,
/// and this test surfaces the mismatch immediately).
const EXPECTED_GRID_DIM: u64 = 22;
const EXPECTED_NUM_CELLS: u64 = EXPECTED_GRID_DIM * EXPECTED_GRID_DIM * EXPECTED_GRID_DIM;
const EXPECTED_MAX_PER_CELL: u64 = 32;

/// Synthetic per-agent kernel that touches the three spatial-grid
/// buffers — the bare minimum the auto-emitter needs to land
/// fixture-owned `spatial_grid_*_buf` fields on `GeneratedRuntime`.
/// Mirrors a tiled MoveBoid-style kernel's binding shape (the real
/// fixture's `physics_ObserveAndAccrue` reads the same three).
fn spatial_consumer_kernel() -> KernelSpec {
    KernelSpec {
        name: "physics_spatial_consumer".to_string(),
        pascal: "PhysicsSpatialConsumer".to_string(),
        entry_point: "cs_physics_spatial_consumer".to_string(),
        cfg_struct: "PhysicsSpatialConsumerCfg".to_string(),
        cfg_build_expr:
            "PhysicsSpatialConsumerCfg { agent_cap: 0, tick: 0, seed: 0, _pad0: 0 }"
                .to_string(),
        cfg_struct_decl: "#[repr(C)]\n#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\npub struct PhysicsSpatialConsumerCfg { agent_cap: u32, tick: u32, seed: u32, _pad0: u32 }".to_string(),
        bindings: vec![
            KernelBinding {
                slot: 0,
                name: "spatial_grid_cells".to_string(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".to_string(),
                bg_source: BgSource::Pool("spatial_grid_cells".to_string()),
            },
            KernelBinding {
                slot: 1,
                name: "spatial_grid_offsets".to_string(),
                access: AccessMode::AtomicStorage,
                wgsl_ty: "u32".to_string(),
                bg_source: BgSource::Pool("spatial_grid_offsets".to_string()),
            },
            KernelBinding {
                slot: 2,
                name: "spatial_grid_starts".to_string(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".to_string(),
                bg_source: BgSource::Pool("spatial_grid_starts".to_string()),
            },
            KernelBinding {
                slot: 3,
                name: "cfg".to_string(),
                access: AccessMode::Uniform,
                wgsl_ty: "PhysicsSpatialConsumerCfg".to_string(),
                bg_source: BgSource::Cfg,
            },
        ],
        kind: KernelKind::Generic,
        y_dim_override: None,
        runtime_cfg_fields: Vec::new(),
    }
}

/// Drives the auto-emitter against a single spatial-consumer kernel
/// and returns the generated `runtime_core.rs` source.
fn synthesize_spatial_fixture() -> String {
    let consumer = spatial_consumer_kernel();
    let artifacts = EmittedArtifacts {
        kernel_index: vec![consumer.name.clone()],
        kernel_specs: vec![consumer],
        ..Default::default()
    };
    synthesize_runtime_core_a2(
        "spatial_smoke",
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
    )
}

/// Find the `let <name>_buf = ...create_buffer(...)` block in the
/// generated source and return ~600 chars of context. Sufficient to
/// see the `BufferDescriptor` size expression.
fn find_buffer_alloc<'a>(src: &'a str, name: &str) -> &'a str {
    let needle = format!("let {name}_buf = gpu.device.create_buffer");
    let start = src.find(&needle).unwrap_or_else(|| {
        panic!("missing buffer alloc for {name} in generated source:\n{src}")
    });
    let after = &src[start..];
    let end = after.len().min(600);
    &after[..end]
}

#[test]
fn spatial_grid_starts_sized_by_num_cells_plus_one() {
    let core = synthesize_spatial_fixture();
    let alloc = find_buffer_alloc(&core, "spatial_grid_starts");
    let expected_slots = EXPECTED_NUM_CELLS + 1;
    let needle = format!("{expected_slots}u64");
    assert!(
        alloc.contains(&needle),
        "spatial_grid_starts buffer must be sized for `GRID_DIM³ + 1` ({expected_slots}) \
         u32 slots — the WGSL reads `spatial_grid_starts[_cell + 1u]` at every cell, \
         so the trailing slot covers the lookahead. Expected `{needle}` in alloc, got:\n{alloc}",
    );
    assert!(
        !alloc.contains("agent_count as u64"),
        "spatial_grid_starts must NOT route through the per-agent default sizing — \
         that's the Gap detective#1 regression. Alloc snippet:\n{alloc}",
    );
}

#[test]
fn spatial_grid_offsets_sized_by_num_cells() {
    let core = synthesize_spatial_fixture();
    let alloc = find_buffer_alloc(&core, "spatial_grid_offsets");
    let needle = format!("{EXPECTED_NUM_CELLS}u64");
    assert!(
        alloc.contains(&needle),
        "spatial_grid_offsets buffer must be sized for `GRID_DIM³` ({EXPECTED_NUM_CELLS}) \
         u32 slots — one atomic counter per cell. Expected `{needle}` in alloc, got:\n{alloc}",
    );
    assert!(
        !alloc.contains("agent_count as u64"),
        "spatial_grid_offsets must NOT route through the per-agent default sizing. \
         Alloc snippet:\n{alloc}",
    );
}

#[test]
fn spatial_grid_cells_sized_by_num_cells_times_max_per_cell() {
    let core = synthesize_spatial_fixture();
    let alloc = find_buffer_alloc(&core, "spatial_grid_cells");
    let expected_slots = EXPECTED_NUM_CELLS * EXPECTED_MAX_PER_CELL;
    let needle = format!("{expected_slots}u64");
    assert!(
        alloc.contains(&needle),
        "spatial_grid_cells buffer must be sized for `GRID_DIM³ * MAX_PER_CELL` \
         ({expected_slots}) u32 slots — covers the legacy `cell * MAX_PER_CELL + slot` \
         indexing path AND the dense counting-sort scatter path. Expected `{needle}` \
         in alloc, got:\n{alloc}",
    );
    assert!(
        !alloc.contains("agent_count as u64"),
        "spatial_grid_cells must NOT route through the per-agent default sizing. \
         Alloc snippet:\n{alloc}",
    );
}

#[test]
fn no_spatial_buffer_falls_through_to_per_agent_default() {
    // Belt-and-braces guard: if the dispatcher in `slot_count_expr_for_*`
    // ever silently falls through, every spatial buffer would be sized
    // at `agent_count * 4`. Scan each spatial buffer's alloc snippet
    // for the canonical per-agent sizing pattern and fail loudly if
    // it appears anywhere.
    let core = synthesize_spatial_fixture();
    for name in [
        "spatial_grid_starts",
        "spatial_grid_offsets",
        "spatial_grid_cells",
    ] {
        let alloc = find_buffer_alloc(&core, name);
        assert!(
            !alloc.contains("(agent_count as u64 * 4u64).max(16)"),
            "{name} fell through to per-agent default sizing — Gap detective#1 \
             regressed. Alloc snippet:\n{alloc}",
        );
    }
}
