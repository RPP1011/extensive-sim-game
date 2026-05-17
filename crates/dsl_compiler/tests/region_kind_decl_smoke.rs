//! Phase 1 smoke probe for `region_kind` + `region_indices` decls.
//! Per spec `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`
//! §6.1.2.
//!
//! Scope: parse + resolve. No codegen (Phase 4 ships the index
//! build kernel; Phase 2 grammars `index` decls).

use dsl_ast::resolve::resolve;

fn round_trip(src: &str) -> Result<dsl_ast::ir::Compilation, dsl_ast::ResolveError> {
    let prog = dsl_compiler::parse(src).expect("parse");
    resolve(prog)
}

#[test]
fn region_kind_with_max_active_registers_in_ir() {
    let src = "
        region_kind Settlement { max_active = 64 }
    ";
    let comp = round_trip(src).expect("resolve");
    assert_eq!(comp.region_kinds.len(), 1);
    let rk = &comp.region_kinds[0];
    assert_eq!(rk.name, "Settlement");
    assert_eq!(rk.max_active, 64);
    // No paired region_indices → empty list. Validated in pass-2;
    // Phase 1 accepts the unpaired case (Phase 2+ will likely
    // require pairing).
    assert!(rk.index_kind_names.is_empty());
}

#[test]
fn region_indices_merges_into_matching_region_kind_slot() {
    // Phase 2a: each name in `region_indices { ... }` must resolve
    // to a declared `index` decl. The 4 stub decls below satisfy
    // the cross-validation.
    let src = "
        index navgrid(region: VoxelRegion) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap, rebuild_on: manual, build {}
        }
        index vismap(region: VoxelRegion) -> Out {
            storage: bitset_pairs(max_cells = 100),
            cost_class: Heavy, rebuild_on: manual, build {}
        }
        index covermap(region: VoxelRegion) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Medium, rebuild_on: manual, build {}
        }
        index surfacemesh(region: VoxelRegion) -> Out {
            storage: mesh_buffer(max_vertices = 100, max_indices = 100),
            cost_class: Cheap, rebuild_on: manual, build {}
        }
        region_kind Settlement     { max_active = 64 }
        region_kind Building       { max_active = 512 }
        region_indices Settlement  { Navgrid, Vismap, CoverMap, SurfaceMesh }
        region_indices Building    { Navgrid, SurfaceMesh }
    ";
    let comp = round_trip(src).expect("resolve");
    assert_eq!(comp.region_kinds.len(), 2);

    let settlement = comp
        .region_kinds
        .iter()
        .find(|k| k.name == "Settlement")
        .expect("Settlement kind");
    assert_eq!(settlement.max_active, 64);
    assert_eq!(
        settlement.index_kind_names,
        vec!["Navgrid", "Vismap", "CoverMap", "SurfaceMesh"]
    );

    let building = comp
        .region_kinds
        .iter()
        .find(|k| k.name == "Building")
        .expect("Building kind");
    assert_eq!(building.max_active, 512);
    assert_eq!(building.index_kind_names, vec!["Navgrid", "SurfaceMesh"]);
}

#[test]
fn region_indices_without_matching_region_kind_rejected() {
    // Per spec §6.1.2: `region_indices Foo {…}` only valid if a
    // `region_kind Foo { … }` decl exists.
    let src = "
        region_indices BattleSite { Navgrid }
    ";
    let result = round_trip(src);
    assert!(
        result.is_err(),
        "expected resolve error for unpaired region_indices, got {:?}",
        result
    );
}

#[test]
fn duplicate_region_kind_name_rejected() {
    let src = "
        region_kind Settlement { max_active = 64 }
        region_kind Settlement { max_active = 100 }
    ";
    let result = round_trip(src);
    assert!(
        result.is_err(),
        "expected resolve error for duplicate region_kind, got {:?}",
        result
    );
}

#[test]
fn duplicate_region_indices_for_same_kind_rejected() {
    let src = "
        region_kind Settlement { max_active = 64 }
        region_indices Settlement { Navgrid }
        region_indices Settlement { Vismap }
    ";
    let result = round_trip(src);
    assert!(
        result.is_err(),
        "expected resolve error for duplicate region_indices, got {:?}",
        result
    );
}

// =====================================================================
// Phase 2a: `index <name>(region: VoxelRegion) -> <Output> { ... }`
// =====================================================================

#[test]
fn index_decl_parses_and_registers() {
    // Empty build body is allowed in Phase 2a/2b (Phase 4 will
    // require a non-empty body when wiring the kernel). Comments
    // inside the body use `//` line form (DSL parser doesn't
    // handle `/* */` block comments anywhere).
    let src = "
        index navgrid(region: VoxelRegion) -> Walkable {
            storage: per_cell_2d(max_cells = 16384, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: chunk_epoch_advance(region.chunks),
            build {}
        }
    ";
    let comp = round_trip(src).expect("resolve");
    assert_eq!(comp.indices.len(), 1);
    let idx = &comp.indices[0];
    assert_eq!(idx.name, "navgrid");
    assert_eq!(idx.region_param_name, "region");
    assert_eq!(idx.output_type_name, "Walkable");
    match idx.storage {
        dsl_ast::ast::IndexStorageShape::PerCell2d {
            max_cells,
            bytes_per_cell,
        } => {
            assert_eq!(max_cells, 16384);
            assert_eq!(bytes_per_cell, 4);
        }
        _ => panic!("expected per_cell_2d; got {:?}", idx.storage),
    }
    assert_eq!(idx.cost_class, dsl_ast::ast::IndexCostClass::Cheap);
    match &idx.rebuild_on {
        dsl_ast::ast::IndexRebuildTrigger::ChunkEpochAdvance { region_field } => {
            assert_eq!(region_field, "chunks");
        }
        _ => panic!("expected chunk_epoch_advance; got {:?}", idx.rebuild_on),
    }
    assert_eq!(idx.build_body_ast.stmts.len(), 0);
}

#[test]
fn index_decl_with_all_storage_shapes_round_trips() {
    let src = "
        index a(region: VoxelRegion) -> AOut {
            storage: per_cell_3d(max_cells = 100, bytes_per_cell = 8),
            cost_class: Medium,
            rebuild_on: chunk_epoch_advance(region.chunks),
            build {}
        }
        index b(region: VoxelRegion) -> BOut {
            storage: bitset_pairs(max_cells = 4096),
            cost_class: Heavy,
            rebuild_on: manual,
            build {}
        }
        index c(region: VoxelRegion) -> COut {
            storage: mesh_buffer(max_vertices = 50000, max_indices = 150000),
            cost_class: Cheap,
            rebuild_on: chunk_epoch_advance(region.chunks),
            build {}
        }
        index d(region: VoxelRegion) -> DOut {
            storage: sparse_grid(max_cells = 200, bytes_per_cell = 16),
            cost_class: Medium,
            rebuild_on: manual,
            build {}
        }
    ";
    let comp = round_trip(src).expect("resolve");
    assert_eq!(comp.indices.len(), 4);
}

#[test]
fn region_indices_now_validates_index_kinds_against_index_decls() {
    // Phase 2a closes the Phase-1 TODO: every name in `region_indices
    // { ... }` must resolve to a declared `index` decl.
    let src_valid = "
        index navgrid(region: VoxelRegion) -> Walkable {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: chunk_epoch_advance(region.chunks),
            build {}
        }
        region_kind Settlement { max_active = 1 }
        region_indices Settlement { Navgrid }
    ";
    let comp = round_trip(src_valid).expect("valid: Navgrid resolves to navgrid index");
    assert_eq!(comp.indices.len(), 1);
    assert_eq!(comp.region_kinds.len(), 1);

    let src_unknown = "
        region_kind Settlement { max_active = 1 }
        region_indices Settlement { Vismap }
    ";
    let result = round_trip(src_unknown);
    assert!(
        result.is_err(),
        "expected resolve error for unknown index kind in region_indices, got {:?}",
        result
    );
}

#[test]
fn index_rejects_non_voxel_region_param_type() {
    let src = "
        index bad(region: Agent) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: manual,
            build {}
        }
    ";
    let result = std::panic::catch_unwind(|| {
        let _ = dsl_compiler::parse(src);
    });
    assert!(
        result.is_err() || dsl_compiler::parse(src).is_err(),
        "expected parse error for non-VoxelRegion region param"
    );
}

#[test]
fn index_rejects_unknown_storage_shape() {
    let src = "
        index bad(region: VoxelRegion) -> Out {
            storage: octree_compressed(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: manual,
            build {}
        }
    ";
    assert!(
        dsl_compiler::parse(src).is_err(),
        "expected parse error for unknown storage shape"
    );
}

#[test]
fn index_rejects_unknown_cost_class() {
    let src = "
        index bad(region: VoxelRegion) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Trivial,
            rebuild_on: manual,
            build {}
        }
    ";
    assert!(
        dsl_compiler::parse(src).is_err(),
        "expected parse error for unknown cost class"
    );
}

#[test]
fn index_build_body_brace_scanner_handles_nested_braces_in_raw_text() {
    // The brace-scanner in parse_index_build_clause MUST handle
    // nested braces so the raw-text capture stays correct even
    // when the body grammar doesn't yet support block exprs. We
    // can't fully exercise it without block exprs in the grammar,
    // but engine calls with multiple parens-but-not-braces are a
    // good proxy — the scanner correctly closes on the outer `}`.
    let src = "
        index ok(region: VoxelRegion) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: manual,
            build {
                let x = engine::column_reduce_xz(region);
                let y = engine::per_cell_classify(x, AGENT_STEP_HEIGHT);
                engine::connect_neighbors(y, AGENT_STEP_HEIGHT)
            }
        }
    ";
    let comp = round_trip(src).expect("resolve");
    let body = &comp.indices[0].build_body;
    assert!(body.contains("engine::column_reduce_xz"));
    assert!(body.contains("engine::connect_neighbors"));
    assert_eq!(comp.indices[0].build_body_ast.stmts.len(), 3);
}

// =====================================================================
// Phase 2b: build body parse
// =====================================================================

#[test]
fn navgrid_build_body_from_spec_parses() {
    // Verbatim example from spec §7.2.
    let src = "
        index navgrid(region: VoxelRegion) -> Walkable {
            storage: per_cell_2d(max_cells = 16384, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: chunk_epoch_advance(region.chunks),
            build {
                let height = engine::column_reduce_xz(region);
                let walk = engine::per_cell_classify(height, classify_walkable);
                engine::connect_neighbors(walk, AGENT_STEP_HEIGHT)
            }
        }
    ";
    // The spec example references `classify_walkable` as an
    // identifier — for Phase 2b we don't know about it (not in
    // KNOWN_INDEX_BUILD_CONSTS, not a local). Test the *shape*
    // by replacing it with a known constant. See follow-up test
    // for the unknown-ident rejection.
    let src_clean = src.replace("classify_walkable", "AGENT_STEP_HEIGHT");
    let comp = round_trip(&src_clean).expect("resolve");
    let idx = &comp.indices[0];
    assert_eq!(idx.build_body_ast.stmts.len(), 3);
    // First two are `let`s; last is the return expression.
    use dsl_ast::ast::IndexBuildStmt;
    assert!(matches!(idx.build_body_ast.stmts[0], IndexBuildStmt::Let { .. }));
    assert!(matches!(idx.build_body_ast.stmts[1], IndexBuildStmt::Let { .. }));
    assert!(matches!(idx.build_body_ast.stmts[2], IndexBuildStmt::Return { .. }));
}

#[test]
fn build_body_rejects_unknown_engine_helper() {
    let src = "
        index bad(region: VoxelRegion) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: manual,
            build {
                engine::nonexistent_helper(region)
            }
        }
    ";
    let err = round_trip(src).expect_err("should fail on unknown helper");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("nonexistent_helper") || msg.contains("unknown engine helper"),
        "expected unknown-helper diagnostic; got: {msg}"
    );
}

#[test]
fn build_body_rejects_unknown_identifier() {
    let src = "
        index bad(region: VoxelRegion) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: manual,
            build {
                let x = engine::column_reduce_xz(some_undefined_ident);
                x
            }
        }
    ";
    let err = round_trip(src).expect_err("should fail on unknown ident");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("unknown identifier") || msg.contains("some_undefined_ident"),
        "expected unknown-ident diagnostic; got: {msg}"
    );
}

#[test]
fn build_body_let_binding_is_in_scope_for_subsequent_stmts() {
    let src = "
        index ok(region: VoxelRegion) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: manual,
            build {
                let x = engine::column_reduce_xz(region);
                let y = engine::per_cell_classify(x, AGENT_STEP_HEIGHT);
                y
            }
        }
    ";
    let comp = round_trip(src).expect("resolve");
    assert_eq!(comp.indices[0].build_body_ast.stmts.len(), 3);
}

#[test]
fn build_body_member_access_restricted_to_region_param() {
    let src = "
        index bad(region: VoxelRegion) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: manual,
            build {
                let x = engine::column_reduce_xz(region);
                x.chunks
            }
        }
    ";
    let err = round_trip(src).expect_err("should reject `.chunks` on a non-region binding");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("member access") || msg.contains(".chunks"),
        "expected member-access diagnostic; got: {msg}"
    );
}

#[test]
fn build_body_stmts_after_return_rejected() {
    let src = "
        index bad(region: VoxelRegion) -> Out {
            storage: per_cell_2d(max_cells = 100, bytes_per_cell = 4),
            cost_class: Cheap,
            rebuild_on: manual,
            build {
                AGENT_STEP_HEIGHT
                AGENT_STEP_HEIGHT
            }
        }
    ";
    // The body parser detects this as a parse error (parser ends
    // up at body_end with leftover unconsumed tokens). Either
    // parse or resolve failure is acceptable — both surface the
    // same defect to the .sim author.
    let parsed = dsl_compiler::parse(src);
    let parse_ok = parsed.is_ok();
    if parse_ok {
        let comp = dsl_ast::resolve::resolve(parsed.unwrap());
        assert!(
            comp.is_err(),
            "expected trailing-stmt to fail at parse OR resolve; both succeeded"
        );
    }
    // If parsing failed, that's also acceptable — the parser
    // surfaced the trailing-tokens defect.
}

// =====================================================================
// Phase 1 (pre-existing): region_kind / region_indices basics
// =====================================================================

#[test]
fn region_kind_supports_optional_trailing_comma() {
    // `max_active = N,` matches the config-field idiom.
    let src = "
        index surfacemesh(region: VoxelRegion) -> Out {
            storage: mesh_buffer(max_vertices = 100, max_indices = 100),
            cost_class: Cheap, rebuild_on: manual, build {}
        }
        region_kind WildernessTile { max_active = 4096, }
        region_indices WildernessTile { SurfaceMesh, }
    ";
    let comp = round_trip(src).expect("resolve");
    assert_eq!(comp.region_kinds[0].name, "WildernessTile");
    assert_eq!(comp.region_kinds[0].max_active, 4096);
    assert_eq!(
        comp.region_kinds[0].index_kind_names,
        vec!["SurfaceMesh"]
    );
}
