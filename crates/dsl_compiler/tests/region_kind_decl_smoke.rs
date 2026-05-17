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
    let src = "
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

#[test]
fn region_kind_supports_optional_trailing_comma() {
    // `max_active = N,` matches the config-field idiom.
    let src = "
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
