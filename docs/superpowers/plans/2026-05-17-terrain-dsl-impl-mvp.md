# Terrain DSL — MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the minimum end-to-end terrain DSL slice — a `.sim` file can declare `terrain { extent, cell_size, seed_purpose, materials, layer fill }`, the compiler emits a `generate_terrain(seed)` function, the runtime spawns it on a worker thread, installs the result into `SimState`, and ticks once. Layer primitives beyond `fill` (heightfield, box, prefab, carve_caves, region, walkable_mask) ship in a follow-up plan.

**Architecture:** New `terrain {}` block in `.sim`, parsed by `dsl_ast`, lowered + emitted by `dsl_compiler`. Per-runtime `build.rs` consumes the compiled artifact and writes `terrain_gen.rs` into `OUT_DIR`. Generation runs on a worker thread (`engine_voxel::TerrainGenHandle`) and feeds an `Arc<dyn TerrainQuery>` into the existing seam.

**Tech Stack:** Rust workspace; `dsl_ast` + `dsl_compiler` (`crates/dsl_compiler/src/cg/{lower,emit}`); `engine_voxel` (wraps external `voxel_engine` crate); `engine::rng::per_agent_u32_pcg` for keyed RNG.

**Source spec:** `docs/superpowers/specs/2026-05-17-terrain-dsl-spec-design.md`

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - `TerrainQuery` trait at `crates/engine/src/terrain.rs:45`
  - `SimState.terrain: Arc<dyn TerrainQuery + Send + Sync>` at `crates/engine/src/state/mod.rs:259`
  - `engine_voxel::VoxelTerrain` at `crates/engine_voxel/src/lib.rs:139` (cubic-extent only — non-cubic deferred)
  - `engine::rng::per_agent_u32_pcg(u32, u32, u32, u32)` at `crates/engine/src/rng.rs:135`
  - `dsl_compiler::cg::lower::driver` and `dsl_compiler::cg::emit::program` (existing lower/emit shape)
  - Search method: `rg`, direct `Read`.

- **Decision:** extend the DSL surface with a new `terrain {}` top-level block. New code is emitter output (the per-runtime `terrain_gen.rs`), satisfying P1. Hand-written code is library code in `engine_voxel` (`MaterialTable`, layer helpers, `TerrainGenHandle`) — explicitly out of the P1 rule-logic scope.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: `crates/dsl_ast/src/lib.rs`, `crates/dsl_ast/src/terrain.rs` (new), `crates/dsl_compiler/src/cg/lower/terrain.rs` (new), `crates/dsl_compiler/src/cg/emit/terrain.rs` (new), `crates/dsl_compiler/src/cg/lower/driver.rs` (wire-up).
  - Generated outputs re-emitted: `terrain_gen.rs` in opted-in runtime `OUT_DIR`. Existing rule emit unchanged.

- **Hand-written downstream code:**
  - `crates/engine_voxel/src/materials.rs` — `MaterialTable` struct (data; not rule logic).
  - `crates/engine_voxel/src/terrain_layers.rs` — `layer_fill` helper (shared library code called by emitter output).
  - `crates/engine_voxel/src/terrain_gen.rs` — `TerrainGenHandle` worker-thread harness.
  - `crates/terrain_probe_runtime/` — new runtime crate for the runtime-gate smoke test.
  Justification: layer helpers must contain hand-written math (noise / AABB iteration / etc.) that is shared across every generated terrain; emitting that math per-runtime would duplicate it 30×. The P1 "no hand-written rule logic" scope is rule-handler code (`crates/engine/src/handlers/`, `cascade/handlers/`, `generated/`), not engine-library data structures.

- **Constitution check:**
  - P1 (Compiler-First): PASS — terrain `generate_terrain` is emitted; layer helpers are library code, not rule handlers.
  - P2 (Schema-Hash on Layout): N/A — no new `SimState` SoA field; terrain lives on the existing `Arc<dyn TerrainQuery>` slot.
  - P3 (Cross-Backend Parity): PASS — annotated `@cpu_only`; the parser-recognised exception. GPU emitter is explicit future work.
  - P4 (`EffectOp` Size Budget): N/A — no new `EffectOp` variants.
  - P5 (Determinism via Keyed PCG): PASS — all entropy via `per_agent_u32_pcg(world_seed, layer_index, cell_index, seed_purpose)`. Ahash drift inherited; mitigation = within-process re-run tests, not byte goldens.
  - P6 (Events Are the Mutation Channel): N/A — terrain init runs pre-tick; per-tick mutation continues via existing `PlaceVoxel`/`Harvest`.
  - P7 (Replayability Flagged): N/A — no new event variants.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — each task ends with a `git commit`.
  - P10 (No Runtime Panic): PASS — generation is off the deterministic tick path; worker thread wraps gen in `catch_unwind`.
  - P11 (Reduction Determinism): N/A — no atomic-append / float reductions in generation.

- **Runtime gate:** Task 13 adds a smoke test at `crates/sims/tests/terrain_probe_smoke.rs` that boots, spawns terrain gen on a worker thread, awaits readiness, wraps the result in `Arc<dyn TerrainQuery + Send + Sync>`, and asserts the `TerrainQuery` surface (`height_at`, `walkable`, `line_of_sight`) reflects the declared materials. The plan does not exercise `SimState::step()` — `engine::step::step` is currently an `unimplemented!()` stub (Plan B1' Task 11 deletion) and tick-level terrain consumption ships with the layer expansion follow-up. The runtime gate here is "observable post-condition on the changed code path": the emitted `generate_terrain` produces a `VoxelTerrain` whose trait methods return the values the DSL declared.
  - `terrain_probe_smoke` at `crates/sims/tests/terrain_probe_smoke.rs` — "boot → async-gen → install in Arc → assert TerrainQuery surface".

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

---

## Files touched

- Create: `crates/dsl_ast/src/terrain.rs` — AST nodes.
- Modify: `crates/dsl_ast/src/lib.rs` — re-export terrain module.
- Modify: `crates/dsl_compiler/src/cg/lower/mod.rs` (or its module root) — register `pub mod terrain`.
- Create: `crates/dsl_compiler/src/cg/lower/terrain.rs` — lower + validate.
- Modify: `crates/dsl_compiler/src/cg/emit/mod.rs` (or its module root) — register `pub mod terrain` + re-export `emit_terrain`.
- Create: `crates/dsl_compiler/src/cg/emit/terrain.rs` — emit Rust source.
- Modify: `crates/dsl_compiler/src/build_helper.rs` — `emit_namespaced` also writes `terrain_gen.rs` to `OUT_DIR/<fixture>/` when the program has a `terrain` block.
- Modify: parser entry point (locate in Task 2) to recognise `terrain { ... }` as a top-level block and attach it to `Program::terrain: Option<TerrainBlock>`.
- Create: `crates/engine_voxel/src/materials.rs` — `MaterialTable`.
- Create: `crates/engine_voxel/src/terrain_layers.rs` — `layer_fill`.
- Create: `crates/engine_voxel/src/terrain_gen.rs` — `TerrainGenHandle`.
- Modify: `crates/engine_voxel/src/lib.rs` — export new modules + extend `VoxelTerrain` with `materials()` accessor.
- Create: `assets/sim/terrain_probe.sim` — minimum-viable fixture exercising the new block.
- Modify: `crates/sims/build.rs` — add `terrain_probe` to the allow-list AND extend the `pub mod <fixture>` stub generator to conditionally include `terrain_gen.rs` (only for fixtures whose OUT_DIR has that file).
- Create: `crates/sims/tests/terrain_probe_smoke.rs` — runtime-gate smoke test.

**Constraints baked in:**

- `extent` is a single `u32` in v1 (cubic). Non-cubic extent requires extending `engine_voxel::VoxelTerrain` and is deferred.
- Only the `fill` layer ships in this plan. Heightfield/box/sphere/cylinder/carve_caves/prefab/region/walkable_mask are follow-up.
- The DSL example in the spec used `(128, 128, 64)`. v1 enforces cubic — the parser accepts a single `u32` literal. Follow-up plan can relax to a tuple after `VoxelTerrain` grows non-cubic support.
- No new `*_runtime` crate. Fixtures go in `assets/sim/` and surface as `sims::<fixture>::*` (the megacrate model; see `crates/sims/build.rs`).
- The runtime gate does NOT call `SimState::step` — that entry point is `unimplemented!()` per Plan B1' Task 11. The gate exercises the `TerrainQuery` surface on the installed `Arc`, which is the actual changed code path.

---

## Task 1: AST node for `terrain { extent, cell_size, seed_purpose }`

**Files:**
- Create: `crates/dsl_ast/src/terrain.rs`
- Modify: `crates/dsl_ast/src/lib.rs`
- Test: `crates/dsl_ast/tests/terrain_node.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_ast/tests/terrain_node.rs
use dsl_ast::terrain::{TerrainBlock, MaterialDecl, LayerDecl, LayerKind};

#[test]
fn terrain_block_construct_and_field_access() {
    let block = TerrainBlock {
        extent: 128,
        cell_size: 0.25,
        seed_purpose: 0xBA5E_7E55,
        materials: vec![],
        layers: vec![],
    };
    assert_eq!(block.extent, 128);
    assert!((block.cell_size - 0.25).abs() < 1e-6);
    assert_eq!(block.seed_purpose, 0xBA5E_7E55);
    assert!(block.materials.is_empty());
    assert!(block.layers.is_empty());
}

#[test]
fn material_decl_defaults() {
    let m = MaterialDecl {
        name: "grass".into(),
        id: 1,
        walkable: true,
        hardness: 1,
        biome_tag: None,
        color: 0x4A8B3A,
        movement_cost: 1.0,
    };
    assert_eq!(m.id, 1);
    assert_eq!(m.color, 0x4A8B3A);
}

#[test]
fn layer_fill_decl() {
    let l = LayerDecl {
        index: 1,
        kind: LayerKind::Fill { material: "grass".into() },
    };
    assert_eq!(l.index, 1);
    matches!(l.kind, LayerKind::Fill { .. });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_ast --test terrain_node`
Expected: FAIL with "could not find `terrain` in the crate root" (or similar — module does not exist).

- [ ] **Step 3: Write minimal implementation**

```rust
// crates/dsl_ast/src/terrain.rs
//! AST for the `terrain { ... }` block. See
//! `docs/superpowers/specs/2026-05-17-terrain-dsl-spec-design.md`.

#[derive(Debug, Clone, PartialEq)]
pub struct TerrainBlock {
    pub extent: u32,
    pub cell_size: f32,
    pub seed_purpose: u32,
    pub materials: Vec<MaterialDecl>,
    pub layers: Vec<LayerDecl>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaterialDecl {
    pub name: String,
    pub id: u8,
    pub walkable: bool,
    pub hardness: u8,
    pub biome_tag: Option<String>,
    pub color: u32,
    pub movement_cost: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerDecl {
    pub index: u32,
    pub kind: LayerKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    Fill { material: String },
}
```

Modify `crates/dsl_ast/src/lib.rs` to add `pub mod terrain;` near the other module declarations.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_ast --test terrain_node`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_ast/src/terrain.rs crates/dsl_ast/src/lib.rs crates/dsl_ast/tests/terrain_node.rs
git commit -m "feat(dsl_ast): terrain block AST (extent, materials, fill layer)"
```

---

## Task 2: Parser for `terrain { extent, cell_size, seed_purpose }`

**Files:**
- Modify: `crates/dsl_compiler/src/parse/grammar.rs` (or the file where top-level blocks are parsed — locate via `rg "pub fn parse_program" crates/dsl_compiler/src/`).
- Test: `crates/dsl_compiler/tests/terrain_parse_basic.rs`

- [ ] **Step 1: Locate the existing parser entry point**

Run: `rg -n "pub fn parse_program|fn parse_top_level|pub fn parse\(" crates/dsl_compiler/src/`
Identify which file dispatches to per-block parsers (`agent`, `rule`, `event`). Add a `terrain` arm there.

- [ ] **Step 2: Write the failing test**

```rust
// crates/dsl_compiler/tests/terrain_parse_basic.rs
use dsl_compiler::parse;   // re-exported from dsl_ast::parse(source: &str) -> Result<Program, ParseError>

#[test]
fn parses_minimum_terrain_block() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 0.5
  seed_purpose: 0xBA5E_7E55
}
"#;
    let program = parse(src).expect("parse");
    let t = program.terrain.expect("terrain block");
    assert_eq!(t.extent, 64);
    assert!((t.cell_size - 0.5).abs() < 1e-6);
    assert_eq!(t.seed_purpose, 0xBA5E_7E55);
}

#[test]
fn rejects_zero_seed_purpose() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 0.5
  seed_purpose: 0
}
"#;
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains("seed_purpose") && msg.contains("non-zero"),
            "expected seed_purpose non-zero error, got: {msg}");
}
```

The `Program` struct (defined in `crates/dsl_ast/src/ast.rs` — locate the struct definition via `rg -n "pub struct Program" crates/dsl_ast/src/`) must grow a `pub terrain: Option<TerrainBlock>` field. Update `Program::default()` / any existing constructors to initialise it to `None`.

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test terrain_parse_basic`
Expected: FAIL — parser does not recognise `terrain`.

- [ ] **Step 4: Implement the parser arm**

Add a `terrain` keyword arm to the top-level block dispatch. Use the existing per-block parser convention (`fn parse_terrain(tokens: &mut TokenStream) -> Result<TerrainBlock, ParseError>`). Required fields:

- `extent: <u32>`
- `cell_size: <f32>`
- `seed_purpose: <u32-hex-or-decimal>`

Validation in the parser (or immediately after): `seed_purpose != 0`. Surface as `ParseError::with_msg("`seed_purpose` must be non-zero")` or the project's existing error helper. Add `pub terrain: Option<TerrainBlock>` to the `Program` struct.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test terrain_parse_basic`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add crates/dsl_compiler/src crates/dsl_compiler/tests/terrain_parse_basic.rs
git commit -m "feat(dsl): parse terrain { extent, cell_size, seed_purpose }"
```

---

## Task 3: Parse the `materials { ... }` sub-block

**Files:**
- Modify: `crates/dsl_compiler/src/parse/grammar.rs` (terrain parser added in Task 2).
- Test: `crates/dsl_compiler/tests/terrain_parse_materials.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/terrain_parse_materials.rs
use dsl_compiler::parse;

#[test]
fn parses_materials_block_with_one_entry() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A }
  }
}
"#;
    let t = parse(src).unwrap().terrain.unwrap();
    assert_eq!(t.materials.len(), 1);
    let g = &t.materials[0];
    assert_eq!(g.name, "grass");
    assert_eq!(g.id, 1);
    assert!(g.walkable);
    assert_eq!(g.hardness, 1);
    assert_eq!(g.color, 0x4A8B3A);
    assert!((g.movement_cost - 1.0).abs() < 1e-6); // default
    assert!(g.biome_tag.is_none());                 // default
}

#[test]
fn rejects_duplicate_material_id() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    a { id: 1, walkable: true,  hardness: 1, color: 0xFFFFFF }
    b { id: 1, walkable: false, hardness: 2, color: 0x000000 }
  }
}
"#;
    let err = parse(src).err().unwrap();
    assert!(format!("{err}").contains("duplicate material id"), "got: {err}");
}

#[test]
fn rejects_material_id_zero() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    void { id: 0, walkable: true, hardness: 0, color: 0 }
  }
}
"#;
    let err = parse(src).err().unwrap();
    assert!(format!("{err}").contains("material id must be 1..=255"), "got: {err}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test terrain_parse_materials`
Expected: FAIL — materials parsing not implemented.

- [ ] **Step 3: Implement the materials sub-block parser**

Inside the terrain parser added in Task 2, parse an optional `materials { ... }` block. For each entry parse `<ident> { id: <u8>, walkable: <bool>, hardness: <u8>, biome_tag: <ident>?, color: <u32-hex>, movement_cost: <f32>? }`. Apply defaults (`movement_cost = 1.0`, `biome_tag = None`). After all materials parse, validate: every id in `1..=255` (reject 0), no duplicate ids.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test terrain_parse_materials`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src crates/dsl_compiler/tests/terrain_parse_materials.rs
git commit -m "feat(dsl): parse terrain materials block + uniqueness validation"
```

---

## Task 4: Parse `layer fill { material: <name> }`

**Files:**
- Modify: `crates/dsl_compiler/src/parse/grammar.rs`.
- Test: `crates/dsl_compiler/tests/terrain_parse_layers.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/terrain_parse_layers.rs
use dsl_compiler::parse;
use dsl_ast::terrain::LayerKind;

#[test]
fn parses_single_fill_layer() {
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: grass }
}
"#;
    let t = parse(src).unwrap().terrain.unwrap();
    assert_eq!(t.layers.len(), 1);
    assert_eq!(t.layers[0].index, 1);
    match &t.layers[0].kind {
        LayerKind::Fill { material } => assert_eq!(material, "grass"),
    }
}

#[test]
fn layer_indices_are_source_order_one_based() {
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    a { id: 1, walkable: true, hardness: 1, color: 0xFF0000 }
    b { id: 2, walkable: true, hardness: 1, color: 0x00FF00 }
  }
  layer fill { material: a }
  layer fill { material: b }
}
"#;
    let t = parse(src).unwrap().terrain.unwrap();
    assert_eq!(t.layers.len(), 2);
    assert_eq!(t.layers[0].index, 1);
    assert_eq!(t.layers[1].index, 2);
}

#[test]
fn rejects_unknown_material_in_fill_layer() {
    // Material name resolution is a lowering concern, not parsing.
    // This test asserts the *parser* still produces an AST with the
    // unresolved name — the failure surfaces in Task 5 (lowering).
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: ghost }
}
"#;
    let t = parse(src).unwrap().terrain.unwrap();
    match &t.layers[0].kind {
        LayerKind::Fill { material } => assert_eq!(material, "ghost"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test terrain_parse_layers`
Expected: FAIL — layer parsing not implemented.

- [ ] **Step 3: Implement layer parsing**

Parse one or more `layer <kind> { ... }` clauses inside the terrain block. v1 supports only `kind = fill` (`material: <ident>`). Assign `index = 1 + (count_already_seen)`. Any unrecognised `<kind>` → `ParseError::with_msg("unknown layer kind: <name>")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test terrain_parse_layers`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src crates/dsl_compiler/tests/terrain_parse_layers.rs
git commit -m "feat(dsl): parse `layer fill { material }` with source-order indexing"
```

---

## Task 5: Lowering — resolve material names + validate layer refs

**Files:**
- Create: `crates/dsl_compiler/src/cg/lower/terrain.rs`
- Modify: `crates/dsl_compiler/src/cg/lower/driver.rs` (call into terrain lowering).
- Test: `crates/dsl_compiler/tests/terrain_lower.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/terrain_lower.rs
use dsl_compiler::{parse, lower::lower_terrain, LowerError};

#[test]
fn lower_resolves_fill_layer_material_to_id() {
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    grass { id: 7, walkable: true, hardness: 1, color: 0x4A8B3A }
  }
  layer fill { material: grass }
}
"#;
    let program = parse(src).unwrap();
    let lowered = lower_terrain(&program.terrain.unwrap()).unwrap();
    assert_eq!(lowered.layers.len(), 1);
    match &lowered.layers[0] {
        dsl_compiler::lower::TerrainLayerIr::Fill { material_id } => {
            assert_eq!(*material_id, 7);
        }
    }
}

#[test]
fn lower_rejects_unknown_material_ref() {
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: ghost }
}
"#;
    let program = parse(src).unwrap();
    let err = lower_terrain(&program.terrain.unwrap()).err().unwrap();
    assert!(matches!(err, LowerError::UnknownMaterial(ref name) if name == "ghost"), "got: {err:?}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test terrain_lower`
Expected: FAIL — `lower_terrain` does not exist.

- [ ] **Step 3: Implement lowering**

```rust
// crates/dsl_compiler/src/cg/lower/terrain.rs
use dsl_ast::terrain::{TerrainBlock, LayerKind};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TerrainIr {
    pub extent: u32,
    pub cell_size: f32,
    pub seed_purpose: u32,
    pub materials: Vec<MaterialIr>,
    pub layers: Vec<TerrainLayerIr>,
}

#[derive(Debug, Clone)]
pub struct MaterialIr {
    pub id: u8,
    pub walkable: bool,
    pub hardness: u8,
    pub biome_tag_hash: u32,   // 0 = none; FNV-1a of biome_tag string otherwise
    pub color: u32,
    pub movement_cost: f32,
}

#[derive(Debug, Clone)]
pub enum TerrainLayerIr {
    Fill { material_id: u8 },
}

#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("unknown material: {0}")]
    UnknownMaterial(String),
}

pub fn lower_terrain(t: &TerrainBlock) -> Result<TerrainIr, LowerError> {
    let by_name: HashMap<&str, u8> = t.materials.iter().map(|m| (m.name.as_str(), m.id)).collect();

    let materials = t.materials.iter().map(|m| MaterialIr {
        id: m.id,
        walkable: m.walkable,
        hardness: m.hardness,
        biome_tag_hash: m.biome_tag.as_deref().map(fnv1a_u32).unwrap_or(0),
        color: m.color,
        movement_cost: m.movement_cost,
    }).collect();

    let mut layers = Vec::with_capacity(t.layers.len());
    for l in &t.layers {
        match &l.kind {
            LayerKind::Fill { material } => {
                let mid = by_name.get(material.as_str()).copied()
                    .ok_or_else(|| LowerError::UnknownMaterial(material.clone()))?;
                layers.push(TerrainLayerIr::Fill { material_id: mid });
            }
        }
    }

    Ok(TerrainIr {
        extent: t.extent,
        cell_size: t.cell_size,
        seed_purpose: t.seed_purpose,
        materials,
        layers,
    })
}

fn fnv1a_u32(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}
```

Wire `pub mod terrain;` into `crates/dsl_compiler/src/cg/lower/mod.rs` (or the lower-module root) and re-export `lower_terrain`, `TerrainIr`, `TerrainLayerIr`, `LowerError`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test terrain_lower`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src crates/dsl_compiler/tests/terrain_lower.rs
git commit -m "feat(dsl): lower terrain block — resolve material names, build TerrainIr"
```

---

## Task 6: `engine_voxel::MaterialTable`

**Files:**
- Create: `crates/engine_voxel/src/materials.rs`
- Modify: `crates/engine_voxel/src/lib.rs` (re-export + `VoxelTerrain::materials()`).
- Test: `crates/engine_voxel/tests/materials.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/engine_voxel/tests/materials.rs
use engine_voxel::{MaterialTable, MaterialRow};

#[test]
fn lookup_by_id() {
    static ROWS: [MaterialRow; 3] = [
        MaterialRow { id: 1, walkable: true,  hardness: 1, biome_tag_hash: 0, color: 0x4A8B3A, movement_cost: 1.0 },
        MaterialRow { id: 2, walkable: false, hardness: 8, biome_tag_hash: 0, color: 0x808080, movement_cost: 1.0 },
        MaterialRow { id: 3, walkable: true,  hardness: 2, biome_tag_hash: 0, color: 0xD9C28A, movement_cost: 1.5 },
    ];
    let t = MaterialTable::new(&ROWS);
    assert_eq!(t.get(1).unwrap().color, 0x4A8B3A);
    assert_eq!(t.get(2).unwrap().walkable, false);
    assert!((t.get(3).unwrap().movement_cost - 1.5).abs() < 1e-6);
    assert!(t.get(0).is_none());     // air sentinel
    assert!(t.get(99).is_none());    // unknown id
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p engine_voxel --test materials`
Expected: FAIL — `MaterialTable` does not exist.

- [ ] **Step 3: Implement `MaterialTable`**

```rust
// crates/engine_voxel/src/materials.rs
//! Const-friendly per-material property table. Stored alongside the
//! `VoxelGrid` inside `VoxelTerrain`; rows are produced by the DSL
//! emitter. See spec section "Material properties".

#[derive(Copy, Clone, Debug)]
pub struct MaterialRow {
    pub id: u8,
    pub walkable: bool,
    pub hardness: u8,
    pub biome_tag_hash: u32,
    pub color: u32,
    pub movement_cost: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct MaterialTable {
    rows: &'static [MaterialRow],
}

impl MaterialTable {
    pub const fn new(rows: &'static [MaterialRow]) -> Self {
        Self { rows }
    }

    pub fn get(&self, id: u8) -> Option<&MaterialRow> {
        if id == 0 { return None; }
        self.rows.iter().find(|r| r.id == id)
    }

    pub fn rows(&self) -> &'static [MaterialRow] { self.rows }
}

/// Empty table used by `VoxelTerrain::new()` / `with_extent()` callers
/// that have not gone through the DSL terrain pipeline.
pub const EMPTY: MaterialTable = MaterialTable::new(&[]);
```

In `crates/engine_voxel/src/lib.rs`:

- Add `pub mod materials;` and `pub use materials::{MaterialRow, MaterialTable};`.
- Add a `materials: MaterialTable` field to `VoxelTerrain`, defaulting to `materials::EMPTY` in `new()` / `with_extent()`.
- Add `pub fn materials(&self) -> MaterialTable { self.materials }`.
- Add a new constructor `pub fn with_extent_and_materials(extent: u32, materials: MaterialTable) -> Self`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p engine_voxel --test materials`
Expected: 1 passed. Also run `cargo test -p engine_voxel` to make sure existing tests still pass (the new field has a default so old call sites should compile).

- [ ] **Step 5: Commit**

```bash
git add crates/engine_voxel/src/materials.rs crates/engine_voxel/src/lib.rs crates/engine_voxel/tests/materials.rs
git commit -m "feat(engine_voxel): MaterialTable + VoxelTerrain materials accessor"
```

---

## Task 7: `layer_fill` helper in `engine_voxel::terrain_layers`

**Files:**
- Create: `crates/engine_voxel/src/terrain_layers.rs`
- Modify: `crates/engine_voxel/src/lib.rs`
- Test: `crates/engine_voxel/tests/terrain_layers_fill.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/engine_voxel/tests/terrain_layers_fill.rs
use engine_voxel::{VoxelTerrain, terrain_layers::layer_fill};

#[test]
fn layer_fill_writes_every_cell() {
    let mut terrain = VoxelTerrain::with_extent(4);
    layer_fill(&mut terrain, /*material_id=*/ 2);
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                assert_eq!(terrain.cell_at(x, y, z), 2, "cell ({x},{y},{z})");
            }
        }
    }
}

#[test]
fn layer_fill_with_air_id_zero_clears() {
    let mut terrain = VoxelTerrain::with_extent(2);
    layer_fill(&mut terrain, 5);
    layer_fill(&mut terrain, 0);
    assert_eq!(terrain.cell_at(0, 0, 0), 0);
    assert_eq!(terrain.cell_at(1, 1, 1), 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p engine_voxel --test terrain_layers_fill`
Expected: FAIL — `terrain_layers` module does not exist.

- [ ] **Step 3: Implement `layer_fill`**

```rust
// crates/engine_voxel/src/terrain_layers.rs
//! Hand-written layer helpers called by emitter-generated
//! `generate_terrain(seed)` functions. Library code (not rule
//! logic) per spec — see AIS justification.

use crate::VoxelTerrain;

/// Uniform-material fill across the cubic extent. Material id `0`
/// clears cells to air. Determinism: deterministic by construction —
/// no RNG consumed.
pub fn layer_fill(terrain: &mut VoxelTerrain, material_id: u8) {
    let n = terrain.extent();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                terrain.set_cell(x, y, z, material_id);
            }
        }
    }
}
```

In `crates/engine_voxel/src/lib.rs`: add `pub mod terrain_layers;`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p engine_voxel --test terrain_layers_fill`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_voxel/src/terrain_layers.rs crates/engine_voxel/src/lib.rs crates/engine_voxel/tests/terrain_layers_fill.rs
git commit -m "feat(engine_voxel): layer_fill helper writes every cell"
```

---

## Task 8: Emitter for `terrain {}` — `generate_terrain(seed)`

**Files:**
- Create: `crates/dsl_compiler/src/cg/emit/terrain.rs`
- Modify: `crates/dsl_compiler/src/cg/emit/program.rs` (call into terrain emit).
- Test: `crates/dsl_compiler/tests/terrain_emit.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/terrain_emit.rs
use dsl_compiler::{parse, lower::lower_terrain, emit::emit_terrain};

#[test]
fn emit_contains_extent_materials_and_fill_call() {
    let src = r#"
terrain {
  extent: 16
  cell_size: 0.5
  seed_purpose: 0xBA5E_7E55
  materials {
    grass { id: 1, walkable: true,  hardness: 1, color: 0x4A8B3A }
    stone { id: 2, walkable: false, hardness: 8, color: 0x808080 }
  }
  layer fill { material: stone }
}
"#;
    let ir = lower_terrain(&parse(src).unwrap().terrain.unwrap()).unwrap();
    let rust_src = emit_terrain(&ir);

    // Smoke checks on emitted text. Goldens are *not* committed
    // (ahash drift caveat); these substring assertions are stable.
    assert!(rust_src.contains("pub const EXTENT: u32 = 16"), "extent missing");
    assert!(rust_src.contains("pub const CELL_SIZE: f32 = 0.5"), "cell_size missing");
    assert!(rust_src.contains("pub const SEED_PURPOSE: u32 = 0xBA5E_7E55") ||
            rust_src.contains("pub const SEED_PURPOSE: u32 = 3126641749"),
            "seed_purpose missing");
    assert!(rust_src.contains("pub static MATERIALS: ::engine_voxel::MaterialTable"),
            "MATERIALS static missing");
    assert!(rust_src.contains("id: 1u8") && rust_src.contains("id: 2u8"),
            "material ids missing");
    assert!(rust_src.contains("pub fn generate_terrain(world_seed: u64)"),
            "generate_terrain signature missing");
    assert!(rust_src.contains("::engine_voxel::terrain_layers::layer_fill"),
            "layer_fill call missing");
    assert!(rust_src.contains("2u8"), "fill material id 2 missing in call");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test terrain_emit`
Expected: FAIL — `emit_terrain` does not exist.

- [ ] **Step 3: Implement the emitter**

```rust
// crates/dsl_compiler/src/cg/emit/terrain.rs
//! Emit `terrain_gen.rs` source from a lowered `TerrainIr`. The output
//! is written by per-runtime `build.rs` into `OUT_DIR/terrain_gen.rs`
//! and `include!`-d by the runtime. The compiler itself never writes
//! to OUT_DIR — that is the build.rs caller's job.

use crate::lower::{TerrainIr, TerrainLayerIr};
use std::fmt::Write;

pub fn emit_terrain(ir: &TerrainIr) -> String {
    let mut out = String::new();
    writeln!(out, "// @generated by dsl_compiler::emit::terrain — DO NOT EDIT").unwrap();
    writeln!(out, "//! Generated terrain initialiser. Source: a `.sim` `terrain {{ }}` block.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub const EXTENT: u32 = {};", ir.extent).unwrap();
    writeln!(out, "pub const CELL_SIZE: f32 = {:?};", ir.cell_size).unwrap();
    writeln!(out, "pub const SEED_PURPOSE: u32 = 0x{:08X};", ir.seed_purpose).unwrap();
    writeln!(out).unwrap();

    // Material rows.
    writeln!(out, "static MATERIAL_ROWS: &[::engine_voxel::MaterialRow] = &[").unwrap();
    for m in &ir.materials {
        writeln!(out,
            "    ::engine_voxel::MaterialRow {{ id: {}u8, walkable: {}, hardness: {}u8, biome_tag_hash: 0x{:08X}u32, color: 0x{:06X}u32, movement_cost: {:?} }},",
            m.id, m.walkable, m.hardness, m.biome_tag_hash, m.color, m.movement_cost
        ).unwrap();
    }
    writeln!(out, "];").unwrap();
    writeln!(out, "pub static MATERIALS: ::engine_voxel::MaterialTable =").unwrap();
    writeln!(out, "    ::engine_voxel::MaterialTable::new(MATERIAL_ROWS);").unwrap();
    writeln!(out).unwrap();

    // generate_terrain.
    writeln!(out, "pub fn generate_terrain(world_seed: u64) -> ::engine_voxel::VoxelTerrain {{").unwrap();
    writeln!(out, "    let _ = world_seed; // unused until noise layers land").unwrap();
    writeln!(out, "    let mut t = ::engine_voxel::VoxelTerrain::with_extent_and_materials(EXTENT, MATERIALS);").unwrap();
    for (i, layer) in ir.layers.iter().enumerate() {
        let idx = (i + 1) as u32;
        match layer {
            TerrainLayerIr::Fill { material_id } => {
                writeln!(out, "    // layer {}: fill", idx).unwrap();
                writeln!(out, "    ::engine_voxel::terrain_layers::layer_fill(&mut t, {}u8);", material_id).unwrap();
            }
        }
    }
    writeln!(out, "    t").unwrap();
    writeln!(out, "}}").unwrap();

    out
}
```

Wire `pub mod terrain;` + `pub use terrain::emit_terrain;` into `crates/dsl_compiler/src/cg/emit/mod.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test terrain_emit`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src crates/dsl_compiler/tests/terrain_emit.rs
git commit -m "feat(dsl): emit generate_terrain(seed) with MaterialTable + layer_fill calls"
```

---

## Task 9: Compile-roundtrip check — emitted source actually compiles

**Files:**
- Test: `crates/dsl_compiler/tests/terrain_emit_compiles.rs`

This is a runtime-gate dry-run — proves the emitter output is well-formed Rust before any runtime adopts it.

- [ ] **Step 1: Write the test**

```rust
// crates/dsl_compiler/tests/terrain_emit_compiles.rs
//! Writes the emitted source to a temp file and invokes `rustc --edition 2021
//! --crate-type lib` to verify it compiles cleanly against `engine_voxel`.
//! Catches syntactic + type errors in emitter output without needing a
//! real runtime to opt in.

use dsl_compiler::{parse, lower::lower_terrain, emit::emit_terrain};
use std::process::Command;

#[test]
fn emitted_terrain_module_compiles() {
    let src = r#"
terrain {
  extent: 8
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A }
  }
  layer fill { material: grass }
}
"#;
    let ir = lower_terrain(&parse(src).unwrap().terrain.unwrap()).unwrap();
    let emitted = emit_terrain(&ir);

    let tmp = tempfile::tempdir().unwrap();
    let src_path = tmp.path().join("terrain_gen.rs");
    std::fs::write(&src_path, &emitted).unwrap();

    // Re-use the workspace's already-built engine_voxel rlib by
    // setting --extern via cargo metadata.
    let metadata = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version=1", "--no-deps"])
        .output().unwrap();
    assert!(metadata.status.success(), "cargo metadata failed");
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| {
            // Fallback: workspace root / target / debug / deps
            let manifest = env!("CARGO_MANIFEST_DIR");
            format!("{}/../../target/debug/deps", manifest)
        });

    let out_path = tmp.path().join("libterrain_gen.rlib");
    let status = Command::new("rustc")
        .args([
            "--edition", "2021",
            "--crate-type", "lib",
            "-L", &target_dir,
            "--extern", "engine_voxel",
            src_path.to_str().unwrap(),
            "-o", out_path.to_str().unwrap(),
        ])
        .status().unwrap();
    assert!(status.success(), "emitted terrain_gen.rs failed to compile");
}
```

Add `tempfile = "3"` to `crates/dsl_compiler/Cargo.toml` `[dev-dependencies]` if absent.

- [ ] **Step 2: Run the test**

Run: `cargo test -p dsl_compiler --test terrain_emit_compiles`
Expected: PASS if Task 8 emitter is sound; FAIL with a rustc diagnostic if not.

- [ ] **Step 3: Fix any emitter issues surfaced**

If rustc complains, edit `crates/dsl_compiler/src/cg/emit/terrain.rs` until the test passes. Most likely culprits: missing `as u32` casts on the `biome_tag_hash` literal, `MaterialTable::new` not being `const` (it should be from Task 6), trailing commas.

- [ ] **Step 4: Commit**

```bash
git add crates/dsl_compiler/tests/terrain_emit_compiles.rs crates/dsl_compiler/Cargo.toml
git commit -m "test(dsl): emitted terrain_gen.rs compiles against engine_voxel"
```

---

## Task 10: `TerrainGenHandle` worker-thread harness

**Files:**
- Create: `crates/engine_voxel/src/terrain_gen.rs`
- Modify: `crates/engine_voxel/src/lib.rs`
- Test: `crates/engine_voxel/tests/terrain_gen_handle.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/engine_voxel/tests/terrain_gen_handle.rs
use engine_voxel::{TerrainGenHandle, TerrainGenError, VoxelTerrain};

fn gen_simple(_seed: u64) -> VoxelTerrain {
    let mut t = VoxelTerrain::with_extent(4);
    for x in 0..4 { for y in 0..4 { for z in 0..4 { t.set_cell(x, y, z, 1); } } }
    t
}

fn gen_panics(_seed: u64) -> VoxelTerrain {
    panic!("intentional panic for catch_unwind test")
}

#[test]
fn block_until_ready_returns_terrain() {
    let handle = TerrainGenHandle::spawn(42, gen_simple);
    let terrain = handle.block_until_ready().expect("ok");
    assert_eq!(terrain.cell_at(0, 0, 0), 1);
    assert_eq!(terrain.cell_at(3, 3, 3), 1);
}

#[test]
fn try_take_returns_none_then_some() {
    let mut handle = TerrainGenHandle::spawn(7, gen_simple);
    // Spin until the worker reports done. Bounded ~1s to avoid hang.
    let mut got = None;
    for _ in 0..1000 {
        if let Some(r) = handle.try_take() { got = Some(r); break; }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let terrain = got.expect("worker completed within bound").expect("ok");
    assert_eq!(terrain.cell_at(0, 0, 0), 1);
}

#[test]
fn panic_in_gen_surfaces_as_error_not_abort() {
    let handle = TerrainGenHandle::spawn(0, gen_panics);
    let err = handle.block_until_ready().expect_err("gen panicked");
    match err {
        TerrainGenError::Panicked(msg) => {
            assert!(msg.contains("intentional panic"), "got: {msg}");
        }
    }
}

#[test]
fn two_runs_same_seed_byte_identical_for_deterministic_gen() {
    let a = TerrainGenHandle::spawn(123, gen_simple).block_until_ready().unwrap();
    let b = TerrainGenHandle::spawn(123, gen_simple).block_until_ready().unwrap();
    for x in 0..4 { for y in 0..4 { for z in 0..4 {
        assert_eq!(a.cell_at(x, y, z), b.cell_at(x, y, z));
    }}}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p engine_voxel --test terrain_gen_handle`
Expected: FAIL — `TerrainGenHandle` does not exist.

- [ ] **Step 3: Implement `TerrainGenHandle`**

```rust
// crates/engine_voxel/src/terrain_gen.rs
//! Worker-thread harness for terrain generation. Generator is a `fn`
//! pointer (not a closure) — keeps the Send-bound story simple.
//! Panics are caught and surfaced as `TerrainGenError::Panicked` so
//! the app does not abort. See spec section "Async execution model".

use crate::VoxelTerrain;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;

#[derive(Debug)]
pub enum TerrainGenError {
    Panicked(String),
}

impl std::fmt::Display for TerrainGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerrainGenError::Panicked(msg) => write!(f, "terrain generation panicked: {msg}"),
        }
    }
}

impl std::error::Error for TerrainGenError {}

pub struct TerrainGenHandle {
    rx: mpsc::Receiver<Result<VoxelTerrain, TerrainGenError>>,
    join: Option<std::thread::JoinHandle<()>>,
    pub seed: u64,
}

impl TerrainGenHandle {
    pub fn spawn(seed: u64, gen_fn: fn(u64) -> VoxelTerrain) -> Self {
        let (tx, rx) = mpsc::channel();
        let join = std::thread::Builder::new()
            .name(format!("terrain-gen-{seed}"))
            .spawn(move || {
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| gen_fn(seed)))
                    .map_err(|payload| {
                        let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                            (*s).to_string()
                        } else if let Some(s) = payload.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "<unknown panic payload>".to_string()
                        };
                        TerrainGenError::Panicked(msg)
                    });
                let _ = tx.send(result);
            })
            .expect("spawn terrain-gen worker thread");
        Self { rx, join: Some(join), seed }
    }

    /// Non-blocking poll. Returns `None` while the worker is still
    /// running; `Some(Ok)` on completion; `Some(Err)` on panic.
    pub fn try_take(&mut self) -> Option<Result<VoxelTerrain, TerrainGenError>> {
        match self.rx.try_recv() {
            Ok(r) => {
                if let Some(j) = self.join.take() { let _ = j.join(); }
                Some(r)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(
                TerrainGenError::Panicked("worker disconnected without result".into())
            )),
        }
    }

    /// Block until ready. Used by tests + headless tools. Production
    /// app loops should poll `try_take` so the UI thread stays
    /// responsive.
    pub fn block_until_ready(self) -> Result<VoxelTerrain, TerrainGenError> {
        let result = self.rx.recv().unwrap_or(Err(TerrainGenError::Panicked(
            "worker disconnected without result".into()
        )));
        if let Some(j) = self.join.into_iter().next() { let _ = j.join(); }
        result
    }
}
```

Add `pub mod terrain_gen;` and `pub use terrain_gen::{TerrainGenHandle, TerrainGenError};` to `crates/engine_voxel/src/lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p engine_voxel --test terrain_gen_handle`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_voxel/src/terrain_gen.rs crates/engine_voxel/src/lib.rs crates/engine_voxel/tests/terrain_gen_handle.rs
git commit -m "feat(engine_voxel): TerrainGenHandle — worker-thread gen with panic containment"
```

---

## Task 11: Wire `emit_terrain` into `build_helper::emit_namespaced`

**Files:**
- Modify: `crates/dsl_compiler/src/build_helper.rs` (function `emit_namespaced_with_strategy` is where per-fixture artifacts are written into `OUT_DIR/<fixture>/`).

- [ ] **Step 1: Read the existing emit_namespaced flow**

Run: `rg -n "fn emit_namespaced_with_strategy" crates/dsl_compiler/src/build_helper.rs` and read the function. Confirm where `generated.rs` and `runtime_core.rs` get written (around lines ~505 and ~647 per existing code). The new write goes alongside them.

- [ ] **Step 2: Write the failing test**

```rust
// crates/dsl_compiler/tests/build_helper_emits_terrain.rs
//! Asserts emit_namespaced writes terrain_gen.rs into the fixture's
//! OUT_DIR subdir when the source has a terrain block — and does NOT
//! write it when there is no terrain block (so existing fixtures
//! stay quiet).

use std::path::PathBuf;
use dsl_compiler::build_helper;

fn fake_env(tmp: &tempfile::TempDir, sim_name: &str, sim_src: &str) -> (PathBuf, PathBuf) {
    // Lay out a fake workspace: <tmp>/crates/sims/, <tmp>/assets/sim/<sim>.sim
    let sims_dir = tmp.path().join("crates/sims");
    let assets_dir = tmp.path().join("assets/sim");
    std::fs::create_dir_all(&sims_dir).unwrap();
    std::fs::create_dir_all(&assets_dir).unwrap();
    let sim_path = assets_dir.join(format!("{sim_name}.sim"));
    std::fs::write(&sim_path, sim_src).unwrap();

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    std::env::set_var("CARGO_MANIFEST_DIR", &sims_dir);
    std::env::set_var("OUT_DIR", &out_dir);
    (sims_dir, out_dir)
}

#[test]
fn writes_terrain_gen_rs_when_source_has_terrain_block() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sims, out_dir) = fake_env(&tmp, "with_terrain", r#"
terrain {
  extent: 8
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: grass }
}
"#);
    build_helper::emit_namespaced("with_terrain");
    let path = out_dir.join("with_terrain/terrain_gen.rs");
    assert!(path.exists(), "terrain_gen.rs not written: {path:?}");
    let body = std::fs::read_to_string(&path).unwrap();
    assert!(body.contains("pub fn generate_terrain"), "missing generate_terrain in: {body}");
}

#[test]
fn skips_terrain_gen_rs_when_source_has_no_terrain_block() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sims, out_dir) = fake_env(&tmp, "no_terrain", "// empty sim\n");
    // emit_namespaced may panic on an entirely empty .sim depending on
    // parser rules — guard with catch_unwind so the test isolates the
    // single behaviour under check.
    let _ = std::panic::catch_unwind(|| build_helper::emit_namespaced("no_terrain"));
    let path = out_dir.join("no_terrain/terrain_gen.rs");
    assert!(!path.exists(), "terrain_gen.rs unexpectedly written for non-terrain source");
}
```

(`tempfile` is already a dev-dep after Task 9.)

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test build_helper_emits_terrain`
Expected: FAIL — emit_namespaced does not yet write `terrain_gen.rs`.

- [ ] **Step 4: Modify `emit_namespaced_with_strategy`**

After the existing `runtime_core.rs` write, add a conditional terrain emit. Locate the parsed program inside the function (`build_helper.rs` already parses + lowers in one pass — re-use the Program it already has). Sketch (adapt to the actual variable names in the file):

```rust
// In emit_namespaced_with_strategy, after the runtime_core.rs write:
//
// `program` is the already-parsed dsl_ast::ast::Program for this fixture.
// `out_dir` is the namespaced OUT_DIR/<fixture>/ PathBuf.

if let Some(terrain_block) = &program.terrain {
    let ir = crate::cg::lower::lower_terrain(terrain_block)
        .unwrap_or_else(|e| panic!("lower terrain for `{fixture_name}`: {e:?}"));
    let body = crate::cg::emit::emit_terrain(&ir);
    fs::write(out_dir.join("terrain_gen.rs"), body)
        .unwrap_or_else(|e| panic!("write {fixture_name}/terrain_gen.rs: {e}"));
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test build_helper_emits_terrain`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add crates/dsl_compiler/src/build_helper.rs crates/dsl_compiler/tests/build_helper_emits_terrain.rs
git commit -m "feat(dsl): emit_namespaced writes terrain_gen.rs when source has terrain block"
```

---

## Task 12: New `terrain_probe.sim` fixture + sims build.rs wire-up

**Files:**
- Create: `assets/sim/terrain_probe.sim`
- Modify: `crates/sims/build.rs` (allow-list + stub generation).

- [ ] **Step 1: Create the fixture**

```text
// assets/sim/terrain_probe.sim
terrain {
  extent: 8
  cell_size: 1.0
  seed_purpose: 0xBA5E_7E55
  materials {
    grass { id: 1, walkable: true,  hardness: 1, color: 0x4A8B3A }
    stone { id: 2, walkable: false, hardness: 8, color: 0x808080 }
  }
  layer fill { material: stone }
}
```

- [ ] **Step 2: Add to the allow-list in `crates/sims/build.rs`**

In the `if !matches!(stem.as_str(), ... )` match arm (around line 39–142), add `"terrain_probe"` alongside the other entries. Alphabetical placement is fine but the file's existing order is informal — match the surrounding style.

- [ ] **Step 3: Extend the stub generator to include terrain_gen.rs when present**

Replace the stub-building block in `sims/build.rs` (around lines 156–171) so the per-fixture include includes a conditional terrain include. The conditional has to be evaluated at build time (when build.rs runs) AND at compile time (the emitted include must be valid Rust). The simplest pattern: read whether OUT_DIR/<fixture>/terrain_gen.rs exists in build.rs and only emit the second `include!` line when it does. Updated block:

```rust
let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
let mut stub = String::new();
stub.push_str(
    "// AUTO-GENERATED by sims/build.rs from assets/sim/*.sim.\n\
     // Do not edit by hand.\n\n",
);
for f in &fixtures {
    let has_terrain = out_dir.join(format!("{f}/terrain_gen.rs")).exists();
    stub.push_str(&format!(
        "#[allow(non_snake_case, unused_imports, unused_variables, dead_code, clippy::all)]\n\
         pub mod {f} {{\n\
         \x20   include!(concat!(env!(\"OUT_DIR\"), \"/{f}/generated.rs\"));\n\
         \x20   include!(concat!(env!(\"OUT_DIR\"), \"/{f}/runtime_core.rs\"));\n",
    ));
    if has_terrain {
        stub.push_str(&format!(
            "\x20   include!(concat!(env!(\"OUT_DIR\"), \"/{f}/terrain_gen.rs\"));\n",
        ));
    }
    stub.push_str("}\n\n");
}
fs::write(out_dir.join("sim_modules.rs"), stub)
    .unwrap_or_else(|e| panic!("write sim_modules.rs: {e}"));
```

- [ ] **Step 4: Verify the megacrate compiles**

Run: `cargo check -p sims 2>&1 | tail -30`
Expected: clean compile. The terrain_probe fixture should now expose `sims::terrain_probe::generate_terrain`, `sims::terrain_probe::MATERIALS`, `sims::terrain_probe::EXTENT`.

If a compile error mentions terrain_probe specifically, it likely means the fixture's `.sim` is missing the agent / rule sections that `runtime_core.rs` expects. Workaround: add a minimal agent declaration to the fixture (one `agent` block with `n: 1`) — terrain init is independent of agent setup but the generated `runtime_core.rs` requires at least one agent decl. Pin the precise minimum by inspecting an existing single-agent fixture: `rg -nA10 "^agent " assets/sim/cooldown_probe.sim | head`.

- [ ] **Step 5: Commit**

```bash
git add assets/sim/terrain_probe.sim crates/sims/build.rs
git commit -m "feat(sims): terrain_probe fixture + conditional terrain_gen.rs include in stub"
```

---

## Task 13: Runtime-gate smoke test in `crates/sims/tests/`

**Files:**
- Create: `crates/sims/tests/terrain_probe_smoke.rs`

The runtime gate is the `TerrainQuery` surface — `height_at`, `walkable`, `line_of_sight` — installed into an `Arc<dyn TerrainQuery>` via the async worker. `SimState::step` is `unimplemented!()` per Plan B1' Task 11, so the smoke test does NOT exercise tick. The changed code path (DSL → emit → engine_voxel) is exercised end-to-end through the worker + the trait surface.

- [ ] **Step 1: Write the smoke test**

```rust
// crates/sims/tests/terrain_probe_smoke.rs
//! Runtime gate (per plan AIS): boot terrain gen on a worker thread,
//! await readiness, install into Arc<dyn TerrainQuery>, assert the
//! trait surface reflects the declared materials.

use std::sync::Arc;
use engine::terrain::TerrainQuery;
use engine_voxel::TerrainGenHandle;
use glam::Vec3;

#[test]
fn async_gen_installs_and_serves_terrain_query() {
    // Generate on a worker thread.
    let handle = TerrainGenHandle::spawn(42, sims::terrain_probe::generate_terrain);
    let terrain = handle.block_until_ready().expect("gen succeeds");

    // Post-condition 1: declared extent + cell value.
    assert_eq!(terrain.extent(), 8);
    assert_eq!(terrain.cell_at(0, 0, 0), 2, "fill stone id=2 should occupy origin");
    assert_eq!(terrain.cell_at(7, 7, 7), 2, "fill stone id=2 should occupy max corner");

    // Post-condition 2: materials table reachable + correct.
    let stone = terrain.materials().get(2).expect("stone in table");
    assert_eq!(stone.walkable, false);
    assert_eq!(stone.hardness, 8);

    // Post-condition 3: install into Arc<dyn TerrainQuery> and
    // exercise the trait surface — this is the actual changed
    // code path the runtime gate must cover.
    let arc: Arc<dyn TerrainQuery + Send + Sync> = Arc::new(terrain);

    // `height_at` over the filled region should be > 0 (solid below).
    let h = arc.height_at(4.0, 4.0);
    assert!(h > 0.0, "expected non-zero ground height over filled region, got {h}");

    // `walkable` for a Walk-mode agent must reflect the stone
    // declaration (walkable: false → false). Use the MovementMode
    // re-export from engine_voxel.
    let walkable = arc.walkable(
        Vec3::new(4.0, 4.0, 4.0),
        engine_voxel::MovementMode::Walk,
    );
    assert!(!walkable, "stone fill must block Walk mode");

    // `line_of_sight` across the solid region must be blocked.
    let los = arc.line_of_sight(Vec3::new(0.5, 4.0, 4.0), Vec3::new(7.5, 4.0, 4.0));
    assert!(!los, "stone fill must block LOS");
}
```

- [ ] **Step 2: Run the smoke test**

Run: `cargo test -p sims --test terrain_probe_smoke`
Expected: 1 passed.

- [ ] **Step 3: If any of the three trait assertions fail**

The likely cause is that `VoxelTerrain::walkable` short-circuits on the cell's voxel value (non-zero = solid) but does NOT yet consult the `MaterialTable` `walkable` flag. For v1 we rely on the existing engine_voxel semantics — non-zero cell = solid = `walkable=false` for Walk mode. If the test asserts a richer "use the MaterialTable walkable flag" behavior and the trait does not yet do that, weaken the assertion to "non-zero cell blocks Walk" or extend `VoxelTerrain::walkable` to consult `materials().get(cell).walkable` first (add this as a Task 13b if needed). Decide based on the existing trait impl: read `crates/engine_voxel/src/lib.rs` around the `TerrainQuery for VoxelTerrain` block.

- [ ] **Step 4: Commit**

```bash
git add crates/sims/tests/terrain_probe_smoke.rs
git commit -m "test(sims): terrain_probe runtime-gate smoke — async-gen + TerrainQuery surface"
```

---

## Task 14: Workspace test — `cargo test` stays green

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `cargo test --workspace 2>&1 | tail -40`
Expected: all crates green. New tests in `dsl_ast`, `dsl_compiler`, `engine_voxel`, and `sims` (`terrain_probe_smoke`) should all appear and pass.

- [ ] **Step 2: If anything is red**

Address each failure. Common causes:
- A pre-existing flaky test (rerun once before debugging).
- The new `VoxelTerrain::materials` field broke a constructor call site → update the call site to use the appropriate constructor.
- A fixture in `crates/sims/build.rs`'s allow-list now fails parsing because the parser learned a new `terrain` keyword that collides with an identifier in an existing `.sim` — grep `assets/sim/*.sim` for the literal `terrain` and rename if so.

- [ ] **Step 3: Commit only if changes were needed**

```bash
git status
# If clean: nothing to commit; move on.
# If files changed: 
git add <files>
git commit -m "fix(terrain-dsl): workspace test fallout from MaterialTable/VoxelTerrain"
```

---

## Plan complete — exit criteria

- [ ] `.sim` files parse a `terrain { extent, cell_size, seed_purpose, materials { ... }, layer fill { material: ... } }` block.
- [ ] Lowering resolves material names to ids; rejects duplicates and unknown refs.
- [ ] Emitter produces a `generate_terrain(world_seed: u64) -> VoxelTerrain` function that compiles against `engine_voxel`.
- [ ] `TerrainGenHandle` runs generation off the main thread and contains panics.
- [ ] `assets/sim/terrain_probe.sim` surfaces as `sims::terrain_probe::generate_terrain`; the worker-thread harness installs it into `Arc<dyn TerrainQuery>` and the trait surface reflects the declared materials.
- [ ] Full workspace `cargo test` passes.

## Follow-up plans (out of scope here)

1. **Layer primitive expansion** — heightfield (perlin/worley/value noise), box / sphere / cylinder, carve_caves, prefab (.vox), region (biome tag), walkable_mask. Each primitive is one task pair: a `layer_*` helper in `engine_voxel::terrain_layers` + a parser/lower/emit arm.
2. **Non-cubic extent** — extend `engine_voxel::VoxelTerrain` to `(x, y, z)` extent; parser accepts a tuple literal for `extent`.
3. **Multi-file terrain inputs** — user-flagged follow-up from the design phase.
4. **Material property access from DSL rules** — surface `materials.get(cell).movement_cost` etc. to the rule language. Will require a schema-hash bump (P2).
5. **GPU generation emitter** — emit a compute shader alongside the CPU generator for large extents.
6. **ahash drift fix** — workspace-wide. Once landed, promote terrain determinism tests from within-process re-runs to committed byte goldens.
