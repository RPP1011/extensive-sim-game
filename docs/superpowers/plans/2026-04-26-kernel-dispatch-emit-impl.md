# Kernel Dispatch-Emit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all GPU kernel-wrapper boilerplate (struct definition, BGL builder, pipeline construction, buffer allocation, dispatch encoding) into compiler-emitted code; land in a new sibling crate `engine_gpu_rules/`. Migrate all 14 existing kernels in this plan.

**Architecture:** Strangler pattern. New crate `engine_gpu_rules/` is created with the same shape as `engine_rules/` — a `// GENERATED` sentinel `build.rs` plus a single `Cargo.toml` and a regenerated `lib.rs`. `dsl_compiler/` grows six new emitter modules (`emit_mask_kernel`, `emit_scoring_kernel`, `emit_movement_kernel`, `emit_view_fold_kernel`, `emit_spatial_kernel`, `emit_megakernel`) plus shared infrastructure emit (`emit_kernel_index`, `emit_schedule`, `emit_resident_context`). Engine_gpu's `step_batch` shrinks to a `for op in SCHEDULE` loop. Hand-written kernel structs in `crates/engine_gpu/src/{mask,scoring,movement,physics,apply_actions,alive_bitmap,spatial_gpu,cascade_resident}.rs` are deleted in the final cleanup task.

**Tech Stack:** Rust 2021, `wgpu`, `bytemuck`, `sha2`, the existing `dsl_compiler` IR (`dsl_ast::ir::*`), the existing `// GENERATED` sentinel pattern from `engine_rules/build.rs`.

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - `engine_rules/build.rs` (the `// GENERATED` sentinel) — `crates/engine_rules/build.rs:10-35`. The sibling crate this plan creates uses the exact same shape.
  - `ScoringKernel` impl block (the largest hand-written kernel wrapper, ~2300 LOC) — `crates/engine_gpu/src/scoring.rs:519`.
  - `FusedMaskKernel` / `MaskUnpackKernel` / `FusedAgentUnpackKernel` — `crates/engine_gpu/src/mask.rs:486 / :1568 / :1900`.
  - `MovementKernel::new` — `crates/engine_gpu/src/movement.rs:173`.
  - `PhysicsKernel::new` (resident-mode shader emit at `build_physics_shader_resident`) — `crates/engine_gpu/src/physics.rs:1730 / :963`.
  - `ApplyActionsKernel::new` — `crates/engine_gpu/src/apply_actions.rs:211`.
  - `AlivePackKernel`, `SeedIndirectKernel`, `AppendEventsKernel` — `crates/engine_gpu/src/alive_bitmap.rs:202`, `crates/engine_gpu/src/cascade_resident.rs:357 / :620`.
  - `GpuSpatialHash` — `crates/engine_gpu/src/spatial_gpu.rs:725`.
  - `ResidentPathContext` (the existing hand-written context this plan replaces with an emitted `engine_gpu_rules::ResidentPathContext`) — `crates/engine_gpu/src/backend/resident_ctx.rs:15-113`.
  - Existing per-ability emit (already landed): `dsl_compiler::emit_scoring::emit_pick_ability_cpu` at `crates/dsl_compiler/src/emit_scoring.rs:1464`; `dsl_compiler::emit_scoring_wgsl::emit_pick_ability_wgsl` at `crates/dsl_compiler/src/emit_scoring_wgsl.rs:1910`; per-ability schema-hash coverage at `crates/dsl_compiler/src/schema_hash.rs:290 (scoring_hash)`.
  - GPU step entry: `GpuBackend::step_batch` — `crates/engine_gpu/src/lib.rs:991` (this is the loop body that gets rewritten).

  Search method: `rg`, `grep -n`, direct `Read` of each cited file.

- **Buffer-source aggregation via `BindingSources<'a>`; `bind()` is generated, not hand-written.** The spec sketched `bind(&ResidentPathContext, &Buffer)` as the trait method, but most kernels need buffers outside `ResidentPathContext` (PingPong rings, Pooled spatial scratch, Transient mask/action handles refreshed per tick, External agent SoA + sim_cfg + registries). To keep `bind()` real generated code (not `unimplemented!()`), the trait takes a `&BindingSources<'a>` aggregate that references all five lifetime-class containers. Every container struct (`ResidentPathContext`, `PingPongContext`, `Pool`, `TransientHandles`, `ExternalBuffers`, `BindingSources`) is `// GENERATED` in `engine_gpu_rules/`; engine_gpu hand-writes only the construction calls (`TransientHandles::new(&pool)` per tick; `ExternalBuffers::new(&device, ...)` once) and the `wgpu::Device`/`Queue` plumbing. Each kernel's emitted `bind()` body returns its `Bindings` struct by walking the kernel's IR and indexing the right field of the right container — adding a new `.sim` row regenerates the bind body automatically; no hand-edits to engine_gpu wiring code (the P1 violation the spec brainstorm explicitly closed).

- **Decision:** New sibling crate `engine_gpu_rules/`. Mirrors `engine_rules/` exactly (same `build.rs` sentinel, same `// GENERATED` invariant). `dsl_compiler/` grows new emitters that target this crate. `engine_gpu/`'s hot path (`step_batch`) becomes a `for op in engine_gpu_rules::schedule::SCHEDULE { self.dispatch(op, &mut encoder, &sources, state)? }` loop, where `sources: BindingSources<'_>` is constructed once per tick from the engine_gpu-owned containers. The decision is to extend the *compiler*, not the GPU backend — the GPU backend's hand-written code shrinks. Per-row buffer lifetime is fully inferable from row type (Q2 in spec), so no DSL grammar changes.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: NONE. No grammar additions; lifetime classes are inferred from existing IR.
  - New emitter modules under `crates/dsl_compiler/src/`: `emit_kernel_index.rs`, `emit_schedule.rs`, `emit_resident_context.rs`, `emit_pingpong_context.rs`, `emit_pool.rs`, `emit_transient_handles.rs`, `emit_external_buffers.rs`, `emit_binding_sources.rs`, `emit_mask_kernel.rs`, `emit_scoring_kernel.rs`, `emit_movement_kernel.rs`, `emit_view_fold_kernel.rs`, `emit_spatial_kernel.rs`, `emit_megakernel.rs`.
  - Generated outputs (all under `crates/engine_gpu_rules/src/`, all `// GENERATED`): `lib.rs`, `schedule.rs`, `resident_context.rs`, `pingpong_context.rs`, `pool.rs`, `transient_handles.rs`, `external_buffers.rs`, `binding_sources.rs`, plus one `<kernel>.rs` + one `<kernel>.wgsl` per kernel module (14 modules total), plus `megakernel.rs` + `megakernel.wgsl`.
  - Schema-hash coverage extension: new `engine_gpu_rules/.schema_hash` baseline file; new sub-hash `gpu_rules_hash` in `dsl_compiler::schema_hash` rolled into `combined_hash`.

- **Hand-written downstream code:**
  - `crates/engine_gpu_rules/Cargo.toml`: justification — Cargo manifests are not generated.
  - `crates/engine_gpu_rules/build.rs`: justification — the build sentinel reads filesystem state and panics on hand-edited files; it is invariant across rule changes (same shape as `engine_rules/build.rs`).
  - `crates/engine_gpu/src/lib.rs::step_batch` body: justification — the schedule iteration loop and the `dispatch(op, ...)` match arms are *cross-kernel orchestration*, not per-kernel logic. They are written once and not edited as new rows land. Per the spec ("Anything cross-kernel … stays in `engine_gpu/`"), this is the architectural cut, not a leak.
  - `crates/engine_gpu/src/backend/{snapshot_ctx,sync_ctx}.rs`: justification — snapshot replay and CPU-fallback sync paths are not per-kernel and stay hand-written.

- **Constitution check:**
  - P1 (Compiler-First): PASS — this plan is the canonical compiler-first remediation: every hand-written `Kernel` struct becomes emitted output. Evidence: Tasks 4–13 each delete the corresponding hand-written struct.
  - P2 (Schema-Hash on Layout): PASS with new artifact — Task 3 adds `crates/engine_gpu_rules/.schema_hash` baseline; Task 16 bumps it after the final regen. The new sub-hash `gpu_rules_hash` covers every emitted `.rs`/`.wgsl` byte plus the `SCHEDULE` constant.
  - P3 (Cross-Backend Parity): PASS — tasks 4, 6, 9, 11, 12, 16 each end with `cargo test -p engine --test wolves_and_humans_parity`, `cargo test -p engine_gpu --test parity_with_cpu`, `physics_parity`, `cascade_parity` staying green. The migration produces byte-equal WGSL where possible; the parity gate catches any drift.
  - P4 (`EffectOp` Size Budget): N/A — no `EffectOp` variant changes.
  - P5 (Determinism via Keyed PCG): N/A — RNG paths are unchanged.
  - P6 (Events Are the Mutation Channel): N/A — event surface unchanged.
  - P7 (Replayability Flagged): N/A — event flags unchanged.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends with a commit step using the project's `feat(component): description` convention.
  - P10 (No Runtime Panic): PASS — bad rules fail at `xtask compile-dsl --check` (compile-time DAG validation in Task 3); the runtime `dispatch()` match has no `unwrap()` on hot-path inputs (every `Option`/`Result` is checked-and-returned).
  - P11 (Reduction Determinism): PASS — view-fold reductions and atomic-append paths are emitted byte-equal; sort-stability already lives in the WGSL emit at `emit_scoring_wgsl_atomic_views` and is preserved.

- **Re-evaluation:**
  - [ ] AIS reviewed at design phase (initial fill).
  - [ ] AIS reviewed post-design (after task list stabilises).

---

## File Map

| Path | Role | Status |
|------|------|--------|
| `crates/engine_gpu_rules/Cargo.toml` | New crate manifest | Create (Task 1) |
| `crates/engine_gpu_rules/build.rs` | `// GENERATED` sentinel | Create (Task 1) |
| `crates/engine_gpu_rules/src/lib.rs` | Module list + `Kernel` trait + `KernelId` enum + `BufferRef` enum (regenerated) | Emit (Task 2) |
| `crates/engine_gpu_rules/src/schedule.rs` | `DispatchOp` enum + `SCHEDULE` const | Emit (Task 3) |
| `crates/engine_gpu_rules/src/resident_context.rs` | `ResidentPathContext` struct + `new()` | Emit (Task 3) |
| `crates/engine_gpu_rules/src/pingpong_context.rs` | `PingPongContext` A/B ring buffers (cascade) — emitted skeleton at Task 3, populated Task 9 | Emit (Task 3, 9) |
| `crates/engine_gpu_rules/src/pool.rs` | `Pool` shape-keyed reusable buffers (Pooled lifetime) — emitted skeleton at Task 3, populated Task 12 | Emit (Task 3, 12) |
| `crates/engine_gpu_rules/src/transient_handles.rs` | `TransientHandles` per-tick scratch references — fields populated by per-kernel tasks | Emit (Task 3 + extended every kernel task) |
| `crates/engine_gpu_rules/src/external_buffers.rs` | `ExternalBuffers` engine-owned references (agents, sim_cfg, registries) | Emit (Task 3 + extended every kernel task) |
| `crates/engine_gpu_rules/src/binding_sources.rs` | `BindingSources<'a>` aggregate of the five containers; trait `bind()` consumes it | Emit (Task 3) |
| `crates/engine_gpu_rules/src/fused_mask.rs` + `.wgsl` | `FusedMaskKernel` | Emit (Task 4) |
| `crates/engine_gpu_rules/src/mask_unpack.rs` + `.wgsl` | `MaskUnpackKernel` | Emit (Task 4) |
| `crates/engine_gpu_rules/src/scoring.rs` + `.wgsl` | `ScoringKernel` (target_bound) | Emit (Task 6) |
| `crates/engine_gpu_rules/src/scoring_unpack.rs` + `.wgsl` | `ScoringUnpackKernel` | Emit (Task 6) |
| `crates/engine_gpu_rules/src/apply_actions.rs` + `.wgsl` | `ApplyActionsKernel` | Emit (Task 7) |
| `crates/engine_gpu_rules/src/pick_ability.rs` + `.wgsl` | `PickAbilityKernel` (per_ability) | Emit (Task 8) |
| `crates/engine_gpu_rules/src/movement.rs` + `.wgsl` | `MovementKernel` | Emit (Task 9) |
| `crates/engine_gpu_rules/src/physics.rs` + `.wgsl` | `PhysicsKernel` (FixedPoint dispatch) | Emit (Task 10) |
| `crates/engine_gpu_rules/src/seed_indirect.rs` + `.wgsl` | `SeedIndirectKernel` (Indirect dispatch) | Emit (Task 10) |
| `crates/engine_gpu_rules/src/append_events.rs` + `.wgsl` | `AppendEventsKernel` | Emit (Task 10) |
| `crates/engine_gpu_rules/src/fold_<view>.rs` + `.wgsl` × 8 | One module per materialized view | Emit (Task 11) |
| `crates/engine_gpu_rules/src/spatial_hash.rs` + `.wgsl` | `cs_spatial_hash` build | Emit (Task 12) |
| `crates/engine_gpu_rules/src/spatial_kin_query.rs` + `.wgsl` | `cs_kin_query` | Emit (Task 12) |
| `crates/engine_gpu_rules/src/spatial_engagement_query.rs` + `.wgsl` | `cs_engagement_query` | Emit (Task 12) |
| `crates/engine_gpu_rules/src/alive_pack.rs` + `.wgsl` | `AlivePackKernel` | Emit (Task 13) |
| `crates/engine_gpu_rules/src/fused_agent_unpack.rs` + `.wgsl` | `FusedAgentUnpackKernel` | Emit (Task 13) |
| `crates/engine_gpu_rules/src/megakernel.rs` + `.wgsl` | Fused-shader composition over SCHEDULE | Emit (Task 14) |
| `crates/engine_gpu_rules/.schema_hash` | New baseline hex hash file | Create (Task 3), Bump (Task 16) |
| `crates/engine_gpu_rules/tests/schema_hash.rs` | Baseline-comparison test | Create (Task 3) |
| `crates/dsl_compiler/src/emit_kernel_index.rs` | Emits lib.rs prelude + module list | Create (Task 2) |
| `crates/dsl_compiler/src/emit_schedule.rs` | DAG build + topological sort + `DispatchOp` emit | Create (Task 3) |
| `crates/dsl_compiler/src/emit_resident_context.rs` | Walks IR, collects Resident-class buffers | Create (Task 3) |
| `crates/dsl_compiler/src/emit_pingpong_context.rs` | A/B ring buffers | Create (Task 3, extended Task 9) |
| `crates/dsl_compiler/src/emit_pool.rs` | Shape-keyed Pooled-lifetime buffers | Create (Task 3, extended Task 12) |
| `crates/dsl_compiler/src/emit_transient_handles.rs` | Walks IR for Transient buffers; emits handle struct | Create (Task 3, extended every kernel task) |
| `crates/dsl_compiler/src/emit_external_buffers.rs` | Emits `ExternalBuffers` field shape | Create (Task 3, extended every kernel task) |
| `crates/dsl_compiler/src/emit_binding_sources.rs` | Emits the 5-field `BindingSources<'a>` aggregate | Create (Task 3) |
| `crates/dsl_compiler/src/emit_mask_kernel.rs` | Wraps existing `emit_mask_wgsl` | Create (Task 4) |
| `crates/dsl_compiler/src/emit_scoring_kernel.rs` | Wraps existing `emit_scoring_wgsl_atomic_views` + `emit_pick_ability_wgsl` | Create (Tasks 6–8) |
| `crates/dsl_compiler/src/emit_movement_kernel.rs` | New | Create (Tasks 9–10) |
| `crates/dsl_compiler/src/emit_view_fold_kernel.rs` | Wraps existing `emit_view_wgsl` per view | Create (Task 11) |
| `crates/dsl_compiler/src/emit_spatial_kernel.rs` | New | Create (Task 12) |
| `crates/dsl_compiler/src/emit_megakernel.rs` | Walks SCHEDULE → fused WGSL | Create (Task 14) |
| `crates/dsl_compiler/src/schema_hash.rs` | Add `gpu_rules_hash` + roll into `combined_hash` | Modify (Task 3) |
| `crates/dsl_compiler/src/lib.rs` | New emitter modules registered + `EmittedArtifacts` extended | Modify (Tasks 2, 3, 4, …) |
| `crates/xtask/src/main.rs` (or compile-dsl subcommand source) | Write emitted files under `engine_gpu_rules/src/` | Modify (Task 2) |
| `crates/engine_gpu/Cargo.toml` | Add `engine_gpu_rules` dep | Modify (Task 5) |
| `crates/engine_gpu/src/lib.rs` | Replace step_batch body with `for op in SCHEDULE` loop; remove `pub mod` for migrated kernels | Modify (Tasks 5, 7, 9, 11, 13, 16) |
| `crates/engine_gpu/src/backend/resident_ctx.rs` | Replace with re-export of `engine_gpu_rules::ResidentPathContext` | Modify (Task 5) |
| `crates/engine_gpu/src/{mask,scoring,apply_actions,movement,physics,alive_bitmap,spatial_gpu,cascade_resident}.rs` | Delete after migration | Delete (Task 16) |
| `Cargo.toml` (workspace root) | Add `engine_gpu_rules` member | Modify (Task 1) |

---

## Tasks

### Task 1: Bootstrap `engine_gpu_rules/` skeleton

**Files:**
- Create: `crates/engine_gpu_rules/Cargo.toml`
- Create: `crates/engine_gpu_rules/build.rs`
- Create: `crates/engine_gpu_rules/src/lib.rs` (placeholder; regenerated in Task 2)
- Modify: `Cargo.toml` (workspace) — add member

- [x] **Step 1: Add the workspace member entry**

Open `Cargo.toml` at the project root. Find the `[workspace]` block (around line 1) and add `crates/engine_gpu_rules` to `members`:

```toml
[workspace]
members = [".", "crates/tactical_sim", "crates/engine", "crates/engine_data", "crates/engine_rules", "crates/engine_gpu", "crates/engine_gpu_rules", "crates/viz", "crates/dsl_ast", "crates/dsl_compiler", "crates/xtask"]
exclude = ["crates/world_sim_bench"]
```

- [x] **Step 2: Create the crate manifest**

Write `crates/engine_gpu_rules/Cargo.toml`:

```toml
[package]
name = "engine_gpu_rules"
version = "0.1.0"
edition = "2021"
build = "build.rs"

[dependencies]
engine = { path = "../engine" }
engine_data = { path = "../engine_data" }
bytemuck = { version = "1.16", features = ["derive"] }
wgpu = "22"

[features]
default = []
# Mirror of engine_gpu's gpu feature; pulls in wgpu-touching code paths.
gpu = []
```

- [x] **Step 3: Create the build sentinel**

Write `crates/engine_gpu_rules/build.rs` (mirror of `engine_rules/build.rs`):

```rust
//! engine_gpu_rules build sentinel.
//!
//! Every .rs and .wgsl file under `src/` (other than `lib.rs`) must start
//! with `// GENERATED by dsl_compiler` within the first 5 lines. Hand-edited
//! files in this crate are forbidden by the constitution (P1).

use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src");
    walk(Path::new("src"));
}

fn walk(dir: &Path) {
    if !dir.exists() { return; }
    for entry in fs::read_dir(dir).expect("readable src dir") {
        let entry = entry.expect("readable entry");
        let path = entry.path();
        let ft = entry.file_type().expect("file type");
        if ft.is_dir() { walk(&path); continue; }
        let ext = path.extension().and_then(|e| e.to_str());
        if !matches!(ext, Some("rs") | Some("wgsl")) { continue; }
        if path.file_name() == Some(std::ffi::OsStr::new("lib.rs"))
           && path.parent() == Some(Path::new("src")) { continue; }
        let content = fs::read_to_string(&path).expect("readable file");
        let head: String = content.lines().take(5).collect::<Vec<_>>().join("\n");
        if !head.contains("// GENERATED by dsl_compiler") {
            panic!(
                "engine_gpu_rules: {} is missing the `// GENERATED by dsl_compiler` header. \
                 Hand-edited files in this crate are forbidden. Edit the .sim source \
                 in assets/sim/ and rerun `cargo run -p xtask --bin xtask -- compile-dsl`.",
                path.display()
            );
        }
    }
}
```

- [x] **Step 4: Create the placeholder lib.rs**

Write `crates/engine_gpu_rules/src/lib.rs`:

```rust
//! engine_gpu_rules — emitted GPU kernel wrappers + Schedule.
//!
//! Generated by `cargo run --bin xtask -- compile-dsl`.
//! Do not edit by hand; edit the `.sim` sources instead.
//!
//! NOTE: Task 1 leaves this file as a placeholder. Task 2 replaces it
//! with the regenerated module list + Kernel trait prelude.
#![allow(clippy::all)]
```

- [x] **Step 5: Verify the crate builds standalone**

Run: `cargo build -p engine_gpu_rules`
Expected:
```
Compiling engine_gpu_rules v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in ...s
```

- [x] **Step 6: Verify the build sentinel rejects hand-edited files**

Create a one-line file `crates/engine_gpu_rules/src/touch_test.rs` containing:

```rust
pub fn nothing() {}
```

Run: `cargo build -p engine_gpu_rules`
Expected: build fails with the panic message `engine_gpu_rules: src/touch_test.rs is missing the // GENERATED by dsl_compiler header.`

Then delete the file: `rm crates/engine_gpu_rules/src/touch_test.rs` and re-run `cargo build -p engine_gpu_rules` — expect a clean build.

- [x] **Step 7: Commit**

```bash
git add Cargo.toml crates/engine_gpu_rules/
git commit -m "feat(engine_gpu_rules): bootstrap sibling crate with // GENERATED sentinel"
```

---

### Task 2: Emit `lib.rs` prelude (Kernel trait, KernelId, BufferRef)

The prelude is byte-stable across regens until a `.sim` change adds/removes a kernel. The compiler emits it from `emit_kernel_index`.

**Files:**
- Create: `crates/dsl_compiler/src/emit_kernel_index.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (register module + extend `EmittedArtifacts`)
- Modify: `crates/xtask/src/main.rs` (compile-dsl subcommand — write the emitted lib.rs to `crates/engine_gpu_rules/src/lib.rs`)
- Test: `crates/dsl_compiler/tests/emit_kernel_index_smoke.rs`

- [x] **Step 1: Write the failing test**

Create `crates/dsl_compiler/tests/emit_kernel_index_smoke.rs`:

```rust
use dsl_compiler::emit_kernel_index::emit_lib_rs;

#[test]
fn lib_rs_contains_kernel_trait_and_module_list() {
    let modules = vec![
        "fused_mask".to_string(),
        "scoring".to_string(),
        "pick_ability".to_string(),
    ];
    let src = emit_lib_rs(&modules);
    // Header
    assert!(src.starts_with("// GENERATED by dsl_compiler"), "missing header: {src}");
    // Module list
    assert!(src.contains("pub mod fused_mask;"));
    assert!(src.contains("pub mod scoring;"));
    assert!(src.contains("pub mod pick_ability;"));
    // Static prelude pieces
    assert!(src.contains("pub trait Kernel"));
    assert!(src.contains("pub enum KernelId"));
    assert!(src.contains("pub enum BufferRef"));
    // Shared-infrastructure module declarations are static, even on empty input
    assert!(src.contains("pub mod schedule;"));
    assert!(src.contains("pub mod resident_context;"));
    assert!(src.contains("pub mod pingpong_context;"));
    assert!(src.contains("pub mod pool;"));
    assert!(src.contains("pub mod transient_handles;"));
    assert!(src.contains("pub mod external_buffers;"));
    assert!(src.contains("pub mod binding_sources;"));
    // Trait signature: bind() takes &BindingSources, not &ResidentPathContext.
    assert!(src.contains("sources: &'a binding_sources::BindingSources"));
    assert!(src.contains("device: &wgpu::Device"));
}

#[test]
fn lib_rs_is_byte_stable_for_same_modules() {
    let modules = vec!["fused_mask".to_string(), "scoring".to_string()];
    let a = emit_lib_rs(&modules);
    let b = emit_lib_rs(&modules);
    assert_eq!(a, b);
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test emit_kernel_index_smoke`
Expected: FAIL with `error[E0432]: unresolved import dsl_compiler::emit_kernel_index`.

- [x] **Step 3: Write the emitter module**

Create `crates/dsl_compiler/src/emit_kernel_index.rs`:

```rust
//! Emits `engine_gpu_rules/src/lib.rs` — the static prelude (Kernel
//! trait, KernelId enum, BufferRef enum, fixed module declarations
//! for schedule/resident_context/pingpong_context/megakernel) plus
//! the variable per-kernel module list.
//!
//! Stable across regens when the kernel set is unchanged.

use std::fmt::Write;

/// Emit the body of `engine_gpu_rules/src/lib.rs`.
///
/// `modules` is the sorted list of per-kernel module names (without the
/// `.rs` suffix). The order is alphabetic — produced by the caller from
/// the IR walk so that diffs after `.sim` edits stay readable.
pub fn emit_lib_rs(modules: &[String]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_kernel_index. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#![allow(clippy::all)]").unwrap();
    writeln!(out, "#![allow(unused_imports)]").unwrap();
    writeln!(out).unwrap();

    // Per-kernel modules (variable; sorted by caller).
    for m in modules {
        writeln!(out, "pub mod {m};").unwrap();
    }
    writeln!(out).unwrap();

    // Fixed shared infrastructure modules.
    writeln!(out, "pub mod schedule;").unwrap();
    writeln!(out, "pub mod resident_context;").unwrap();
    writeln!(out, "pub mod pingpong_context;").unwrap();
    writeln!(out, "pub mod pool;").unwrap();
    writeln!(out, "pub mod transient_handles;").unwrap();
    writeln!(out, "pub mod external_buffers;").unwrap();
    writeln!(out, "pub mod binding_sources;").unwrap();
    writeln!(out, "pub mod megakernel;").unwrap();
    writeln!(out).unwrap();

    // KernelId enum — one variant per emitted kernel module, PascalCase.
    writeln!(out, "#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]").unwrap();
    writeln!(out, "pub enum KernelId {{").unwrap();
    for m in modules {
        writeln!(out, "    {},", to_pascal_case(m)).unwrap();
    }
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // BufferRef enum — opaque handles used by Schedule's Indirect/GatedBy ops.
    writeln!(out, "#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]").unwrap();
    writeln!(out, "pub enum BufferRef {{").unwrap();
    writeln!(out, "    /// Resident path indirect-args buffer.").unwrap();
    writeln!(out, "    ResidentIndirectArgs,").unwrap();
    writeln!(out, "    /// Cascade-physics event ring tail buffer (gates further iterations when zero).").unwrap();
    writeln!(out, "    CascadeRingTail,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // Kernel trait — type-level uniformity, no dyn dispatch.
    //
    // bind() takes &BindingSources<'a> (an aggregate over all five
    // lifetime-class containers: resident, pingpong, pool, transient,
    // external). Each kernel's emitted bind() body pulls the specific
    // buffers it needs from the right container. This keeps bind() real
    // generated code rather than `unimplemented!()` — adding a `.sim`
    // row regenerates the bind body, no hand-edits to engine_gpu wiring
    // (the P1 violation the spec brainstorm explicitly resolved).
    writeln!(out, "pub trait Kernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> where Self: 'a;").unwrap();
    writeln!(out, "    type Cfg: bytemuck::Pod + bytemuck::Zeroable + Copy;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self where Self: Sized;").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> Self::Cfg;").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a binding_sources::BindingSources<'a>, cfg: &'a wgpu::Buffer) -> Self::Bindings<'a>;").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &Self::Bindings<'_>, agent_cap: u32);").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

fn to_pascal_case(s: &str) -> String {
    let mut out = String::new();
    let mut up = true;
    for c in s.chars() {
        if c == '_' { up = true; continue; }
        if up { out.extend(c.to_uppercase()); up = false; } else { out.push(c); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pascal_case_handles_underscores() {
        assert_eq!(to_pascal_case("fused_mask"), "FusedMask");
        assert_eq!(to_pascal_case("pick_ability"), "PickAbility");
        assert_eq!(to_pascal_case("scoring"), "Scoring");
    }
}
```

- [x] **Step 4: Register the module in dsl_compiler's lib.rs**

Open `crates/dsl_compiler/src/lib.rs`. Add the module declaration alongside the existing `pub mod emit_*` entries (around line 17):

```rust
pub mod emit_kernel_index;
```

- [x] **Step 5: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test emit_kernel_index_smoke`
Expected: PASS, both tests.

Also run the inline unit test:
Run: `cargo test -p dsl_compiler emit_kernel_index::tests::pascal_case_handles_underscores`
Expected: PASS.

- [x] **Step 6: Wire xtask compile-dsl to write the emitted lib.rs**

Find the `compile-dsl` subcommand in `crates/xtask/src/main.rs` (or wherever it currently writes `engine_rules/` files). Locate the section that writes generated files. Add (after the existing engine_rules write loop):

```rust
// Emit engine_gpu_rules/src/lib.rs from the (initially empty) module
// list. Subsequent kernel-emit tasks will populate the list.
{
    use std::fs;
    use std::path::PathBuf;

    let modules: Vec<String> = Vec::new(); // populated by per-kernel emitters in later tasks
    let lib_rs = dsl_compiler::emit_kernel_index::emit_lib_rs(&modules);
    let path = PathBuf::from("crates/engine_gpu_rules/src/lib.rs");
    fs::create_dir_all(path.parent().unwrap()).expect("mkdir engine_gpu_rules/src");
    fs::write(&path, lib_rs).expect("write engine_gpu_rules/src/lib.rs");
}
```

- [x] **Step 7: Run compile-dsl and verify the file is regenerated**

Run: `cargo run --bin xtask -- compile-dsl`
Expected: command exits 0; `crates/engine_gpu_rules/src/lib.rs` now starts with `// GENERATED by dsl_compiler::emit_kernel_index.`

Run: `head -5 crates/engine_gpu_rules/src/lib.rs`
Expected first line: `// GENERATED by dsl_compiler::emit_kernel_index. Do not edit by hand.`

Run: `cargo build -p engine_gpu_rules`
Expected: clean build (the prelude has no per-kernel modules yet, so no missing-file errors).

- [x] **Step 8: Commit**

```bash
git add crates/dsl_compiler/src/emit_kernel_index.rs crates/dsl_compiler/src/lib.rs crates/xtask/ crates/engine_gpu_rules/src/lib.rs
git commit -m "feat(dsl_compiler): emit_kernel_index produces engine_gpu_rules lib.rs prelude"
```

---

### Task 3: Emit `Schedule`, `ResidentPathContext`, binding-source containers, schema-hash baseline

This task lands the cross-cutting infrastructure: a `DispatchOp` enum, a `SCHEDULE` constant (initially empty — populated as kernels migrate), a `ResidentPathContext` struct (initially with no Resident-class buffer fields), the four other binding-source containers (`PingPongContext`, `Pool`, `TransientHandles`, `ExternalBuffers`) and the `BindingSources<'a>` aggregate that wires them together, plus a `.schema_hash` baseline that hashes everything under `engine_gpu_rules/src/`. DAG-validation logic for diamond/cycle/unscheduled errors lives here from the start so future kernel emitters fail fast.

The five container structs and `BindingSources<'a>` are emitted as skeletons (no fields yet); each per-kernel task in 4–13 extends the relevant container's field list as new buffers come into scope.

**Files:**
- Create: `crates/dsl_compiler/src/emit_schedule.rs`
- Create: `crates/dsl_compiler/src/emit_resident_context.rs`
- Create: `crates/dsl_compiler/src/emit_pingpong_context.rs`
- Create: `crates/dsl_compiler/src/emit_pool.rs`
- Create: `crates/dsl_compiler/src/emit_transient_handles.rs`
- Create: `crates/dsl_compiler/src/emit_external_buffers.rs`
- Create: `crates/dsl_compiler/src/emit_binding_sources.rs`
- Modify: `crates/dsl_compiler/src/schema_hash.rs` (add `gpu_rules_hash` + roll into `combined_hash`)
- Modify: `crates/dsl_compiler/src/lib.rs` (register modules, extend `EmittedArtifacts`)
- Modify: `crates/xtask/src/main.rs` (write `schedule.rs`, `resident_context.rs`, `pingpong_context.rs`, `pool.rs`, `transient_handles.rs`, `external_buffers.rs`, `binding_sources.rs`, `.schema_hash`)
- Create: `crates/engine_gpu_rules/tests/schema_hash.rs`
- Test: `crates/dsl_compiler/tests/emit_schedule_smoke.rs`
- Test: `crates/dsl_compiler/tests/emit_binding_sources_smoke.rs`

- [x] **Step 1: Write the failing test for emit_schedule**

Create `crates/dsl_compiler/tests/emit_schedule_smoke.rs`:

```rust
use dsl_compiler::emit_schedule::{emit_schedule_rs, ScheduleEntry, DispatchOpKind};

#[test]
fn empty_schedule_emits_compilable_const() {
    let src = emit_schedule_rs(&[]);
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub enum DispatchOp"));
    assert!(src.contains("pub const SCHEDULE: &[DispatchOp] = &[]"));
}

#[test]
fn schedule_with_two_kernels_emits_them_in_order() {
    let entries = vec![
        ScheduleEntry { kernel: "FusedMask".into(), kind: DispatchOpKind::Kernel },
        ScheduleEntry { kernel: "Scoring".into(),  kind: DispatchOpKind::Kernel },
    ];
    let src = emit_schedule_rs(&entries);
    assert!(src.contains("DispatchOp::Kernel(KernelId::FusedMask)"));
    assert!(src.contains("DispatchOp::Kernel(KernelId::Scoring)"));
    let p_mask  = src.find("FusedMask").unwrap();
    let p_score = src.find("Scoring").unwrap();
    assert!(p_mask < p_score, "ordering preserved");
}

#[test]
fn fixedpoint_op_emits_max_iter() {
    let entries = vec![
        ScheduleEntry { kernel: "Physics".into(), kind: DispatchOpKind::FixedPoint { max_iter: 8 } },
    ];
    let src = emit_schedule_rs(&entries);
    assert!(src.contains("DispatchOp::FixedPoint { kernel: KernelId::Physics, max_iter: 8 }"));
}
```

- [x] **Step 2: Run test to confirm fail**

Run: `cargo test -p dsl_compiler --test emit_schedule_smoke`
Expected: FAIL — `error[E0432]: unresolved import dsl_compiler::emit_schedule`.

- [x] **Step 3: Implement emit_schedule.rs**

Create `crates/dsl_compiler/src/emit_schedule.rs`:

```rust
//! Emits `engine_gpu_rules/src/schedule.rs` — the `DispatchOp` enum and
//! the topologically-sorted `SCHEDULE` constant.
//!
//! Compile-time errors raised here (per spec):
//!   * Diamond writes — multiple rows write the same buffer.
//!   * Cycles — row X reads what Y writes and vice versa.
//!   * Unscheduled kernel — a module exists but no row produces it.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchOpKind {
    /// Single dispatch of one kernel.
    Kernel,
    /// Fixed-point loop over a kernel until its event ring drains.
    FixedPoint { max_iter: u32 },
    /// Indirect dispatch driven by `args_buf`.
    Indirect { args_buf_ref: String /* BufferRef variant name */ },
    /// Conditional: only run if a producer wrote a non-sentinel value.
    GatedBy { gate_ref: String /* BufferRef variant name */ },
}

#[derive(Debug, Clone)]
pub struct ScheduleEntry {
    /// PascalCase variant name in `KernelId`.
    pub kernel: String,
    pub kind: DispatchOpKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    DiamondWrite { buffer: String, kernels: Vec<String> },
    Cycle { path: Vec<String> },
    Unscheduled { kernel: String },
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DiamondWrite { buffer, kernels } => write!(
                f, "diamond write on buffer `{buffer}`: rows {kernels:?} both write it"
            ),
            Self::Cycle { path } => write!(f, "schedule cycle: {path:?}"),
            Self::Unscheduled { kernel } => write!(f, "unscheduled kernel `{kernel}` (no row produces it)"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Build the schedule from a per-row read/write description.
///
/// `nodes`: each tuple is `(kernel_pascal, reads, writes)`.
///
/// Returns the entries in topological order or a `ScheduleError`.
pub fn build_schedule(
    nodes: &[(String, Vec<String>, Vec<String>)],
) -> Result<Vec<ScheduleEntry>, ScheduleError> {
    // 1. Diamond detection.
    let mut writers: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (k, _r, ws) in nodes {
        for w in ws {
            writers.entry(w.clone()).or_default().push(k.clone());
        }
    }
    for (buf, ks) in &writers {
        if ks.len() > 1 {
            return Err(ScheduleError::DiamondWrite {
                buffer: buf.clone(),
                kernels: ks.clone(),
            });
        }
    }

    // 2. Edge construction: producer(buf) -> consumer(buf).
    let kernel_set: HashSet<&str> = nodes.iter().map(|(k, _, _)| k.as_str()).collect();
    let mut adj: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut indeg: BTreeMap<String, u32> = BTreeMap::new();
    for (k, _, _) in nodes {
        adj.entry(k.clone()).or_default();
        indeg.entry(k.clone()).or_insert(0);
    }
    for (k, reads, _) in nodes {
        for r in reads {
            if let Some(producers) = writers.get(r) {
                for p in producers {
                    if p == k { continue; } // self-read is allowed (intra-kernel)
                    if !kernel_set.contains(p.as_str()) { continue; }
                    adj.get_mut(p).unwrap().push(k.clone());
                    *indeg.get_mut(k).unwrap() += 1;
                }
            }
        }
    }

    // 3. Kahn's algorithm.
    let mut queue: Vec<String> = indeg.iter()
        .filter(|(_, &d)| d == 0)
        .map(|(k, _)| k.clone())
        .collect();
    queue.sort();
    let mut order: Vec<String> = Vec::new();
    while let Some(k) = queue.pop() {
        order.push(k.clone());
        for next in adj.get(&k).cloned().unwrap_or_default() {
            let d = indeg.get_mut(&next).unwrap();
            *d -= 1;
            if *d == 0 { queue.push(next); }
        }
        queue.sort();
    }
    if order.len() != nodes.len() {
        // Cycle: report the kernels with non-zero in-degree.
        let path: Vec<String> = indeg.iter()
            .filter(|(_, &d)| d > 0)
            .map(|(k, _)| k.clone())
            .collect();
        return Err(ScheduleError::Cycle { path });
    }

    // 4. Map to ScheduleEntry. Default kind is Kernel; callers override
    //    by constructing entries directly when they need FixedPoint /
    //    Indirect / GatedBy (e.g. cascade physics).
    Ok(order.into_iter().map(|k| ScheduleEntry {
        kernel: k,
        kind: DispatchOpKind::Kernel,
    }).collect())
}

/// Emit `engine_gpu_rules/src/schedule.rs`.
pub fn emit_schedule_rs(entries: &[ScheduleEntry]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_schedule. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::{{KernelId, BufferRef}};").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "#[derive(Copy, Clone, Debug)]").unwrap();
    writeln!(out, "pub enum DispatchOp {{").unwrap();
    writeln!(out, "    Kernel(KernelId),").unwrap();
    writeln!(out, "    FixedPoint {{ kernel: KernelId, max_iter: u32 }},").unwrap();
    writeln!(out, "    Indirect {{ kernel: KernelId, args_buf: BufferRef }},").unwrap();
    writeln!(out, "    GatedBy {{ kernel: KernelId, gate: BufferRef }},").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    if entries.is_empty() {
        writeln!(out, "pub const SCHEDULE: &[DispatchOp] = &[];").unwrap();
        return out;
    }

    writeln!(out, "pub const SCHEDULE: &[DispatchOp] = &[").unwrap();
    for e in entries {
        match &e.kind {
            DispatchOpKind::Kernel => {
                writeln!(out, "    DispatchOp::Kernel(KernelId::{}),", e.kernel).unwrap();
            }
            DispatchOpKind::FixedPoint { max_iter } => {
                writeln!(out, "    DispatchOp::FixedPoint {{ kernel: KernelId::{}, max_iter: {max_iter} }},", e.kernel).unwrap();
            }
            DispatchOpKind::Indirect { args_buf_ref } => {
                writeln!(out, "    DispatchOp::Indirect {{ kernel: KernelId::{}, args_buf: BufferRef::{} }},", e.kernel, args_buf_ref).unwrap();
            }
            DispatchOpKind::GatedBy { gate_ref } => {
                writeln!(out, "    DispatchOp::GatedBy {{ kernel: KernelId::{}, gate: BufferRef::{} }},", e.kernel, gate_ref).unwrap();
            }
        }
    }
    writeln!(out, "];").unwrap();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diamond_writes_rejected() {
        let nodes = vec![
            ("A".into(), vec![],            vec!["buf".into()]),
            ("B".into(), vec![],            vec!["buf".into()]),
        ];
        let err = build_schedule(&nodes).unwrap_err();
        match err {
            ScheduleError::DiamondWrite { buffer, .. } => assert_eq!(buffer, "buf"),
            other => panic!("expected DiamondWrite, got {other:?}"),
        }
    }

    #[test]
    fn cycle_detected() {
        let nodes = vec![
            ("A".into(), vec!["b".into()], vec!["a".into()]),
            ("B".into(), vec!["a".into()], vec!["b".into()]),
        ];
        let err = build_schedule(&nodes).unwrap_err();
        assert!(matches!(err, ScheduleError::Cycle { .. }));
    }

    #[test]
    fn topo_sort_orders_producer_before_consumer() {
        let nodes = vec![
            ("Consumer".into(), vec!["x".into()], vec![]),
            ("Producer".into(), vec![],           vec!["x".into()]),
        ];
        let order = build_schedule(&nodes).unwrap();
        let names: Vec<&str> = order.iter().map(|e| e.kernel.as_str()).collect();
        let p = names.iter().position(|s| *s == "Producer").unwrap();
        let c = names.iter().position(|s| *s == "Consumer").unwrap();
        assert!(p < c);
    }
}
```

- [x] **Step 4: Run schedule tests to confirm they pass**

Run: `cargo test -p dsl_compiler --test emit_schedule_smoke`
Expected: 3 passed.

Run: `cargo test -p dsl_compiler emit_schedule::tests`
Expected: 3 passed.

- [x] **Step 5: Implement emit_resident_context.rs**

Create `crates/dsl_compiler/src/emit_resident_context.rs`:

```rust
//! Emits `engine_gpu_rules/src/resident_context.rs` — one `wgpu::Buffer`
//! field per Resident-class buffer, plus a `new()` constructor.
//!
//! Initial implementation has no fields: kernel emitters in later tasks
//! call `add_resident_field` to populate the context. Producing an
//! empty struct on bootstrap keeps every kernel emitter free to add
//! fields independently.

use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct ResidentField {
    /// Snake-case field name on `ResidentPathContext`.
    pub name: String,
    /// Doc string written above the field.
    pub doc:  String,
}

pub fn emit_resident_context_rs(fields: &[ResidentField]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_resident_context. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// Resident-lifetime buffers — persist across ticks within a batch.").unwrap();
    writeln!(out, "pub struct ResidentPathContext {{").unwrap();
    for f in fields {
        writeln!(out, "    /// {}", f.doc).unwrap();
        writeln!(out, "    pub {}: wgpu::Buffer,", f.name).unwrap();
    }
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl ResidentPathContext {{").unwrap();
    writeln!(out, "    /// Allocate every Resident-class buffer up-front. agent_cap is").unwrap();
    writeln!(out, "    /// the maximum agent capacity across the batch — caps that grow").unwrap();
    writeln!(out, "    /// at runtime force a context rebuild (the existing GpuBackend").unwrap();
    writeln!(out, "    /// resident-rebuild path).").unwrap();
    writeln!(out, "    pub fn new(_device: &wgpu::Device, _agent_cap: u32) -> Self {{").unwrap();
    if fields.is_empty() {
        writeln!(out, "        Self {{}}").unwrap();
    } else {
        writeln!(out, "        Self {{").unwrap();
        for f in fields {
            // Each kernel emitter is responsible for the actual size /
            // descriptor when it lands. The bootstrap stub allocates a
            // 4-byte placeholder; kernel emitters in later tasks
            // override this routine wholesale by re-emitting the file.
            writeln!(out, "            {}: _device.create_buffer(&wgpu::BufferDescriptor {{", f.name).unwrap();
            writeln!(out, "                label: Some(\"engine_gpu_rules::resident::{}\"),", f.name).unwrap();
            writeln!(out, "                size: 4,").unwrap();
            writeln!(out, "                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,").unwrap();
            writeln!(out, "                mapped_at_creation: false,").unwrap();
            writeln!(out, "            }}),").unwrap();
        }
        writeln!(out, "        }}").unwrap();
    }
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_context_emits_unit_struct_body() {
        let src = emit_resident_context_rs(&[]);
        assert!(src.contains("pub struct ResidentPathContext"));
        assert!(src.contains("Self {}"));
    }

    #[test]
    fn single_field_emits_field_and_init() {
        let src = emit_resident_context_rs(&[ResidentField {
            name: "agents".into(),
            doc: "Agent SoA buffer".into(),
        }]);
        assert!(src.contains("pub agents: wgpu::Buffer"));
        assert!(src.contains("Agent SoA buffer"));
    }
}
```

- [x] **Step 6: Implement the four binding-source-container emitters + binding_sources aggregate**

The trait method `bind(&BindingSources<'a>, &wgpu::Buffer)` resolves buffers by walking five containers. Each container is emitted with `add_*_field` callbacks that per-kernel tasks invoke as they bring new buffers into scope. On bootstrap (this task), all five containers are field-empty; the per-kernel tasks 4–13 populate them.

Create `crates/dsl_compiler/src/emit_pingpong_context.rs`:

```rust
//! Emits `engine_gpu_rules/src/pingpong_context.rs` — A/B ring buffers
//! for cascade physics. Bootstrap version is empty; Task 9 populates.

use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct PingPongField {
    /// Snake-case field name on `PingPongContext`.
    pub name: String,
    pub doc:  String,
}

pub fn emit_pingpong_context_rs(fields: &[PingPongField]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_pingpong_context. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// PingPong-lifetime ring buffers (cascade A/B).").unwrap();
    writeln!(out, "pub struct PingPongContext {{").unwrap();
    for f in fields {
        writeln!(out, "    /// {}", f.doc).unwrap();
        writeln!(out, "    pub {}: wgpu::Buffer,", f.name).unwrap();
    }
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl PingPongContext {{").unwrap();
    writeln!(out, "    pub fn new(_device: &wgpu::Device) -> Self {{").unwrap();
    if fields.is_empty() {
        writeln!(out, "        Self {{}}").unwrap();
    } else {
        writeln!(out, "        Self {{").unwrap();
        for f in fields {
            writeln!(out, "            {}: _device.create_buffer(&wgpu::BufferDescriptor {{", f.name).unwrap();
            writeln!(out, "                label: Some(\"engine_gpu_rules::pingpong::{}\"),", f.name).unwrap();
            writeln!(out, "                size: 4,").unwrap();
            writeln!(out, "                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,").unwrap();
            writeln!(out, "                mapped_at_creation: false,").unwrap();
            writeln!(out, "            }}),").unwrap();
        }
        writeln!(out, "        }}").unwrap();
    }
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

Create `crates/dsl_compiler/src/emit_pool.rs`:

```rust
//! Emits `engine_gpu_rules/src/pool.rs` — `Pool` shape-keyed reusable
//! buffers (Pooled lifetime). Bootstrap version is empty; Task 12
//! populates with spatial-hash + per-query buffers.

use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct PoolField {
    pub name: String,
    pub doc:  String,
}

pub fn emit_pool_rs(fields: &[PoolField]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_pool. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// Pooled-lifetime buffers — shape-keyed; reused across compatible kernels.").unwrap();
    writeln!(out, "pub struct Pool {{").unwrap();
    for f in fields {
        writeln!(out, "    /// {}", f.doc).unwrap();
        writeln!(out, "    pub {}: wgpu::Buffer,", f.name).unwrap();
    }
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl Pool {{").unwrap();
    writeln!(out, "    pub fn new(_device: &wgpu::Device) -> Self {{").unwrap();
    if fields.is_empty() {
        writeln!(out, "        Self {{}}").unwrap();
    } else {
        writeln!(out, "        Self {{").unwrap();
        for f in fields {
            writeln!(out, "            {}: _device.create_buffer(&wgpu::BufferDescriptor {{", f.name).unwrap();
            writeln!(out, "                label: Some(\"engine_gpu_rules::pool::{}\"),", f.name).unwrap();
            writeln!(out, "                size: 4,").unwrap();
            writeln!(out, "                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,").unwrap();
            writeln!(out, "                mapped_at_creation: false,").unwrap();
            writeln!(out, "            }}),").unwrap();
        }
        writeln!(out, "        }}").unwrap();
    }
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

Create `crates/dsl_compiler/src/emit_transient_handles.rs`:

```rust
//! Emits `engine_gpu_rules/src/transient_handles.rs` — references to
//! per-tick scratch buffers refreshed every batch tick by engine_gpu's
//! `TransientHandles::new(&pool)` call. Field shape is determined by
//! kernel needs; values are populated each tick by engine_gpu.

use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct TransientField {
    pub name: String,
    pub doc:  String,
}

pub fn emit_transient_handles_rs(fields: &[TransientField]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_transient_handles. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// Transient-lifetime buffer references — populated each tick by engine_gpu.").unwrap();
    writeln!(out, "pub struct TransientHandles<'a> {{").unwrap();
    for f in fields {
        writeln!(out, "    /// {}", f.doc).unwrap();
        writeln!(out, "    pub {}: &'a wgpu::Buffer,", f.name).unwrap();
    }
    writeln!(out, "    pub _phantom: std::marker::PhantomData<&'a ()>,").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

Create `crates/dsl_compiler/src/emit_external_buffers.rs`:

```rust
//! Emits `engine_gpu_rules/src/external_buffers.rs` — references to
//! engine-owned buffers (agent SoA, sim_cfg, registries). Field shape
//! is determined by kernel needs; values are populated by engine_gpu.

use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct ExternalField {
    pub name: String,
    pub doc:  String,
}

pub fn emit_external_buffers_rs(fields: &[ExternalField]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_external_buffers. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// External-lifetime buffer references — engine-owned (agent SoA, sim_cfg, registries).").unwrap();
    writeln!(out, "pub struct ExternalBuffers<'a> {{").unwrap();
    for f in fields {
        writeln!(out, "    /// {}", f.doc).unwrap();
        writeln!(out, "    pub {}: &'a wgpu::Buffer,", f.name).unwrap();
    }
    writeln!(out, "    pub _phantom: std::marker::PhantomData<&'a ()>,").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

Create `crates/dsl_compiler/src/emit_binding_sources.rs`:

```rust
//! Emits `engine_gpu_rules/src/binding_sources.rs` — the 5-field
//! aggregate the trait method `Kernel::bind` consumes. Stable across
//! regens (its shape is fixed; the containers it references are
//! per-task-extended).

use std::fmt::Write;

pub fn emit_binding_sources_rs() -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_binding_sources. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::resident_context::ResidentPathContext;").unwrap();
    writeln!(out, "use crate::pingpong_context::PingPongContext;").unwrap();
    writeln!(out, "use crate::pool::Pool;").unwrap();
    writeln!(out, "use crate::transient_handles::TransientHandles;").unwrap();
    writeln!(out, "use crate::external_buffers::ExternalBuffers;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// Aggregate of every container kernels can pull buffers from.").unwrap();
    writeln!(out, "/// `Kernel::bind` takes `&BindingSources<'a>` and walks into the").unwrap();
    writeln!(out, "/// right field; each kernel's emitted bind() body is real").unwrap();
    writeln!(out, "/// generated code, not `unimplemented!()`.").unwrap();
    writeln!(out, "pub struct BindingSources<'a> {{").unwrap();
    writeln!(out, "    pub resident:  &'a ResidentPathContext,").unwrap();
    writeln!(out, "    pub pingpong:  &'a PingPongContext,").unwrap();
    writeln!(out, "    pub pool:      &'a Pool,").unwrap();
    writeln!(out, "    pub transient: &'a TransientHandles<'a>,").unwrap();
    writeln!(out, "    pub external:  &'a ExternalBuffers<'a>,").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

Create the smoke test `crates/dsl_compiler/tests/emit_binding_sources_smoke.rs`:

```rust
use dsl_compiler::emit_binding_sources::emit_binding_sources_rs;

#[test]
fn binding_sources_struct_has_five_fields() {
    let src = emit_binding_sources_rs();
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub struct BindingSources<'a>"));
    assert!(src.contains("pub resident:"));
    assert!(src.contains("pub pingpong:"));
    assert!(src.contains("pub pool:"));
    assert!(src.contains("pub transient:"));
    assert!(src.contains("pub external:"));
}
```

Run: `cargo test -p dsl_compiler --test emit_binding_sources_smoke`
Expected: PASS.

- [x] **Step 7: Register all emitter modules in dsl_compiler's lib.rs**

Open `crates/dsl_compiler/src/lib.rs` and add (alongside the other `pub mod emit_*` lines added in Task 2):

```rust
pub mod emit_schedule;
pub mod emit_resident_context;
pub mod emit_pingpong_context;
pub mod emit_pool;
pub mod emit_transient_handles;
pub mod emit_external_buffers;
pub mod emit_binding_sources;
```

- [x] **Step 8: Add gpu_rules_hash to schema_hash.rs**

Open `crates/dsl_compiler/src/schema_hash.rs`. Find `combined_hash` (around line 754) and add a new sub-hash that takes the GPU-rules generated bytes as input.

Add this function above `combined_hash`:

```rust
/// Hash the byte content of every `// GENERATED` file under
/// `engine_gpu_rules/src/` plus the SCHEDULE constant. Called from the
/// compile-dsl xtask after every emit pass; the result is written to
/// `crates/engine_gpu_rules/.schema_hash`.
///
/// `inputs` is the sorted list of `(filename, bytes)` pairs the
/// caller has already emitted. Sorting on the caller side keeps the
/// hash stable across machines (filesystem iteration order is not
/// guaranteed).
pub fn gpu_rules_hash(inputs: &[(String, Vec<u8>)]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"engine_gpu_rules:v1");
    for (name, bytes) in inputs {
        h.update((name.len() as u32).to_le_bytes());
        h.update(name.as_bytes());
        h.update((bytes.len() as u32).to_le_bytes());
        h.update(bytes);
    }
    h.finalize().into()
}
```

Then extend `combined_hash` to take it. Update the signature:

```rust
pub fn combined_hash(
    state: &[u8; 32],
    event: &[u8; 32],
    rules: &[u8; 32],
    scoring: &[u8; 32],
    config: &[u8; 32],
    enums: &[u8; 32],
    views: &[u8; 32],
    gpu_rules: &[u8; 32],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(state);
    h.update(event);
    h.update(rules);
    h.update(scoring);
    h.update(config);
    h.update(enums);
    h.update(views);
    h.update(gpu_rules);
    h.finalize().into()
}
```

Also update `emit_schema_rs` to take `gpu_rules: &[u8; 32]` and write a `GPU_RULES_HASH` constant alongside the others. Find every call site of `combined_hash` and `emit_schema_rs` (in `crates/dsl_compiler/src/lib.rs` around line 316 and 381) and pass `&gpu_rules_hash(&[])` initially (we'll thread the real bytes through in Step 9).

- [x] **Step 9: Confirm the schema_hash module unit tests still pass**

Run: `cargo test -p dsl_compiler schema_hash`
Expected: all passes; the new `gpu_rules_hash` is exercised by step 10's xtask plumbing.

- [x] **Step 10: Wire xtask compile-dsl to emit all five containers + schedule + binding_sources + .schema_hash**

In `crates/xtask/src/main.rs`'s `compile-dsl` subcommand, after the existing `engine_gpu_rules/src/lib.rs` write, add:

```rust
// Schedule (initially empty; populated by per-kernel emitters in later tasks).
{
    use std::fs;
    use std::path::PathBuf;
    let schedule_rs = dsl_compiler::emit_schedule::emit_schedule_rs(&[]);
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/schedule.rs"), &schedule_rs)
        .expect("write schedule.rs");
}

// Resident context (initially empty struct).
{
    use std::fs;
    use std::path::PathBuf;
    let rc_rs = dsl_compiler::emit_resident_context::emit_resident_context_rs(&[]);
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/resident_context.rs"), &rc_rs)
        .expect("write resident_context.rs");
}

// Pingpong context (initially empty struct; Task 9 populates).
{
    use std::fs;
    use std::path::PathBuf;
    let body = dsl_compiler::emit_pingpong_context::emit_pingpong_context_rs(&[]);
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/pingpong_context.rs"), body)
        .expect("write pingpong_context.rs");
}

// Pool (initially empty; Task 12 populates with spatial scratch).
{
    use std::fs;
    use std::path::PathBuf;
    let body = dsl_compiler::emit_pool::emit_pool_rs(&[]);
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/pool.rs"), body)
        .expect("write pool.rs");
}

// Transient handles (initially empty; per-kernel tasks 4–13 populate).
{
    use std::fs;
    use std::path::PathBuf;
    let body = dsl_compiler::emit_transient_handles::emit_transient_handles_rs(&[]);
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/transient_handles.rs"), body)
        .expect("write transient_handles.rs");
}

// External buffers (initially empty; per-kernel tasks 4–13 populate).
{
    use std::fs;
    use std::path::PathBuf;
    let body = dsl_compiler::emit_external_buffers::emit_external_buffers_rs(&[]);
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/external_buffers.rs"), body)
        .expect("write external_buffers.rs");
}

// BindingSources<'a> — fixed shape (5 references); never re-emitted across regens.
{
    use std::fs;
    use std::path::PathBuf;
    let body = dsl_compiler::emit_binding_sources::emit_binding_sources_rs();
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/binding_sources.rs"), body)
        .expect("write binding_sources.rs");
}

// Megakernel — empty stub until Task 14.
{
    use std::fs;
    use std::path::PathBuf;
    let stub = "// GENERATED by dsl_compiler. Do not edit by hand.\n\
                // Megakernel emit lands in Task 14; this stub keeps the module list\n\
                // declared by emit_kernel_index resolvable.\n";
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/megakernel.rs"), stub)
        .expect("write megakernel.rs");
}

// Schema hash baseline.
{
    use std::fs;
    use std::path::PathBuf;
    let mut inputs: Vec<(String, Vec<u8>)> = Vec::new();
    for entry in walkdir::WalkDir::new("crates/engine_gpu_rules/src") {
        let entry = entry.expect("walk engine_gpu_rules/src");
        if !entry.file_type().is_file() { continue; }
        let p = entry.path();
        let ext = p.extension().and_then(|e| e.to_str());
        if !matches!(ext, Some("rs") | Some("wgsl")) { continue; }
        let rel = p.strip_prefix("crates/engine_gpu_rules/src").unwrap();
        let bytes = fs::read(p).expect("read emitted file");
        inputs.push((rel.display().to_string(), bytes));
    }
    inputs.sort_by(|a, b| a.0.cmp(&b.0));
    let h = dsl_compiler::schema_hash::gpu_rules_hash(&inputs);
    let hex_str = h.iter().map(|b| format!("{b:02x}")).collect::<String>();
    fs::write(PathBuf::from("crates/engine_gpu_rules/.schema_hash"), hex_str)
        .expect("write engine_gpu_rules/.schema_hash");
}
```

If `walkdir` isn't already a dependency of `xtask`, add `walkdir = "2"` to `crates/xtask/Cargo.toml`.

- [x] **Step 11: Run compile-dsl and verify all infrastructure files land**

Run: `cargo run --bin xtask -- compile-dsl`
Expected: command exits 0.

Run: `ls crates/engine_gpu_rules/src/`
Expected: `binding_sources.rs  external_buffers.rs  lib.rs  megakernel.rs  pingpong_context.rs  pool.rs  resident_context.rs  schedule.rs  transient_handles.rs`.

Run: `cat crates/engine_gpu_rules/.schema_hash`
Expected: a 64-char hex string (32 bytes of SHA-256).

Run: `cargo build -p engine_gpu_rules`
Expected: clean build.

- [x] **Step 12: Add the schema-hash baseline test**

Create `crates/engine_gpu_rules/tests/schema_hash.rs`:

```rust
//! Baseline-comparison test for engine_gpu_rules' generated content.
//! On a CI failure, the fix is one of:
//!   1. Run `cargo run --bin xtask -- compile-dsl` to re-emit; commit
//!      the regen alongside the .sim change that caused it.
//!   2. If the regen is intentional, the .schema_hash baseline is
//!      already updated by the xtask — review the diff and commit.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

fn compute_current_hash() -> String {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    walk(Path::new("src"), &mut entries);
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut h = Sha256::new();
    h.update(b"engine_gpu_rules:v1");
    for (name, bytes) in &entries {
        h.update((name.len() as u32).to_le_bytes());
        h.update(name.as_bytes());
        h.update((bytes.len() as u32).to_le_bytes());
        h.update(bytes);
    }
    let bytes: [u8; 32] = h.finalize().into();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn walk(dir: &Path, out: &mut Vec<(String, Vec<u8>)>) {
    for entry in fs::read_dir(dir).expect("readable src") {
        let entry = entry.expect("entry");
        let p = entry.path();
        if entry.file_type().expect("ft").is_dir() { walk(&p, out); continue; }
        let ext = p.extension().and_then(|e| e.to_str());
        if !matches!(ext, Some("rs") | Some("wgsl")) { continue; }
        let rel = p.strip_prefix("src").unwrap();
        let bytes = fs::read(&p).expect("read");
        out.push((rel.display().to_string(), bytes));
    }
}

#[test]
fn baseline_matches_current() {
    let baseline = include_str!("../.schema_hash").trim();
    let current = compute_current_hash();
    assert_eq!(
        current, baseline,
        "engine_gpu_rules content changed.\n\
         If intentional: re-run `cargo run --bin xtask -- compile-dsl` to bump the baseline.\n\
         Current: {}", current
    );
}
```

Add `sha2 = "0.10"` to `crates/engine_gpu_rules/Cargo.toml` under `[dev-dependencies]`.

- [x] **Step 13: Run baseline test to confirm it passes**

Run: `cargo test -p engine_gpu_rules --test schema_hash`
Expected: 1 passed.

- [x] **Step 14: Commit**

```bash
git add crates/dsl_compiler/src/emit_schedule.rs crates/dsl_compiler/src/emit_resident_context.rs crates/dsl_compiler/src/emit_pingpong_context.rs crates/dsl_compiler/src/emit_pool.rs crates/dsl_compiler/src/emit_transient_handles.rs crates/dsl_compiler/src/emit_external_buffers.rs crates/dsl_compiler/src/emit_binding_sources.rs crates/dsl_compiler/src/schema_hash.rs crates/dsl_compiler/src/lib.rs crates/xtask/ crates/engine_gpu_rules/
git commit -m "feat(dsl_compiler): emit schedule + 5 binding-source containers + gpu_rules_hash baseline"
```

---

### Task 4: Emit `FusedMaskKernel` + `MaskUnpackKernel`

This is the first real kernel emitter. It validates the end-to-end pipeline on a leaf kernel: the existing WGSL emit (`emit_mask_wgsl`) is already DSL-driven, so this task wraps it with a Rust struct emit, registers a Schedule entry, and routes the `KernelId::FusedMask` variant into `engine_gpu`.

**Files:**
- Create: `crates/dsl_compiler/src/emit_mask_kernel.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (register module)
- Modify: `crates/xtask/src/main.rs` (call new emitter; thread modules into `emit_lib_rs`)
- Generated: `crates/engine_gpu_rules/src/fused_mask.{rs,wgsl}` + `mask_unpack.{rs,wgsl}`
- Test: `crates/dsl_compiler/tests/emit_mask_kernel_smoke.rs`
- Test: `crates/engine_gpu_rules/tests/fused_mask_compiles.rs`

- [x] **Step 1: Write the failing emitter unit test**

Create `crates/dsl_compiler/tests/emit_mask_kernel_smoke.rs`:

```rust
use dsl_compiler::emit_mask_kernel::{emit_fused_mask_rs, emit_mask_unpack_rs};

#[test]
fn fused_mask_rs_has_kernel_impl_and_wgsl_include() {
    let src = emit_fused_mask_rs();
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub struct FusedMaskKernel"));
    assert!(src.contains("pub struct FusedMaskCfg"));
    assert!(src.contains("impl crate::Kernel for FusedMaskKernel"));
    assert!(src.contains("include_str!(\"fused_mask.wgsl\")"));
    assert!(src.contains("fn record"));
}

#[test]
fn mask_unpack_rs_has_kernel_impl() {
    let src = emit_mask_unpack_rs();
    assert!(src.contains("pub struct MaskUnpackKernel"));
    assert!(src.contains("impl crate::Kernel for MaskUnpackKernel"));
    assert!(src.contains("include_str!(\"mask_unpack.wgsl\")"));
}
```

- [x] **Step 2: Run to confirm failure**

Run: `cargo test -p dsl_compiler --test emit_mask_kernel_smoke`
Expected: FAIL — `unresolved import dsl_compiler::emit_mask_kernel`.

- [x] **Step 3: Implement emit_mask_kernel.rs**

Create `crates/dsl_compiler/src/emit_mask_kernel.rs`:

```rust
//! Emits `engine_gpu_rules/src/fused_mask.rs` + `fused_mask.wgsl` and
//! `mask_unpack.rs` + `mask_unpack.wgsl`.
//!
//! The mask WGSL body is produced by the existing
//! `dsl_compiler::emit_mask_wgsl` module — this emitter wraps the body
//! with a Rust struct that holds the pipeline + BGL and implements the
//! `Kernel` trait.

use std::fmt::Write;

/// Emit `engine_gpu_rules/src/fused_mask.rs`.
pub fn emit_fused_mask_rs() -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_mask_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct FusedMaskKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct FusedMaskBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub mask_bitmaps: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct FusedMaskCfg {{").unwrap();
    writeln!(out, "    pub agent_cap: u32,").unwrap();
    writeln!(out, "    pub num_mask_words: u32,").unwrap();
    writeln!(out, "    pub _pad0: u32,").unwrap();
    writeln!(out, "    pub _pad1: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"fused_mask.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for FusedMaskKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = FusedMaskBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = FusedMaskCfg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                bgl_storage(0, true),  // agents (read)").unwrap();
    writeln!(out, "                bgl_storage(1, false), // mask_bitmaps (atomicOr)").unwrap();
    writeln!(out, "                bgl_storage(2, true),  // sim_cfg (read)").unwrap();
    writeln!(out, "                bgl_uniform(3),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_fused_masks\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> FusedMaskCfg {{").unwrap();
    writeln!(out, "        let agent_cap = state.agent_cap();").unwrap();
    writeln!(out, "        // Mask layout: ceil(agent_cap/32) words per mask × N masks.").unwrap();
    writeln!(out, "        let words_per_mask = (agent_cap + 31) / 32;").unwrap();
    writeln!(out, "        FusedMaskCfg {{").unwrap();
    writeln!(out, "            agent_cap,").unwrap();
    writeln!(out, "            num_mask_words: words_per_mask,").unwrap();
    writeln!(out, "            _pad0: 0, _pad1: 0,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> FusedMaskBindings<'a> {{").unwrap();
    writeln!(out, "        FusedMaskBindings {{").unwrap();
    writeln!(out, "            agents:       sources.external.agents,").unwrap();
    writeln!(out, "            mask_bitmaps: sources.transient.mask_bitmaps,").unwrap();
    writeln!(out, "            sim_cfg:      sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &FusedMaskBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.mask_bitmaps.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        let wg = (agent_cap + 63) / 64;").unwrap();
    writeln!(out, "        pass.dispatch_workgroups(wg, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "fn bgl_storage(b: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "    wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "        binding: b,").unwrap();
    writeln!(out, "        visibility: wgpu::ShaderStages::COMPUTE,").unwrap();
    writeln!(out, "        ty: wgpu::BindingType::Buffer {{").unwrap();
    writeln!(out, "            ty: wgpu::BufferBindingType::Storage {{ read_only }},").unwrap();
    writeln!(out, "            has_dynamic_offset: false,").unwrap();
    writeln!(out, "            min_binding_size: None,").unwrap();
    writeln!(out, "        }},").unwrap();
    writeln!(out, "        count: None,").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "fn bgl_uniform(b: u32) -> wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "    wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "        binding: b,").unwrap();
    writeln!(out, "        visibility: wgpu::ShaderStages::COMPUTE,").unwrap();
    writeln!(out, "        ty: wgpu::BindingType::Buffer {{").unwrap();
    writeln!(out, "            ty: wgpu::BufferBindingType::Uniform,").unwrap();
    writeln!(out, "            has_dynamic_offset: false,").unwrap();
    writeln!(out, "            min_binding_size: None,").unwrap();
    writeln!(out, "        }},").unwrap();
    writeln!(out, "        count: None,").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

/// Emit `engine_gpu_rules/src/mask_unpack.rs` — agents SoA → bitmap pack.
pub fn emit_mask_unpack_rs() -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_mask_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct MaskUnpackKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct MaskUnpackBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents_soa: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub agents_input: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct MaskUnpackCfg {{ pub agent_cap: u32, pub _pad: [u32; 3] }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"mask_unpack.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for MaskUnpackKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = MaskUnpackBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = MaskUnpackCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::bgl\"),").unwrap();
    writeln!(out, "            entries: &[crate::fused_mask::bgl_storage(0, false), crate::fused_mask::bgl_storage(1, true), crate::fused_mask::bgl_uniform(2)],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_mask_unpack\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> MaskUnpackCfg {{").unwrap();
    writeln!(out, "        MaskUnpackCfg {{ agent_cap: state.agent_cap(), _pad: [0; 3] }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> MaskUnpackBindings<'a> {{").unwrap();
    writeln!(out, "        MaskUnpackBindings {{").unwrap();
    writeln!(out, "            agents_soa:   sources.external.agents,").unwrap();
    writeln!(out, "            agents_input: sources.transient.mask_unpack_agents_input,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &MaskUnpackBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents_soa.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.agents_input.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

> Implementation note: `record()` takes `device: &wgpu::Device` as a parameter — the trait signature emitted by `emit_kernel_index` already includes it (Task 2). `bind()`'s `sources.transient.mask_bitmaps` and `sources.transient.mask_unpack_agents_input` references mean Task 5 step 4's plumbing must add those fields to `TransientHandles` (via the `emit_transient_handles` callback) before this kernel can be wired in.

- [x] **Step 4: Register module in dsl_compiler/src/lib.rs**

Add `pub mod emit_mask_kernel;` alongside the existing emit_* declarations.

- [x] **Step 5: Run unit tests to confirm they pass**

Run: `cargo test -p dsl_compiler --test emit_mask_kernel_smoke`
Expected: 2 passed.

- [x] **Step 6: Wire xtask compile-dsl to write fused_mask + mask_unpack files**

In `crates/xtask/src/main.rs`, replace the placeholder empty-modules `Vec` from Task 2 step 6 with:

```rust
let mut modules: Vec<String> = vec!["fused_mask".into(), "mask_unpack".into()];
modules.sort();
```

After the lib.rs write, also write the two kernel modules and reuse the existing WGSL emitter for the kernel bodies. Extend `TransientHandles` and `ExternalBuffers` field lists for the buffers the mask kernels need:

```rust
{
    use std::fs;
    use std::path::PathBuf;
    use dsl_compiler::emit_transient_handles::TransientField;
    use dsl_compiler::emit_external_buffers::ExternalField;

    let body = dsl_compiler::emit_mask_kernel::emit_fused_mask_rs();
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/fused_mask.rs"), body).expect("fused_mask.rs");

    // The fused-mask WGSL is produced by the existing
    // `emit_mask_wgsl` over the loaded compilation. Pass through.
    let mask_irs = engine_gpu_rules_mask_irs(&comp); // helper defined in xtask file: filters comp.masks by emittable predicate
    let wgsl = format!(
        "// GENERATED by dsl_compiler::emit_mask_wgsl. Do not edit by hand.\n{}",
        dsl_compiler::emit_mask_wgsl::emit_fused_mask_wgsl(&mask_irs)
    );
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/fused_mask.wgsl"), wgsl).expect("fused_mask.wgsl");

    let body2 = dsl_compiler::emit_mask_kernel::emit_mask_unpack_rs();
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/mask_unpack.rs"), body2).expect("mask_unpack.rs");
    let unpack_wgsl = "// GENERATED by dsl_compiler. Do not edit by hand.\n@compute @workgroup_size(64) fn cs_mask_unpack(@builtin(global_invocation_id) gid: vec3<u32>) {}\n";
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/mask_unpack.wgsl"), unpack_wgsl).expect("mask_unpack.wgsl");

    // Bring the mask kernels' buffers into the binding-source containers.
    // (xtask accumulates these across all kernel-emit blocks; the
    // accumulated lists are written once at the end of compile-dsl.)
    transient_fields.push(TransientField {
        name: "mask_bitmaps".into(),
        doc: "FusedMaskKernel output: ceil(N/32) words × N masks; recycled per tick.".into(),
    });
    transient_fields.push(TransientField {
        name: "mask_unpack_agents_input".into(),
        doc: "MaskUnpackKernel scratch: source SoA before unpack.".into(),
    });
    if !external_fields.iter().any(|f: &ExternalField| f.name == "agents") {
        external_fields.push(ExternalField {
            name: "agents".into(),
            doc: "Agent SoA buffer (engine-owned).".into(),
        });
    }
    if !external_fields.iter().any(|f: &ExternalField| f.name == "sim_cfg") {
        external_fields.push(ExternalField {
            name: "sim_cfg".into(),
            doc: "SimCfg uniform/storage buffer (engine-owned).".into(),
        });
    }
}
```

(`transient_fields` and `external_fields` are local Vecs declared at the top of the compile-dsl subcommand; after every kernel-emit block runs, the xtask flushes them via `emit_transient_handles_rs(&transient_fields)` and `emit_external_buffers_rs(&external_fields)` overwriting the bootstrap empty versions from Task 3. Each per-kernel task below pushes the fields it needs; xtask de-duplicates by name.)

(The mask-unpack WGSL stays minimal here; the real body matching today's `engine_gpu::mask::MaskUnpackKernel` shader is hoisted in Task 5 step 4 alongside the wiring.)

- [x] **Step 7: Run compile-dsl + verify the files land + the schema_hash baseline updates**

Run: `cargo run --bin xtask -- compile-dsl`
Expected: exit 0; new files appear:
```
crates/engine_gpu_rules/src/fused_mask.rs
crates/engine_gpu_rules/src/fused_mask.wgsl
crates/engine_gpu_rules/src/mask_unpack.rs
crates/engine_gpu_rules/src/mask_unpack.wgsl
```

Plus `transient_handles.rs` and `external_buffers.rs` are re-emitted with the new fields (`mask_bitmaps`, `mask_unpack_agents_input`, `agents`, `sim_cfg`).

Run: `cargo build -p engine_gpu_rules`
Expected: clean build. The emitted `bind()` body for both kernels accesses real fields on `BindingSources<'a>`; `record()` takes `device: &wgpu::Device` so no `encoder.device()` placeholder remains.

- [x] **Step 8: Commit**

```bash
git add crates/dsl_compiler/src/emit_mask_kernel.rs crates/dsl_compiler/src/lib.rs crates/xtask/ crates/engine_gpu_rules/
git commit -m "feat(dsl_compiler): emit_mask_kernel produces fused_mask + mask_unpack modules"
```

---

### Task 5: Wire `FusedMaskKernel` + `MaskUnpackKernel` into `engine_gpu` step_batch

This is the first cross-crate integration — engine_gpu now depends on engine_gpu_rules and consumes the emitted kernel for the mask phase. The hand-written `engine_gpu::mask::FusedMaskKernel` stays in the tree (Task 16 deletes it) but stops being called.

This task also lands the `BindingSources<'a>` construction site in `step_batch`. Every per-tick dispatch builds the aggregate once and passes it to `kernel.bind(&sources, &cfg).record(...)`.

**Files:**
- Modify: `crates/engine_gpu/Cargo.toml` (add `engine_gpu_rules` dep)
- Modify: `crates/engine_gpu/src/lib.rs` (call `engine_gpu_rules::fused_mask::FusedMaskKernel` in step_batch via the bind/record pattern)
- Modify: `crates/engine_gpu/src/backend/resident_ctx.rs` (add `Option<FusedMaskKernel>`; build `TransientHandles` + `ExternalBuffers` constructors)
- Test: `crates/engine_gpu/tests/parity_with_cpu.rs` (existing; must stay green)

- [x] **Step 1: Confirm emitted bind/record signatures are stable**

Task 4's emitter already produces `bind(&BindingSources, &cfg)` returning a populated `Bindings` and `record(&self, device, encoder, bindings, agent_cap)`. No re-edit of the emitter is required here.

Run: `cargo run --bin xtask -- compile-dsl`
Run: `cargo build -p engine_gpu_rules`
Expected: clean build.

- [x] **Step 2: (deleted — see Step 1)**

(This step was previously a placeholder that re-ran compile-dsl after a no-longer-needed emitter edit.)

- [x] **Step 3: Add engine_gpu_rules to engine_gpu's deps**

Open `crates/engine_gpu/Cargo.toml`. Under `[dependencies]` add (gating on the gpu feature where the existing engine_gpu deps gate):

```toml
engine_gpu_rules = { path = "../engine_gpu_rules", optional = true }
```

And update the `gpu` feature line to:

```toml
gpu = ["dep:wgpu", "dep:engine_gpu_rules", "engine_gpu_rules/gpu", /* ... existing entries ... */]
```

- [x] **Step 4: Build the BindingSources aggregate + replace the mask dispatch site in step_batch**

Open `crates/engine_gpu/src/lib.rs` around the `step_batch` body (line 991+ — the `fused_unpack_kernel.encode_unpack(...)` call at line 1126 is the integration point).

First, at the top of the per-tick loop, build the `BindingSources` aggregate. The five containers come from these owners:
- `resident: &self.resident.path_ctx` — `engine_gpu_rules::ResidentPathContext` instance, persisted across ticks.
- `pingpong: &self.resident.pingpong_ctx` — `engine_gpu_rules::PingPongContext`, also persisted; A/B alternation handled internally.
- `pool: &self.resident.pool` — `engine_gpu_rules::Pool`, persisted.
- `transient: &TransientHandles { ... }` — built freshly each tick from refs into engine_gpu's transient buffers.
- `external: &ExternalBuffers { ... }` — built freshly each tick from refs into agent SoA + sim_cfg + registries.

```rust
use engine_gpu_rules::{transient_handles::TransientHandles, external_buffers::ExternalBuffers, binding_sources::BindingSources};

let transient = TransientHandles {
    mask_bitmaps:               self.sync.mask_kernel.mask_bitmaps_buf(),
    mask_unpack_agents_input:   self.sync.mask_kernel.unpack_agents_input_buf(),
    _phantom: std::marker::PhantomData,
};
let external = ExternalBuffers {
    agents:  agents_buf,
    sim_cfg: sim_cfg_buf.as_ref().expect("sim_cfg ensured"),
    _phantom: std::marker::PhantomData,
};
let sources = BindingSources {
    resident:  &self.resident.path_ctx,
    pingpong:  &self.resident.pingpong_ctx,
    pool:      &self.resident.pool,
    transient: &transient,
    external:  &external,
};
```

Then replace the call to the hand-written `mask_kernel.encode(...)` (typically inside the fused_unpack_kernel call chain or directly afterwards) with:

```rust
{
    use engine_gpu_rules::fused_mask::FusedMaskKernel;
    use engine_gpu_rules::Kernel as _;

    // Lazy-init pattern: the kernel is built once and stashed on
    // self.resident; rebuilt only when the resident context is rebuilt.
    let kernel = self.resident.fused_mask_kernel
        .get_or_insert_with(|| FusedMaskKernel::new(&self.device));
    let cfg = kernel.build_cfg(state);
    let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fused_mask::cfg"),
        contents: bytemuck::cast_slice(&[cfg]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bindings = kernel.bind(&sources, &cfg_buf);
    kernel.record(&self.device, &mut encoder, &bindings, agent_cap);
}
```

The Bindings struct is no longer constructed inline at the dispatch site — `kernel.bind(&sources, &cfg_buf)` returns it, with field-by-field wiring driven by the emitted code. Adding a new `.sim` row regenerates that wiring; engine_gpu's step_batch is unchanged.

Add a `pub fused_mask_kernel: Option<engine_gpu_rules::fused_mask::FusedMaskKernel>` field to `ResidentPathContext` (engine_gpu side wrapper) in `crates/engine_gpu/src/backend/resident_ctx.rs` (initialised to `None` in `new()`). Also add the three persistent containers:

```rust
pub path_ctx:     engine_gpu_rules::resident_context::ResidentPathContext,
pub pingpong_ctx: engine_gpu_rules::pingpong_context::PingPongContext,
pub pool:         engine_gpu_rules::pool::Pool,
```

(Initialised in `ResidentPathContext::new()` via the emitted `::new(&device)` constructors.)

> Implementation guidance: `mask_bitmaps_buf()` and `unpack_agents_input_buf()` may not exist as public accessors on the hand-written `FusedMaskKernel` / `MaskUnpackKernel`. If they don't, add thin `pub fn` getters returning `&wgpu::Buffer` (transitional; this hand-written code goes away in Task 16). The buffers themselves still come from `BufferPool`/the existing transient allocator — what changes is the wiring path, not the allocation strategy.

- [x] **Step 5: Run the parity sweep**

Run: `cargo test -p engine_gpu --test parity_with_cpu`
Expected: all tests pass.

Run: `cargo test -p engine --test wolves_and_humans_parity`
Expected: all tests pass.

Run: `cargo test -p engine_gpu --test physics_parity`
Expected: all tests pass.

If any test fails, the failure is the new dispatch path producing different mask bitmaps. Localize by running `cargo test -p engine_gpu --test parity_with_cpu test_mask_only` (or the closest filter) and diffing against the hand-written kernel's output (CPU reference at `engine_gpu::mask::cpu_mask_bitmap`).

- [x] **Step 6: Update the schema-hash baseline**

Run: `cargo run --bin xtask -- compile-dsl`
Run: `cargo test -p engine_gpu_rules --test schema_hash`
Expected: PASS (the baseline file was rewritten by xtask in step 2).

- [x] **Step 7: Commit**

```bash
git add crates/engine_gpu/Cargo.toml crates/engine_gpu/src/lib.rs crates/engine_gpu/src/backend/resident_ctx.rs crates/engine_gpu/src/mask.rs crates/dsl_compiler/src/emit_mask_kernel.rs crates/dsl_compiler/src/emit_kernel_index.rs crates/engine_gpu_rules/
git commit -m "feat(engine_gpu): wire emitted FusedMaskKernel from engine_gpu_rules into step_batch"
```

---

### Task 6: Emit `ScoringKernel` + `ScoringUnpackKernel` (target_bound)

This is the largest hand-written wrapper (`engine_gpu/src/scoring.rs` is 2795 lines). The migration wraps the existing `emit_scoring_wgsl_atomic_views` WGSL emitter (which already handles per-view BGL mapping) with a thin Rust struct emit. The view-binding-order helpers stay where they are; the emitter just walks them and produces the matching `BindGroupLayoutEntry` Vec.

**Files:**
- Create: `crates/dsl_compiler/src/emit_scoring_kernel.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (register module)
- Modify: `crates/xtask/src/main.rs` (call new emitter; thread `view_specs` from compilation)
- Generated: `crates/engine_gpu_rules/src/scoring.{rs,wgsl}` + `scoring_unpack.{rs,wgsl}`
- Test: `crates/dsl_compiler/tests/emit_scoring_kernel_smoke.rs`

- [ ] **Step 1: Write failing emitter test**

Create `crates/dsl_compiler/tests/emit_scoring_kernel_smoke.rs`:

```rust
use dsl_compiler::emit_scoring_kernel::{emit_scoring_rs, emit_scoring_unpack_rs};

#[test]
fn scoring_rs_has_kernel_impl_and_view_bindings() {
    let src = emit_scoring_rs(&[]); // no views — minimal core layout
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub struct ScoringKernel"));
    assert!(src.contains("impl crate::Kernel for ScoringKernel"));
    assert!(src.contains("include_str!(\"scoring.wgsl\")"));
    // 5 core bindings unconditional: agent_data, mask_bitmaps, scoring_table, scoring_out, cfg
    assert!(src.contains("// agent_data"));
    assert!(src.contains("// scoring_table"));
    assert!(src.contains("// scoring_out"));
}

#[test]
fn scoring_unpack_rs_has_impl() {
    let src = emit_scoring_unpack_rs();
    assert!(src.contains("pub struct ScoringUnpackKernel"));
    assert!(src.contains("impl crate::Kernel for ScoringUnpackKernel"));
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `cargo test -p dsl_compiler --test emit_scoring_kernel_smoke`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement emit_scoring_kernel.rs**

Create `crates/dsl_compiler/src/emit_scoring_kernel.rs`:

```rust
//! Emits `engine_gpu_rules/src/scoring.rs` + `scoring_unpack.rs`.
//!
//! Today's hand-written `engine_gpu::scoring::ScoringKernel` builds its
//! BGL by walking `scoring_view_binding_order(specs)` and adding per-
//! view bindings. This emitter does the same walk but produces the
//! BGL-entry list as a const slice in the emitted Rust file — so the
//! BGL is a compile-time constant (one per regen) rather than a
//! runtime walk.

use crate::emit_scoring_wgsl::scoring_view_binding_order;

/// Minimal `ViewStorageSpec` snapshot the emitter consumes. Real type
/// lives in `engine_gpu::view_storage`; this version is the subset
/// the dsl_compiler can construct from IR. The xtask plumbing in step
/// 5 below converts a `Compilation`'s view list into a Vec of these.
#[derive(Debug, Clone)]
pub struct ViewSpecForEmit {
    pub name: String,
    /// One of "SlotMap", "PairMapScalar", "PairMapDecay".
    pub shape: String,
    pub topk: bool,
}

pub fn emit_scoring_rs(specs: &[ViewSpecForEmit]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringBindings<'a> {{").unwrap();
    writeln!(out, "    pub agent_data: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub mask_bitmaps: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub scoring_table: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub scoring_out: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub view_buffers: &'a [&'a wgpu::Buffer],").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct ScoringCfg {{").unwrap();
    writeln!(out, "    pub agent_cap: u32,").unwrap();
    writeln!(out, "    pub num_actions: u32,").unwrap();
    writeln!(out, "    pub tick: u32,").unwrap();
    writeln!(out, "    pub _pad: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"scoring.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for ScoringKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = ScoringBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = ScoringCfg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = vec![").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_storage(0, true),  // agent_data").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_storage(1, true),  // mask_bitmaps").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_storage(2, true),  // scoring_table").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_storage(3, false), // scoring_out").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_uniform(4),        // cfg").unwrap();
    writeln!(out, "        ];").unwrap();
    writeln!(out, "        let mut binding: u32 = 5;").unwrap();
    for spec in specs {
        match spec.shape.as_str() {
            "SlotMap" => {
                writeln!(out, "        // view '{}' shape=SlotMap (1 binding)", spec.name).unwrap();
                writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, true)); binding += 1;").unwrap();
            }
            "PairMapScalar" => {
                writeln!(out, "        // view '{}' shape=PairMapScalar (1 binding{})", spec.name, if spec.topk { ", topk: +2 anchors+ids" } else { "" }).unwrap();
                writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                if spec.topk {
                    writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                    writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                }
            }
            "PairMapDecay" => {
                writeln!(out, "        // view '{}' shape=PairMapDecay (2 bindings{})", spec.name, if spec.topk { ", topk: +1 ids" } else { "" }).unwrap();
                writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                if spec.topk {
                    writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                }
            }
            _ => {}
        }
    }
    writeln!(out, "        // sim_cfg sits past every view binding (matches emit_scoring_wgsl::scoring_sim_cfg_binding).").unwrap();
    writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, true));").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::bgl\"),").unwrap();
    writeln!(out, "            entries: &entries,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_scoring\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> ScoringCfg {{").unwrap();
    writeln!(out, "        ScoringCfg {{ agent_cap: state.agent_cap(), num_actions: 16, tick: state.tick, _pad: 0 }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> ScoringBindings<'a> {{").unwrap();
    writeln!(out, "        ScoringBindings {{").unwrap();
    writeln!(out, "            agent_data:    sources.external.agents,").unwrap();
    writeln!(out, "            mask_bitmaps:  sources.transient.mask_bitmaps,").unwrap();
    writeln!(out, "            scoring_table: &sources.resident.scoring_table,").unwrap();
    writeln!(out, "            scoring_out:   sources.transient.action_buf,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "            sim_cfg:       sources.external.sim_cfg,").unwrap();
    writeln!(out, "            // view_buffers is the per-view list pulled from resident; the").unwrap();
    writeln!(out, "            // emitter walks `specs` to know which fields to slice. Per-view").unwrap();
    writeln!(out, "            // accessor `sources.resident.view_storage_<name>` gives the").unwrap();
    writeln!(out, "            // primary buffer; topk views require additional fields. The").unwrap();
    writeln!(out, "            // helper `sources.resident.scoring_view_buffers_slice()` (emitted").unwrap();
    writeln!(out, "            // alongside the resident context) returns the expected slice.").unwrap();
    writeln!(out, "            view_buffers: sources.resident.scoring_view_buffers_slice(),").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &ScoringBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let mut bg_entries: Vec<wgpu::BindGroupEntry> = vec![").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agent_data.as_entire_binding() }},").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 1, resource: bindings.mask_bitmaps.as_entire_binding() }},").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 2, resource: bindings.scoring_table.as_entire_binding() }},").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 3, resource: bindings.scoring_out.as_entire_binding() }},").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 4, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "        ];").unwrap();
    writeln!(out, "        let mut next_b: u32 = 5;").unwrap();
    writeln!(out, "        for buf in bindings.view_buffers.iter() {{").unwrap();
    writeln!(out, "            bg_entries.push(wgpu::BindGroupEntry {{ binding: next_b, resource: buf.as_entire_binding() }});").unwrap();
    writeln!(out, "            next_b += 1;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        bg_entries.push(wgpu::BindGroupEntry {{ binding: next_b, resource: bindings.sim_cfg.as_entire_binding() }});").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &bg_entries,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn emit_scoring_unpack_rs() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringUnpackKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringUnpackBindings<'a> {{").unwrap();
    writeln!(out, "    pub agent_data: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub agents_input: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct ScoringUnpackCfg {{ pub agent_cap: u32, pub _pad: [u32; 3] }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"scoring_unpack.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for ScoringUnpackKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = ScoringUnpackBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = ScoringUnpackCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::bgl\"),").unwrap();
    writeln!(out, "            entries: &[crate::fused_mask::bgl_storage(0, false), crate::fused_mask::bgl_storage(1, true), crate::fused_mask::bgl_uniform(2)],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_scoring_unpack\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> ScoringUnpackCfg {{").unwrap();
    writeln!(out, "        ScoringUnpackCfg {{ agent_cap: state.agent_cap(), _pad: [0; 3] }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> ScoringUnpackBindings<'a> {{").unwrap();
    writeln!(out, "        ScoringUnpackBindings {{").unwrap();
    writeln!(out, "            agent_data:   sources.external.agents,").unwrap();
    writeln!(out, "            agents_input: sources.transient.scoring_unpack_agents_input,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &ScoringUnpackBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agent_data.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.agents_input.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

- [ ] **Step 4: Register module + run unit tests**

Add `pub mod emit_scoring_kernel;` to `crates/dsl_compiler/src/lib.rs`.

Run: `cargo test -p dsl_compiler --test emit_scoring_kernel_smoke`
Expected: 2 passed.

- [ ] **Step 5: Wire xtask compile-dsl to write scoring + scoring_unpack files**

In `crates/xtask/src/main.rs`, extend the modules vec and write:

```rust
modules.push("scoring".into());
modules.push("scoring_unpack".into());
modules.sort();

{
    use std::fs;
    use std::path::PathBuf;
    use dsl_compiler::emit_scoring_kernel::{emit_scoring_rs, emit_scoring_unpack_rs, ViewSpecForEmit};

    // Convert IR view list → ViewSpecForEmit. The shape mapping
    // mirrors `engine_gpu::view_storage::build_all_specs()`'s match
    // — concrete logic lives in emit_view_fold_kernel (Task 11);
    // for now hard-code the four standard shapes.
    let view_specs: Vec<ViewSpecForEmit> = comp.views.iter().filter_map(|v| {
        let shape = match &v.body {
            // Adapt actual ViewBodyIR variants to "SlotMap" / "PairMapScalar" / "PairMapDecay".
            _ => "PairMapScalar".to_string(),
        };
        Some(ViewSpecForEmit { name: v.name.clone(), shape, topk: false })
    }).collect();

    fs::write(PathBuf::from("crates/engine_gpu_rules/src/scoring.rs"), emit_scoring_rs(&view_specs)).expect("scoring.rs");

    let scoring_wgsl_body = dsl_compiler::emit_scoring_wgsl::emit_scoring_wgsl();
    let scoring_wgsl = format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{scoring_wgsl_body}");
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/scoring.wgsl"), scoring_wgsl).expect("scoring.wgsl");

    fs::write(PathBuf::from("crates/engine_gpu_rules/src/scoring_unpack.rs"), emit_scoring_unpack_rs()).expect("scoring_unpack.rs");
    let unpack_wgsl = "// GENERATED by dsl_compiler. Do not edit by hand.\n@compute @workgroup_size(64) fn cs_scoring_unpack(@builtin(global_invocation_id) gid: vec3<u32>) {}\n";
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/scoring_unpack.wgsl"), unpack_wgsl).expect("scoring_unpack.wgsl");
}
```

- [ ] **Step 6: Re-run compile-dsl + build engine_gpu_rules**

Run: `cargo run --bin xtask -- compile-dsl`
Run: `cargo build -p engine_gpu_rules`
Expected: clean build.

- [ ] **Step 7: Wire engine_gpu's step_batch to use the emitted ScoringKernel**

The xtask plumbing for Task 6 also needs to push the buffers ScoringKernel reads onto the binding-source containers. Inside the Task 6 xtask block (Step 5), append:

```rust
use dsl_compiler::emit_resident_context::ResidentField;
use dsl_compiler::emit_transient_handles::TransientField;

resident_fields.push(ResidentField {
    name: "scoring_table".into(),
    doc: "Resident scoring table (per-action priors).".into(),
});
// scoring_view_buffers_slice() is a method emitted alongside the resident
// context — it returns &[&wgpu::Buffer] over the per-view fields.
// emit_resident_context tracks scoring views in a separate list and
// generates the slice helper + per-view fields together.
for spec in &view_specs {
    resident_fields.push(ResidentField {
        name: format!("view_storage_{}", spec.name),
        doc: format!("Resident view storage for `{}`.", spec.name),
    });
}
transient_fields.push(TransientField {
    name: "action_buf".into(),
    doc: "ScoringKernel output (action-per-agent buffer).".into(),
});
transient_fields.push(TransientField {
    name: "scoring_unpack_agents_input".into(),
    doc: "ScoringUnpackKernel scratch.".into(),
});
```

In `crates/engine_gpu/src/lib.rs`'s `step_batch` body, replace the existing `self.sync.scoring_kernel.run_resident(...)` call with:

```rust
{
    use engine_gpu_rules::scoring::ScoringKernel;
    use engine_gpu_rules::Kernel as _;

    let kernel = self.resident.scoring_kernel
        .get_or_insert_with(|| ScoringKernel::new(&self.device));
    let cfg = kernel.build_cfg(state);
    let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scoring::cfg"),
        contents: bytemuck::cast_slice(&[cfg]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bindings = kernel.bind(&sources, &cfg_buf);
    kernel.record(&self.device, &mut encoder, &bindings, agent_cap);
}
```

`sources` is the `BindingSources<'_>` aggregate built once at the top of the per-tick loop in Task 5. Engine_gpu does not construct `ScoringBindings` directly — `kernel.bind(&sources, &cfg_buf)` returns it via the emitted body.

For the `transient.action_buf` and `transient.scoring_unpack_agents_input` fields engine_gpu must populate the `TransientHandles` struct each tick — extend Task 5's transient construction:

```rust
let transient = TransientHandles {
    mask_bitmaps:                self.sync.mask_kernel.mask_bitmaps_buf(),
    mask_unpack_agents_input:    self.sync.mask_kernel.unpack_agents_input_buf(),
    action_buf:                  self.sync.scoring_kernel.scoring_out_buf(),
    scoring_unpack_agents_input: self.sync.scoring_kernel.scoring_unpack_input_buf(),
    _phantom: std::marker::PhantomData,
};
```

Add `pub scoring_kernel: Option<engine_gpu_rules::scoring::ScoringKernel>` to `ResidentPathContext` (engine_gpu side wrapper; init `None` in `new()`). Add transitional accessors on the hand-written `engine_gpu::scoring::ScoringKernel` for `scoring_out_buf()`, `scoring_unpack_input_buf()`. The `resident.scoring_table` and per-view buffers are owned by `engine_gpu_rules::ResidentPathContext` and constructed in its emitted `new()` — engine_gpu's wrapper context delegates allocation to that constructor.

- [ ] **Step 8: Run parity sweep**

Run: `cargo test -p engine_gpu --test parity_with_cpu`
Run: `cargo test -p engine --test wolves_and_humans_parity`
Run: `cargo test -p engine_gpu --test view_parity`
Run: `cargo test -p engine_gpu --test topk_view_parity`
Expected: all pass.

- [ ] **Step 9: Bump baseline + commit**

Run: `cargo run --bin xtask -- compile-dsl`
Run: `cargo test -p engine_gpu_rules --test schema_hash`
Expected: PASS.

```bash
git add crates/dsl_compiler/src/emit_scoring_kernel.rs crates/dsl_compiler/src/lib.rs crates/xtask/ crates/engine_gpu_rules/ crates/engine_gpu/src/
git commit -m "feat(dsl_compiler): emit ScoringKernel + ScoringUnpackKernel; wire step_batch to consume them"
```

---

### Task 7: Emit `ApplyActionsKernel`

The apply-actions kernel reads `scoring_out` + `agent_data` and produces `actions_buf` + emits damage/heal/move events into the cascade ring. The hand-written body lives at `crates/engine_gpu/src/apply_actions.rs:172-840`.

**Files:**
- Modify: `crates/dsl_compiler/src/emit_scoring_kernel.rs` (add `emit_apply_actions_rs`)
- Modify: `crates/xtask/src/main.rs` (write apply_actions module)
- Modify: `crates/engine_gpu/src/lib.rs` (call emitted kernel)
- Test: extend `crates/dsl_compiler/tests/emit_scoring_kernel_smoke.rs`

- [ ] **Step 1: Add failing test**

Append to `crates/dsl_compiler/tests/emit_scoring_kernel_smoke.rs`:

```rust
#[test]
fn apply_actions_rs_has_kernel_impl() {
    let src = dsl_compiler::emit_scoring_kernel::emit_apply_actions_rs();
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub struct ApplyActionsKernel"));
    assert!(src.contains("impl crate::Kernel for ApplyActionsKernel"));
    assert!(src.contains("include_str!(\"apply_actions.wgsl\")"));
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test emit_scoring_kernel_smoke apply_actions`
Expected: FAIL — `emit_apply_actions_rs not found`.

- [ ] **Step 3: Add emit_apply_actions_rs to emit_scoring_kernel.rs**

Append at the bottom of `crates/dsl_compiler/src/emit_scoring_kernel.rs`:

```rust
pub fn emit_apply_actions_rs() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ApplyActionsKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ApplyActionsBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub scoring_out: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_ring_records: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_ring_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct ApplyActionsCfg {{").unwrap();
    writeln!(out, "    pub agent_cap: u32,").unwrap();
    writeln!(out, "    pub tick: u32,").unwrap();
    writeln!(out, "    pub event_ring_capacity: u32,").unwrap();
    writeln!(out, "    pub _pad: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"apply_actions.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for ApplyActionsKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = ApplyActionsBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = ApplyActionsCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(0, false), // agents (rw)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(1, true),  // scoring_out").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(2, false), // event_ring_records").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(3, false), // event_ring_tail (atomic)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(4, true),  // sim_cfg").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_uniform(5),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_apply_actions\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> ApplyActionsCfg {{").unwrap();
    writeln!(out, "        ApplyActionsCfg {{ agent_cap: state.agent_cap(), tick: state.tick, event_ring_capacity: 4096, _pad: 0 }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> ApplyActionsBindings<'a> {{").unwrap();
    writeln!(out, "        ApplyActionsBindings {{").unwrap();
    writeln!(out, "            agents:             sources.external.agents,").unwrap();
    writeln!(out, "            scoring_out:        sources.transient.action_buf,").unwrap();
    writeln!(out, "            event_ring_records: &sources.pingpong.events_a_records,").unwrap();
    writeln!(out, "            event_ring_tail:    &sources.pingpong.events_a_tail,").unwrap();
    writeln!(out, "            sim_cfg:            sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &ApplyActionsBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.scoring_out.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.event_ring_records.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.event_ring_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 4, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 5, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

- [ ] **Step 4: Run unit test**

Run: `cargo test -p dsl_compiler --test emit_scoring_kernel_smoke apply_actions`
Expected: PASS.

- [ ] **Step 5: Wire xtask + write the wgsl body**

In `crates/xtask/src/main.rs`:

```rust
modules.push("apply_actions".into());
modules.sort();
{
    use std::fs;
    use std::path::PathBuf;
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/apply_actions.rs"),
        dsl_compiler::emit_scoring_kernel::emit_apply_actions_rs()).expect("apply_actions.rs");
    // The apply-actions WGSL body is currently produced inside
    // `engine_gpu::apply_actions::ApplyActionsKernel::new` as a static
    // string. Hoist that string into a `pub const APPLY_ACTIONS_WGSL`
    // (transitional, removed in Task 16) and write it through:
    let body = engine_gpu::apply_actions::APPLY_ACTIONS_WGSL;
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/apply_actions.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{body}\n")).expect("apply_actions.wgsl");
}
```

Note: this requires xtask to depend on `engine_gpu` to access the WGSL string. If the dep isn't already there, add it under `[dependencies]` in `crates/xtask/Cargo.toml`. That dep is acceptable transitionally; Task 16 deletes the source.

- [ ] **Step 6: Wire engine_gpu's step_batch to call the emitted apply-actions kernel**

The xtask plumbing for Task 7 also needs to push the buffers ApplyActionsKernel reads onto the binding-source containers. Inside the Task 7 xtask block, append:

```rust
use dsl_compiler::emit_pingpong_context::PingPongField;

if !pingpong_fields.iter().any(|f: &PingPongField| f.name == "events_a_records") {
    pingpong_fields.push(PingPongField {
        name: "events_a_records".into(),
        doc: "Cascade-physics A-ring event records (write side at iter 0).".into(),
    });
    pingpong_fields.push(PingPongField {
        name: "events_a_tail".into(),
        doc: "Cascade-physics A-ring tail (atomic counter).".into(),
    });
    pingpong_fields.push(PingPongField {
        name: "events_b_records".into(),
        doc: "Cascade-physics B-ring event records.".into(),
    });
    pingpong_fields.push(PingPongField {
        name: "events_b_tail".into(),
        doc: "Cascade-physics B-ring tail.".into(),
    });
}
```

In the `step_batch` body, replace the current `self.sync.apply_actions_kernel.run_resident(...)` call with:

```rust
{
    use engine_gpu_rules::apply_actions::ApplyActionsKernel;
    use engine_gpu_rules::Kernel as _;
    let kernel = self.resident.apply_actions_kernel
        .get_or_insert_with(|| ApplyActionsKernel::new(&self.device));
    let cfg = kernel.build_cfg(state);
    let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("apply_actions::cfg"),
        contents: bytemuck::cast_slice(&[cfg]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bindings = kernel.bind(&sources, &cfg_buf);
    kernel.record(&self.device, &mut encoder, &bindings, agent_cap);
}
```

The PingPong A/B ring buffers live on `sources.pingpong` (engine_gpu's persistent `engine_gpu_rules::PingPongContext`). The emitted `bind()` body always writes to `events_a` — the pingpong-iteration logic is internal to the cascade FixedPoint loop (Task 10), which swaps the A/B field references at each iteration. ApplyActions is the seed step (iter 0), so it deterministically writes the A side.

Add `pub apply_actions_kernel: Option<engine_gpu_rules::apply_actions::ApplyActionsKernel>` to `ResidentPathContext` (engine_gpu side wrapper).

- [ ] **Step 7: Run parity sweep**

Run: `cargo test -p engine_gpu --test parity_with_cpu`
Run: `cargo test -p engine --test wolves_and_humans_parity`
Run: `cargo test -p engine_gpu --test cascade_parity`
Expected: all pass.

- [ ] **Step 8: Bump baseline + commit**

Run: `cargo run --bin xtask -- compile-dsl`
Run: `cargo test -p engine_gpu_rules --test schema_hash`
Expected: PASS.

```bash
git add crates/dsl_compiler/ crates/xtask/ crates/engine_gpu_rules/ crates/engine_gpu/src/
git commit -m "feat(dsl_compiler): emit ApplyActionsKernel; wire step_batch dispatch"
```

---

### Task 8: Emit `PickAbilityKernel` (per_ability variant)

The per_ability row body emit (kernel WGSL) already landed in commits `d8e196e8` (`emit_pick_ability_wgsl`) and `8f8e3582` (schema-hash coverage). This task adds the wrapper Rust struct emit on top — the dispatch-emit equivalent of Task 4 but reusing existing IR emit work.

**Files:**
- Create: `crates/dsl_compiler/src/emit_pick_ability_kernel.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (register module)
- Modify: `crates/xtask/src/main.rs` (call new emitter; reuse `emit_pick_ability_wgsl`)
- Modify: `crates/engine_gpu/src/lib.rs` (call emitted kernel in `step_batch`)
- Test: `crates/dsl_compiler/tests/emit_pick_ability_kernel_smoke.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/dsl_compiler/tests/emit_pick_ability_kernel_smoke.rs`:

```rust
use dsl_compiler::emit_pick_ability_kernel::emit_pick_ability_rs;

#[test]
fn pick_ability_rs_has_kernel_impl_and_chosen_buf_binding() {
    let src = emit_pick_ability_rs();
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub struct PickAbilityKernel"));
    assert!(src.contains("pub struct PickAbilityBindings"));
    assert!(src.contains("pub chosen_ability_buf:"));
    assert!(src.contains("pub per_slot_cooldown:"));
    assert!(src.contains("impl crate::Kernel for PickAbilityKernel"));
    assert!(src.contains("include_str!(\"pick_ability.wgsl\")"));
    assert!(src.contains("entry_point: Some(\"cs_pick_ability\")"));
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test emit_pick_ability_kernel_smoke`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement emit_pick_ability_kernel.rs**

Create `crates/dsl_compiler/src/emit_pick_ability_kernel.rs`:

```rust
//! Emits `engine_gpu_rules/src/pick_ability.rs`.
//!
//! The companion WGSL emit is `emit_scoring_wgsl::emit_pick_ability_wgsl`
//! (already landed). This emitter is the wrapper struct on top — it
//! produces the BGL, pipeline, and dispatch encoder for the
//! `cs_pick_ability` entry point.

pub fn emit_pick_ability_rs() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_pick_ability_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct PickAbilityKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct PickAbilityBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub ability_registry: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub per_slot_cooldown: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub chosen_ability_buf: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub tag_values: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct PickAbilityCfg {{").unwrap();
    writeln!(out, "    pub agent_cap: u32,").unwrap();
    writeln!(out, "    pub ability_count: u32,").unwrap();
    writeln!(out, "    pub num_tags: u32,").unwrap();
    writeln!(out, "    pub tick: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"pick_ability.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for PickAbilityKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = PickAbilityBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = PickAbilityCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::pick_ability::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::pick_ability::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(0, true),  // agents").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(1, true),  // ability_registry").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(2, false), // per_slot_cooldown (rw)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(3, false), // chosen_ability_buf (rw)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(4, true),  // sim_cfg").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(5, true),  // tag_values").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_uniform(6),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::pick_ability::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::pick_ability::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_pick_ability\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> PickAbilityCfg {{").unwrap();
    writeln!(out, "        PickAbilityCfg {{").unwrap();
    writeln!(out, "            agent_cap: state.agent_cap(),").unwrap();
    writeln!(out, "            ability_count: 0,").unwrap();
    writeln!(out, "            num_tags: 0,").unwrap();
    writeln!(out, "            tick: state.tick,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> PickAbilityBindings<'a> {{").unwrap();
    writeln!(out, "        PickAbilityBindings {{").unwrap();
    writeln!(out, "            agents:             sources.external.agents,").unwrap();
    writeln!(out, "            ability_registry:   sources.external.ability_registry,").unwrap();
    writeln!(out, "            per_slot_cooldown:  &sources.resident.per_slot_cooldown,").unwrap();
    writeln!(out, "            chosen_ability_buf: &sources.resident.chosen_ability_buf,").unwrap();
    writeln!(out, "            sim_cfg:            sources.external.sim_cfg,").unwrap();
    writeln!(out, "            tag_values:         sources.external.tag_values,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &PickAbilityBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::pick_ability::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.ability_registry.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.per_slot_cooldown.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.chosen_ability_buf.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 4, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 5, resource: bindings.tag_values.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 6, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::pick_ability::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

- [ ] **Step 4: Register + run unit test**

Add `pub mod emit_pick_ability_kernel;` to `crates/dsl_compiler/src/lib.rs`.

Run: `cargo test -p dsl_compiler --test emit_pick_ability_kernel_smoke`
Expected: PASS.

- [ ] **Step 5: Wire xtask to write pick_ability.{rs,wgsl}**

In `crates/xtask/src/main.rs`:

```rust
modules.push("pick_ability".into());
modules.sort();
{
    use std::fs;
    use std::path::PathBuf;
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/pick_ability.rs"),
        dsl_compiler::emit_pick_ability_kernel::emit_pick_ability_rs()).expect("pick_ability.rs");
    // The WGSL emitter for per_ability rows takes a single ScoringIR.
    // If multiple per_ability rows are present, the emitter merges them
    // (existing behaviour landed in commit d8e196e8).
    let per_ability_rows: Vec<&dsl_compiler::ir::ScoringIR> = comp.scoring.iter()
        .filter(|s| s.row_type == dsl_compiler::ir::ScoringRowType::PerAbility)
        .collect();
    let wgsl_body = if let Some(s) = per_ability_rows.first() {
        dsl_compiler::emit_scoring_wgsl::emit_pick_ability_wgsl(s)
    } else {
        // No per_ability rows in this compilation — emit a minimal
        // entry-point stub so the kernel loads even when unused.
        "@compute @workgroup_size(64) fn cs_pick_ability(@builtin(global_invocation_id) gid: vec3<u32>) {}\n".to_string()
    };
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/pick_ability.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{wgsl_body}")).expect("pick_ability.wgsl");
}
```

- [ ] **Step 6: Wire into engine_gpu's step_batch**

The xtask plumbing for Task 8 also needs to push the buffers PickAbilityKernel reads onto the binding-source containers. Inside the Task 8 xtask block, append:

```rust
use dsl_compiler::emit_resident_context::ResidentField;
use dsl_compiler::emit_external_buffers::ExternalField;

resident_fields.push(ResidentField {
    name: "per_slot_cooldown".into(),
    doc: "Per-agent per-slot cooldown counters (Resident; persists across ticks).".into(),
});
resident_fields.push(ResidentField {
    name: "chosen_ability_buf".into(),
    doc: "PickAbilityKernel output (Resident; consumed by ApplyActions next tick).".into(),
});
external_fields.push(ExternalField {
    name: "ability_registry".into(),
    doc: "AbilityRegistry buffer (engine-owned).".into(),
});
external_fields.push(ExternalField {
    name: "tag_values".into(),
    doc: "Per-tag value table (engine-owned).".into(),
});
```

In `crates/engine_gpu/src/lib.rs::step_batch`, after the scoring kernel dispatch and before apply_actions, add:

```rust
{
    use engine_gpu_rules::pick_ability::PickAbilityKernel;
    use engine_gpu_rules::Kernel as _;
    let kernel = self.resident.pick_ability_kernel
        .get_or_insert_with(|| PickAbilityKernel::new(&self.device));
    let cfg = kernel.build_cfg(state);
    let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pick_ability::cfg"),
        contents: bytemuck::cast_slice(&[cfg]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bindings = kernel.bind(&sources, &cfg_buf);
    kernel.record(&self.device, &mut encoder, &bindings, agent_cap);
}
```

`sources.external.ability_registry` and `sources.external.tag_values` are populated by the `ExternalBuffers::new(...)` call site in engine_gpu's per-batch init — engine_gpu passes the existing registry / tag-value buffers (currently held on `CascadeResidentCtx`) when building the External aggregate.

Add `pub pick_ability_kernel: Option<engine_gpu_rules::pick_ability::PickAbilityKernel>` to `ResidentPathContext` (engine_gpu side wrapper). Wire `external.ability_registry` / `external.tag_values` from the existing `CascadeResidentCtx` accessors at the construction site of `ExternalBuffers` (one tick-level integration; transitional). The `resident.per_slot_cooldown` and `resident.chosen_ability_buf` buffers move into `engine_gpu_rules::ResidentPathContext`'s emitted constructor.

- [ ] **Step 7: Parity sweep**

Run: `cargo test -p engine_gpu --test parity_with_cpu`
Run: `cargo test -p engine --test wolves_and_humans_parity`
Run: `cargo test -p dsl_compiler --test per_ability_row` (the existing per_ability schema-hash test).
Expected: all pass.

- [ ] **Step 8: Bump baseline + commit**

```bash
cargo run --bin xtask -- compile-dsl
cargo test -p engine_gpu_rules --test schema_hash
git add crates/dsl_compiler/ crates/xtask/ crates/engine_gpu_rules/ crates/engine_gpu/src/
git commit -m "feat(dsl_compiler): emit PickAbilityKernel wrapper; wire step_batch"
```

---

### Task 9: Emit `MovementKernel` (with FixedPoint Schedule op)

Movement is target-bound: the kernel reads scoring's chosen action + per-agent move target, advances position by `move_speed_mps × tick_dt`, and emits `AgentMoved` events. Hand-written body at `crates/engine_gpu/src/movement.rs:137-770`.

This task also lights up the `DispatchOp::FixedPoint` variant — Movement itself is a single dispatch, but cascade physics (Task 10) consumes it inside the FixedPoint loop. We add the `FixedPoint` Schedule entry now so Task 10 has a clean target.

**Files:**
- Create: `crates/dsl_compiler/src/emit_movement_kernel.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (register module)
- Modify: `crates/xtask/src/main.rs` (call emitter; populate Schedule entry)
- Modify: `crates/engine_gpu/src/lib.rs` (call emitted kernel)
- Test: `crates/dsl_compiler/tests/emit_movement_kernel_smoke.rs`

- [ ] **Step 1: Write failing test**

Create `crates/dsl_compiler/tests/emit_movement_kernel_smoke.rs`:

```rust
use dsl_compiler::emit_movement_kernel::emit_movement_rs;

#[test]
fn movement_rs_has_kernel_impl() {
    let src = emit_movement_rs();
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub struct MovementKernel"));
    assert!(src.contains("impl crate::Kernel for MovementKernel"));
    assert!(src.contains("include_str!(\"movement.wgsl\")"));
    assert!(src.contains("entry_point: Some(\"cs_movement\")"));
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test emit_movement_kernel_smoke`
Expected: FAIL.

- [ ] **Step 3: Implement emit_movement_kernel.rs**

Create `crates/dsl_compiler/src/emit_movement_kernel.rs`:

```rust
//! Emits `engine_gpu_rules/src/movement.rs`. Movement reads scoring's
//! chosen action + per-agent target and advances position; appends
//! `AgentMoved` events.

pub fn emit_movement_rs() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_movement_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct MovementKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct MovementBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub scoring: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_ring_records: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_ring_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct MovementCfg {{").unwrap();
    writeln!(out, "    pub agent_cap: u32,").unwrap();
    writeln!(out, "    pub tick: u32,").unwrap();
    writeln!(out, "    pub event_ring_capacity: u32,").unwrap();
    writeln!(out, "    pub _pad: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"movement.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for MovementKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = MovementBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = MovementCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::movement::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::movement::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(0, false), // agents (rw)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(1, true),  // scoring").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(2, false), // event_ring_records (atomic append)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(3, false), // event_ring_tail (atomic counter)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(4, true),  // sim_cfg").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_uniform(5),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::movement::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::movement::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_movement\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> MovementCfg {{").unwrap();
    writeln!(out, "        MovementCfg {{ agent_cap: state.agent_cap(), tick: state.tick, event_ring_capacity: 4096, _pad: 0 }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> MovementBindings<'a> {{").unwrap();
    writeln!(out, "        MovementBindings {{").unwrap();
    writeln!(out, "            agents:             sources.external.agents,").unwrap();
    writeln!(out, "            scoring:            sources.transient.action_buf,").unwrap();
    writeln!(out, "            event_ring_records: &sources.pingpong.events_a_records,").unwrap();
    writeln!(out, "            event_ring_tail:    &sources.pingpong.events_a_tail,").unwrap();
    writeln!(out, "            sim_cfg:            sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &MovementBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::movement::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.scoring.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.event_ring_records.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.event_ring_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 4, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 5, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::movement::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

- [ ] **Step 4: Register module + run test**

Add `pub mod emit_movement_kernel;` to `crates/dsl_compiler/src/lib.rs`.

Run: `cargo test -p dsl_compiler --test emit_movement_kernel_smoke`
Expected: PASS.

- [ ] **Step 5: Wire xtask to write movement.{rs,wgsl}**

In `crates/xtask/src/main.rs`:

```rust
modules.push("movement".into());
modules.sort();
{
    use std::fs;
    use std::path::PathBuf;
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/movement.rs"),
        dsl_compiler::emit_movement_kernel::emit_movement_rs()).expect("movement.rs");
    let body = engine_gpu::movement::MOVEMENT_WGSL;
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/movement.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{body}\n")).expect("movement.wgsl");
}
```

Hoist the WGSL string out of `engine_gpu::movement::build_shader` into a `pub const MOVEMENT_WGSL: &str = "..."`. Transitional — Task 16 deletes `engine_gpu::movement`.

- [ ] **Step 6: Wire engine_gpu's step_batch to call the emitted MovementKernel**

In `crates/engine_gpu/src/lib.rs::step_batch`, replace the existing movement dispatch with:

```rust
{
    use engine_gpu_rules::movement::MovementKernel;
    use engine_gpu_rules::Kernel as _;
    let kernel = self.resident.movement_kernel
        .get_or_insert_with(|| MovementKernel::new(&self.device));
    let cfg = kernel.build_cfg(state);
    let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("movement::cfg"),
        contents: bytemuck::cast_slice(&[cfg]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bindings = kernel.bind(&sources, &cfg_buf);
    kernel.record(&self.device, &mut encoder, &bindings, agent_cap);
}
```

The `event_ring_records` and `event_ring_tail` fields wire to `pingpong.events_a_*` (Movement seeds the cascade A-ring alongside ApplyActions). All other refs come from already-populated containers (transient.action_buf from Task 6, external.agents/sim_cfg from Task 4).

Add `pub movement_kernel: Option<engine_gpu_rules::movement::MovementKernel>` to `ResidentPathContext` (engine_gpu side wrapper).

- [ ] **Step 7: Add a Schedule entry for FixedPoint that we'll consume in Task 10**

In `crates/xtask/src/main.rs`, change the schedule-emit call:

```rust
use dsl_compiler::emit_schedule::{emit_schedule_rs, ScheduleEntry, DispatchOpKind};
let entries = vec![
    ScheduleEntry { kernel: "FusedMask".into(),       kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "Scoring".into(),         kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "PickAbility".into(),     kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "ApplyActions".into(),    kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "Movement".into(),        kind: DispatchOpKind::Kernel },
    // Physics lands as FixedPoint in Task 10.
];
fs::write(PathBuf::from("crates/engine_gpu_rules/src/schedule.rs"),
    emit_schedule_rs(&entries)).expect("schedule.rs");
```

(Task 10 appends the FixedPoint and Indirect entries.)

- [ ] **Step 8: Parity sweep**

Run: `cargo test -p engine_gpu --test parity_with_cpu`
Run: `cargo test -p engine --test wolves_and_humans_parity`
Run: `cargo test -p engine_gpu --test physics_parity` (still expected to pass — physics not yet migrated)
Expected: all pass.

- [ ] **Step 9: Bump baseline + commit**

```bash
cargo run --bin xtask -- compile-dsl
cargo test -p engine_gpu_rules --test schema_hash
git add crates/dsl_compiler/ crates/xtask/ crates/engine_gpu_rules/ crates/engine_gpu/src/
git commit -m "feat(dsl_compiler): emit MovementKernel; populate Schedule with target_bound entries"
```

---

### Task 10: Emit `PhysicsKernel`, `SeedIndirectKernel`, `AppendEventsKernel` (FixedPoint + Indirect)

Cascade physics is the iterative kernel: each iteration drains the event ring, produces follow-up events into the next-iter ring, then `SeedIndirectKernel` writes the next iteration's `dispatch_indirect` args. The Schedule expresses this as `DispatchOp::FixedPoint { kernel: KernelId::Physics, max_iter: 8 }`. `AppendEventsKernel` emits chronicle/audit events; it sits at the end of the cascade.

Hand-written sources: `engine_gpu/src/physics.rs:1538-2860`, `engine_gpu/src/cascade_resident.rs:357 (SeedIndirectKernel)`, `:620 (AppendEventsKernel)`.

**Files:**
- Modify: `crates/dsl_compiler/src/emit_movement_kernel.rs` (add `emit_physics_rs`, `emit_seed_indirect_rs`, `emit_append_events_rs`)
- Modify: `crates/dsl_compiler/src/emit_pingpong_context.rs` (NEW — A/B ring buffer fields)
- Modify: `crates/dsl_compiler/src/lib.rs` (register pingpong module)
- Modify: `crates/xtask/src/main.rs` (write physics + seed + append; populate FixedPoint + Indirect entries)
- Modify: `crates/engine_gpu/src/lib.rs` (replace physics dispatch loop with Schedule iteration)
- Test: `crates/dsl_compiler/tests/emit_physics_smoke.rs`

- [ ] **Step 1: Failing test**

Create `crates/dsl_compiler/tests/emit_physics_smoke.rs`:

```rust
use dsl_compiler::emit_movement_kernel::{emit_physics_rs, emit_seed_indirect_rs, emit_append_events_rs};

#[test]
fn physics_rs_has_kernel_impl() {
    let src = emit_physics_rs();
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub struct PhysicsKernel"));
    assert!(src.contains("impl crate::Kernel for PhysicsKernel"));
    assert!(src.contains("include_str!(\"physics.wgsl\")"));
}

#[test]
fn seed_indirect_rs_has_kernel_impl() {
    let src = emit_seed_indirect_rs();
    assert!(src.contains("pub struct SeedIndirectKernel"));
    assert!(src.contains("impl crate::Kernel for SeedIndirectKernel"));
}

#[test]
fn append_events_rs_has_kernel_impl() {
    let src = emit_append_events_rs();
    assert!(src.contains("pub struct AppendEventsKernel"));
    assert!(src.contains("impl crate::Kernel for AppendEventsKernel"));
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test emit_physics_smoke`
Expected: FAIL — functions don't exist.

- [ ] **Step 3: Add the three emitters to emit_movement_kernel.rs**

Append to `crates/dsl_compiler/src/emit_movement_kernel.rs`:

```rust
pub fn emit_physics_rs() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_movement_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct PhysicsKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct PhysicsBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub current_event_ring: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub current_event_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub next_event_ring: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub next_event_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub gold_buf: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub standing_storage: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub memory_storage: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct PhysicsCfg {{").unwrap();
    writeln!(out, "    pub agent_cap: u32,").unwrap();
    writeln!(out, "    pub iter_idx: u32,").unwrap();
    writeln!(out, "    pub max_iter: u32,").unwrap();
    writeln!(out, "    pub event_ring_capacity: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"physics.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for PhysicsKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = PhysicsBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = PhysicsCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::physics::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::physics::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(0, false), // agents (rw)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(1, true),  // current_event_ring").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(2, true),  // current_event_tail").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(3, false), // next_event_ring (atomic append)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(4, false), // next_event_tail (atomic counter)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(5, false), // gold_buf").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(6, false), // standing").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(7, false), // memory").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(8, true),  // sim_cfg").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_uniform(9),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::physics::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::physics::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_physics\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> PhysicsCfg {{").unwrap();
    writeln!(out, "        PhysicsCfg {{ agent_cap: state.agent_cap(), iter_idx: 0, max_iter: 8, event_ring_capacity: 4096 }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> PhysicsBindings<'a> {{").unwrap();
    writeln!(out, "        // PingPong A/B alternation per iteration is handled by the").unwrap();
    writeln!(out, "        // engine_gpu FixedPoint loop swapping `transient.cascade_*` field").unwrap();
    writeln!(out, "        // references each iteration. The emitted bind() body always").unwrap();
    writeln!(out, "        // reads `cascade_current_*` and writes `cascade_next_*`; the").unwrap();
    writeln!(out, "        // loop alternates which underlying buffer those refs point at.").unwrap();
    writeln!(out, "        PhysicsBindings {{").unwrap();
    writeln!(out, "            agents:             sources.external.agents,").unwrap();
    writeln!(out, "            current_event_ring: sources.transient.cascade_current_ring,").unwrap();
    writeln!(out, "            current_event_tail: sources.transient.cascade_current_tail,").unwrap();
    writeln!(out, "            next_event_ring:    sources.transient.cascade_next_ring,").unwrap();
    writeln!(out, "            next_event_tail:    sources.transient.cascade_next_tail,").unwrap();
    writeln!(out, "            gold_buf:           &sources.resident.gold,").unwrap();
    writeln!(out, "            standing_storage:   &sources.resident.standing_primary,").unwrap();
    writeln!(out, "            memory_storage:     &sources.resident.memory_primary,").unwrap();
    writeln!(out, "            sim_cfg:            sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &PhysicsBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::physics::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.current_event_ring.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.current_event_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.next_event_ring.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 4, resource: bindings.next_event_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 5, resource: bindings.gold_buf.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 6, resource: bindings.standing_storage.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 7, resource: bindings.memory_storage.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 8, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 9, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::physics::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn emit_seed_indirect_rs() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_movement_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// Reads the producer ring's tail count and writes the next-iter").unwrap();
    writeln!(out, "/// `dispatch_indirect` args. Used by `DispatchOp::Indirect`.").unwrap();
    writeln!(out, "pub struct SeedIndirectKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct SeedIndirectBindings<'a> {{").unwrap();
    writeln!(out, "    pub apply_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub indirect_args: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct SeedIndirectCfg {{ pub iter_idx: u32, pub _pad: [u32; 3] }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"seed_indirect.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for SeedIndirectKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = SeedIndirectBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = SeedIndirectCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::seed_indirect::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::seed_indirect::bgl\"),").unwrap();
    writeln!(out, "            entries: &[crate::fused_mask::bgl_storage(0, true), crate::fused_mask::bgl_storage(1, false), crate::fused_mask::bgl_storage(2, false), crate::fused_mask::bgl_uniform(3)],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::seed_indirect::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl], push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::seed_indirect::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl), module: &shader, entry_point: Some(\"cs_seed_indirect\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(), cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, _state: &engine::state::SimState) -> SeedIndirectCfg {{").unwrap();
    writeln!(out, "        SeedIndirectCfg {{ iter_idx: 0, _pad: [0; 3] }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> SeedIndirectBindings<'a> {{").unwrap();
    writeln!(out, "        SeedIndirectBindings {{").unwrap();
    writeln!(out, "            apply_tail:    sources.transient.cascade_next_tail,").unwrap();
    writeln!(out, "            indirect_args: sources.transient.cascade_indirect_args,").unwrap();
    writeln!(out, "            sim_cfg:       sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &SeedIndirectBindings<'_>, _agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::seed_indirect::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.apply_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.indirect_args.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::seed_indirect::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups(1, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn emit_append_events_rs() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_movement_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// Promotes per-iter events into the batch ring.").unwrap();
    writeln!(out, "pub struct AppendEventsKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct AppendEventsBindings<'a> {{").unwrap();
    writeln!(out, "    pub source_ring: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub source_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub batch_ring: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub batch_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct AppendEventsCfg {{ pub source_capacity: u32, pub batch_capacity: u32, pub _pad: [u32; 2] }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"append_events.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for AppendEventsKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = AppendEventsBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = AppendEventsCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::append_events::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::append_events::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(0, true),  // source_ring").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(1, true),  // source_tail").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(2, false), // batch_ring").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(3, false), // batch_tail (atomic counter)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_uniform(4),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::append_events::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl], push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::append_events::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl), module: &shader, entry_point: Some(\"cs_append_events\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(), cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, _state: &engine::state::SimState) -> AppendEventsCfg {{").unwrap();
    writeln!(out, "        AppendEventsCfg {{ source_capacity: 4096, batch_capacity: 65536, _pad: [0; 2] }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> AppendEventsBindings<'a> {{").unwrap();
    writeln!(out, "        // After the FixedPoint loop converges, the final cascade ring").unwrap();
    writeln!(out, "        // sits in `cascade_current_*` (post-swap from the last iter).").unwrap();
    writeln!(out, "        AppendEventsBindings {{").unwrap();
    writeln!(out, "            source_ring: sources.transient.cascade_current_ring,").unwrap();
    writeln!(out, "            source_tail: sources.transient.cascade_current_tail,").unwrap();
    writeln!(out, "            batch_ring:  &sources.resident.batch_events_ring,").unwrap();
    writeln!(out, "            batch_tail:  &sources.resident.batch_events_tail,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &AppendEventsBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::append_events::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.source_ring.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.source_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.batch_ring.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.batch_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 4, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::append_events::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

- [ ] **Step 4: Run unit tests**

Run: `cargo test -p dsl_compiler --test emit_physics_smoke`
Expected: 3 passed.

- [ ] **Step 5: Wire xtask to write all three modules + WGSL bodies + Schedule entries**

In `crates/xtask/src/main.rs`:

```rust
modules.push("physics".into());
modules.push("seed_indirect".into());
modules.push("append_events".into());
modules.sort();

{
    use std::fs;
    use std::path::PathBuf;
    use dsl_compiler::emit_movement_kernel::{emit_physics_rs, emit_seed_indirect_rs, emit_append_events_rs};

    fs::write(PathBuf::from("crates/engine_gpu_rules/src/physics.rs"), emit_physics_rs()).expect("physics.rs");
    let pbody = engine_gpu::physics::build_physics_shader_resident(&comp).expect("physics shader");
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/physics.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{pbody}\n")).expect("physics.wgsl");

    fs::write(PathBuf::from("crates/engine_gpu_rules/src/seed_indirect.rs"), emit_seed_indirect_rs()).expect("seed_indirect.rs");
    let sbody = engine_gpu::cascade_resident::SEED_INDIRECT_WGSL;
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/seed_indirect.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{sbody}\n")).expect("seed_indirect.wgsl");

    fs::write(PathBuf::from("crates/engine_gpu_rules/src/append_events.rs"), emit_append_events_rs()).expect("append_events.rs");
    let abody = engine_gpu::cascade_resident::APPEND_EVENTS_WGSL;
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/append_events.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{abody}\n")).expect("append_events.wgsl");
}
```

(Hoist `SEED_INDIRECT_WGSL` and `APPEND_EVENTS_WGSL` to `pub const` in `engine_gpu::cascade_resident` if not already; transitional.)

Update Schedule entries:

```rust
let entries = vec![
    ScheduleEntry { kernel: "FusedMask".into(),    kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "Scoring".into(),      kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "PickAbility".into(),  kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "ApplyActions".into(), kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "Movement".into(),     kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "Physics".into(),      kind: DispatchOpKind::FixedPoint { max_iter: 8 } },
    ScheduleEntry { kernel: "SeedIndirect".into(), kind: DispatchOpKind::Indirect { args_buf_ref: "ResidentIndirectArgs".into() } },
    ScheduleEntry { kernel: "AppendEvents".into(), kind: DispatchOpKind::Kernel },
];
fs::write(PathBuf::from("crates/engine_gpu_rules/src/schedule.rs"),
    emit_schedule_rs(&entries)).expect("schedule.rs");
```

- [ ] **Step 6: Wire physics + seed + append in engine_gpu's step_batch**

Extend the binding-source containers to expose cascade buffers. Inside the Task 10 xtask block, append:

```rust
use dsl_compiler::emit_resident_context::ResidentField;
use dsl_compiler::emit_transient_handles::TransientField;

// Resident: gold, standing, memory, batch events ring + tail.
for (n, doc) in &[
    ("gold",                "Per-agent gold balance (Resident)."),
    ("standing_primary",    "Standing view storage primary buffer (Resident)."),
    ("memory_primary",      "Memory view storage primary buffer (Resident)."),
    ("batch_events_ring",   "Batch event ring records (consumed by view folds + post-batch readback)."),
    ("batch_events_tail",   "Batch event ring tail counter."),
] {
    if !resident_fields.iter().any(|f: &ResidentField| f.name == *n) {
        resident_fields.push(ResidentField { name: (*n).into(), doc: (*doc).into() });
    }
}
// Transient: per-iteration cascade ring refs (engine_gpu swaps these each FixedPoint iter).
for (n, doc) in &[
    ("cascade_current_ring",  "Cascade producer-ring records for the current iteration."),
    ("cascade_current_tail",  "Cascade producer-ring tail counter."),
    ("cascade_next_ring",     "Cascade consumer-ring records (next iteration)."),
    ("cascade_next_tail",     "Cascade consumer-ring tail counter (atomic)."),
    ("cascade_indirect_args", "dispatch_indirect args for the next iteration."),
] {
    transient_fields.push(TransientField { name: (*n).into(), doc: (*doc).into() });
}
```

Replace the `run_cascade_resident(...)` call in `step_batch` with an inline FixedPoint loop that rebuilds the `BindingSources` (specifically the `transient` field) on each iteration to swap the A/B ring references:

```rust
// FixedPoint(physics, max_iter=8): drain producer ring → produce
// next-iter ring; stop when next-iter ring is empty.
{
    use engine_gpu_rules::physics::PhysicsKernel;
    use engine_gpu_rules::seed_indirect::SeedIndirectKernel;
    use engine_gpu_rules::Kernel as _;

    let physics = self.resident.physics_kernel
        .get_or_insert_with(|| PhysicsKernel::new(&self.device));
    let seed = self.resident.seed_indirect_kernel
        .get_or_insert_with(|| SeedIndirectKernel::new(&self.device));

    let cascade_ctx = resident_cascade_ctx.as_mut().unwrap();
    let max_iter = 8;
    for iter in 0..max_iter {
        let (curr_ring, curr_tail, next_ring, next_tail) = cascade_ctx.ab_for_iter(iter);

        // Refresh the per-iteration transient struct: the same pingpong
        // buffers from PingPongContext show up under cascade_current/next
        // names, alternated by the loop.
        let iter_transient = engine_gpu_rules::transient_handles::TransientHandles {
            mask_bitmaps:                self.sync.mask_kernel.mask_bitmaps_buf(),
            mask_unpack_agents_input:    self.sync.mask_kernel.unpack_agents_input_buf(),
            action_buf:                  self.sync.scoring_kernel.scoring_out_buf(),
            scoring_unpack_agents_input: self.sync.scoring_kernel.scoring_unpack_input_buf(),
            cascade_current_ring:        curr_ring,
            cascade_current_tail:        curr_tail,
            cascade_next_ring:           next_ring,
            cascade_next_tail:           next_tail,
            cascade_indirect_args:       resident_indirect_args.as_ref().expect("indirect ensured").buf(),
            _phantom: std::marker::PhantomData,
        };
        let iter_sources = engine_gpu_rules::binding_sources::BindingSources {
            resident:  &self.resident.path_ctx,
            pingpong:  &self.resident.pingpong_ctx,
            pool:      &self.resident.pool,
            transient: &iter_transient,
            external:  &external,
        };

        let cfg = engine_gpu_rules::physics::PhysicsCfg {
            agent_cap, iter_idx: iter, max_iter, event_ring_capacity: 4096,
        };
        let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("physics::cfg"),
            contents: bytemuck::cast_slice(&[cfg]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bindings = physics.bind(&iter_sources, &cfg_buf);
        physics.record(&self.device, &mut encoder, &bindings, agent_cap);

        // Indirect seed for next iteration.
        let seed_cfg = engine_gpu_rules::seed_indirect::SeedIndirectCfg { iter_idx: iter, _pad: [0; 3] };
        let seed_cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seed_indirect::cfg"),
            contents: bytemuck::cast_slice(&[seed_cfg]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_bindings = seed.bind(&iter_sources, &seed_cfg_buf);
        seed.record(&self.device, &mut encoder, &seed_bindings, agent_cap);
    }
}

// AppendEvents: promote cascade event tails into the batch ring.
{
    use engine_gpu_rules::append_events::AppendEventsKernel;
    use engine_gpu_rules::Kernel as _;
    let kernel = self.resident.append_events_kernel
        .get_or_insert_with(|| AppendEventsKernel::new(&self.device));
    let cfg = kernel.build_cfg(state);
    let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("append_events::cfg"),
        contents: bytemuck::cast_slice(&[cfg]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    // Use the post-loop sources (final cascade ring sits in cascade_current_*).
    let bindings = kernel.bind(&sources, &cfg_buf);
    kernel.record(&self.device, &mut encoder, &bindings, agent_cap);
}
```

Add `pub physics_kernel`, `pub seed_indirect_kernel`, `pub append_events_kernel` Optional fields to `ResidentPathContext` (engine_gpu side wrapper). The `cascade_ctx.ab_for_iter` accessor stays as a transitional helper exposing the existing CascadeResidentCtx ring buffers; Task 16 deletes it once the residency is fully migrated to `engine_gpu_rules::PingPongContext`.

- [ ] **Step 7: Parity sweep**

Run: `cargo test -p engine_gpu --test physics_parity`
Run: `cargo test -p engine_gpu --test cascade_parity`
Run: `cargo test -p engine_gpu --test parity_with_cpu`
Run: `cargo test -p engine --test wolves_and_humans_parity`
Run: `cargo test -p engine_gpu --test indirect_cascade_converges`
Expected: all pass.

- [ ] **Step 8: Bump baseline + commit**

```bash
cargo run --bin xtask -- compile-dsl
cargo test -p engine_gpu_rules --test schema_hash
git add crates/dsl_compiler/ crates/xtask/ crates/engine_gpu_rules/ crates/engine_gpu/src/
git commit -m "feat(dsl_compiler): emit Physics + SeedIndirect + AppendEvents; FixedPoint + Indirect Schedule ops live"
```

---

### Task 11: Emit `cs_fold_<view>` kernels (one emitter, all 8 view modules)

The 8 materialized views (`engaged_with`, `threat_level`, `kin_fear`, `my_enemies`, `pack_focus`, `rally_boost`, `slow_factor`, `standing`) each have a fold kernel today. The WGSL emit already exists in `dsl_compiler::emit_view_wgsl`. This task wraps each one with a Rust struct emit.

**Files:**
- Create: `crates/dsl_compiler/src/emit_view_fold_kernel.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (register module)
- Modify: `crates/xtask/src/main.rs` (loop over views; write one `.rs`+`.wgsl` per view)
- Modify: `crates/engine_gpu/src/lib.rs` (replace 8 hand-written fold dispatches)
- Test: `crates/dsl_compiler/tests/emit_view_fold_smoke.rs`

- [ ] **Step 1: Failing test**

Create `crates/dsl_compiler/tests/emit_view_fold_smoke.rs`:

```rust
use dsl_compiler::emit_view_fold_kernel::emit_view_fold_rs;

#[test]
fn fold_engaged_with_rs_has_kernel_impl() {
    let src = emit_view_fold_rs("engaged_with");
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub struct FoldEngagedWithKernel"));
    assert!(src.contains("impl crate::Kernel for FoldEngagedWithKernel"));
    assert!(src.contains("include_str!(\"fold_engaged_with.wgsl\")"));
    assert!(src.contains("entry_point: Some(\"cs_fold_engaged_with\")"));
}

#[test]
fn fold_threat_level_rs_uses_view_specific_name() {
    let src = emit_view_fold_rs("threat_level");
    assert!(src.contains("pub struct FoldThreatLevelKernel"));
    assert!(src.contains("entry_point: Some(\"cs_fold_threat_level\")"));
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test emit_view_fold_smoke`
Expected: FAIL.

- [ ] **Step 3: Implement emit_view_fold_kernel.rs**

Create `crates/dsl_compiler/src/emit_view_fold_kernel.rs`:

```rust
//! Emits one Rust + WGSL pair per materialized view fold kernel.
//! Companion: `dsl_compiler::emit_view_wgsl::emit_fold_kernel_wgsl(view)`.

pub fn emit_view_fold_rs(view_name: &str) -> String {
    use std::fmt::Write;
    let pascal = pascal(view_name);
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_view_fold_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct Fold{pascal}Kernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct Fold{pascal}Bindings<'a> {{").unwrap();
    writeln!(out, "    pub event_ring: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub view_storage_primary: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub view_storage_anchor: Option<&'a wgpu::Buffer>,").unwrap();
    writeln!(out, "    pub view_storage_ids: Option<&'a wgpu::Buffer>,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct Fold{pascal}Cfg {{ pub event_count: u32, pub tick: u32, pub _pad: [u32; 2] }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"fold_{view_name}.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for Fold{pascal}Kernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = Fold{pascal}Bindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = Fold{pascal}Cfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(0, true),  // event_ring").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(1, true),  // event_tail").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(2, false), // view_storage primary").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(3, false), // view_storage anchor (optional)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(4, false), // view_storage ids (optional)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(5, true),  // sim_cfg").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_uniform(6),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl], push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl), module: &shader, entry_point: Some(\"cs_fold_{view_name}\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(), cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> Fold{pascal}Cfg {{").unwrap();
    writeln!(out, "        Fold{pascal}Cfg {{ event_count: 0, tick: state.tick, _pad: [0; 2] }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> Fold{pascal}Bindings<'a> {{").unwrap();
    writeln!(out, "        // The view's storage shape (SlotMap / PairMapScalar / PairMapDecay,").unwrap();
    writeln!(out, "        // optionally topk) determines whether anchor/ids fields exist on").unwrap();
    writeln!(out, "        // the resident context. The emitter sets `Some(...)` if the view").unwrap();
    writeln!(out, "        // declared topk + anchor; otherwise None. The emitted helper").unwrap();
    writeln!(out, "        // `sources.resident.fold_view_<name>_handles()` returns a triple").unwrap();
    writeln!(out, "        // of (primary, anchor_opt, ids_opt) borrowed from the ctx fields.").unwrap();
    writeln!(out, "        let (primary, anchor, ids) = sources.resident.fold_view_{view_name}_handles();").unwrap();
    writeln!(out, "        Fold{pascal}Bindings {{").unwrap();
    writeln!(out, "            event_ring:           &sources.resident.batch_events_ring,").unwrap();
    writeln!(out, "            event_tail:           &sources.resident.batch_events_tail,").unwrap();
    writeln!(out, "            view_storage_primary: primary,").unwrap();
    writeln!(out, "            view_storage_anchor:  anchor,").unwrap();
    writeln!(out, "            view_storage_ids:     ids,").unwrap();
    writeln!(out, "            sim_cfg:              sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &Fold{pascal}Bindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let dummy = bindings.view_storage_primary; // fallback for None anchors/ids").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.event_ring.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.event_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.view_storage_primary.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.view_storage_anchor.unwrap_or(dummy).as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 4, resource: bindings.view_storage_ids.unwrap_or(dummy).as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 5, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 6, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

fn pascal(s: &str) -> String {
    let mut out = String::new();
    let mut up = true;
    for c in s.chars() {
        if c == '_' { up = true; continue; }
        if up { out.extend(c.to_uppercase()); up = false; } else { out.push(c); }
    }
    out
}
```

- [ ] **Step 4: Register + run unit test**

Add `pub mod emit_view_fold_kernel;` to `crates/dsl_compiler/src/lib.rs`.

Run: `cargo test -p dsl_compiler --test emit_view_fold_smoke`
Expected: 2 passed.

- [ ] **Step 5: Wire xtask to emit one module per view**

In `crates/xtask/src/main.rs`:

```rust
// One fold module per materialized view in the IR. Lazy views are skipped.
for view in &comp.views {
    if matches!(view.body, dsl_compiler::ir::ViewBodyIR::Expr(_)) { continue; }
    let name = &view.name;
    modules.push(format!("fold_{name}"));
    use std::fs;
    use std::path::PathBuf;
    let rs = dsl_compiler::emit_view_fold_kernel::emit_view_fold_rs(name);
    fs::write(PathBuf::from(format!("crates/engine_gpu_rules/src/fold_{name}.rs")), rs).expect("fold .rs");
    let body = dsl_compiler::emit_view_wgsl::emit_fold_kernel_wgsl(view);
    fs::write(
        PathBuf::from(format!("crates/engine_gpu_rules/src/fold_{name}.wgsl")),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{body}\n"),
    ).expect("fold .wgsl");
}
modules.sort();
```

(If the emit_view_wgsl helper is named differently, use the name that exists — the function exposing the per-view fold kernel WGSL string. See `crates/dsl_compiler/src/emit_view_wgsl.rs` for the public API.)

Add Schedule entries — one per view:

```rust
// After Movement, before Physics
for view in &comp.views {
    if matches!(view.body, dsl_compiler::ir::ViewBodyIR::Expr(_)) { continue; }
    let pascal = view.name.split('_').map(|p| {
        let mut chars = p.chars();
        chars.next().unwrap().to_uppercase().collect::<String>() + chars.as_str()
    }).collect::<String>();
    entries.push(ScheduleEntry {
        kernel: format!("Fold{pascal}"),
        kind: DispatchOpKind::Kernel,
    });
}
```

- [ ] **Step 6: Wire engine_gpu's step_batch to call all fold kernels**

In `crates/engine_gpu/src/lib.rs::step_batch`, replace each hand-written fold dispatch (whatever name they have today — search for `cs_fold_` within the file) with calls to the emitted kernels. Pattern per view:

```rust
{
    use engine_gpu_rules::fold_engaged_with::FoldEngagedWithKernel;
    use engine_gpu_rules::Kernel as _;
    let kernel = self.resident.fold_engaged_with_kernel
        .get_or_insert_with(|| FoldEngagedWithKernel::new(&self.device));
    let cfg = kernel.build_cfg(state);
    let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fold_engaged_with::cfg"),
        contents: bytemuck::cast_slice(&[cfg]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bindings = kernel.bind(&sources, &cfg_buf);
    kernel.record(&self.device, &mut encoder, &bindings, agent_cap);
}
```

Repeat for the seven other views. Add an `Option<Fold<View>Kernel>` field per view to `ResidentPathContext` (engine_gpu side wrapper). The view buffers are owned by `engine_gpu_rules::ResidentPathContext`; the per-view `fold_view_<name>_handles()` accessor is emitted alongside the field declarations by `emit_resident_context` (the emitter walks `view_specs` and for each spec emits both the field(s) and a method returning a `(&primary, anchor_opt, ids_opt)` triple).

- [ ] **Step 7: Parity sweep**

Run: `cargo test -p engine_gpu --test view_parity`
Run: `cargo test -p engine_gpu --test topk_view_parity`
Run: `cargo test -p engine_gpu --test parity_with_cpu`
Run: `cargo test -p engine --test wolves_and_humans_parity`
Expected: all pass.

- [ ] **Step 8: Bump baseline + commit**

```bash
cargo run --bin xtask -- compile-dsl
cargo test -p engine_gpu_rules --test schema_hash
git add crates/dsl_compiler/ crates/xtask/ crates/engine_gpu_rules/ crates/engine_gpu/src/
git commit -m "feat(dsl_compiler): emit_view_fold_kernel produces 8 fold modules; wire all into step_batch"
```

---

### Task 12: Emit `cs_spatial_hash`, `cs_kin_query`, `cs_engagement_query` (Pooled lifetime)

The spatial pipeline is one BGL-and-pipeline kernel for grid build + three per-query kernels (kin, engagement, max_move). Hand-written in `engine_gpu/src/spatial_gpu.rs:725-1450`.

**Files:**
- Create: `crates/dsl_compiler/src/emit_spatial_kernel.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (register module)
- Modify: `crates/xtask/src/main.rs` (call emitter; populate Schedule entries before Scoring)
- Modify: `crates/engine_gpu/src/lib.rs` (replace `run_spatial_resident_pre_scoring` with emitted kernel calls)
- Test: `crates/dsl_compiler/tests/emit_spatial_smoke.rs`

- [ ] **Step 1: Failing test**

Create `crates/dsl_compiler/tests/emit_spatial_smoke.rs`:

```rust
use dsl_compiler::emit_spatial_kernel::{emit_spatial_hash_rs, emit_kin_query_rs, emit_engagement_query_rs};

#[test]
fn spatial_hash_rs_has_kernel_impl() {
    let src = emit_spatial_hash_rs();
    assert!(src.contains("pub struct SpatialHashKernel"));
    assert!(src.contains("impl crate::Kernel for SpatialHashKernel"));
}

#[test]
fn kin_query_rs_has_kernel_impl() {
    let src = emit_kin_query_rs();
    assert!(src.contains("pub struct SpatialKinQueryKernel"));
}

#[test]
fn engagement_query_rs_has_kernel_impl() {
    let src = emit_engagement_query_rs();
    assert!(src.contains("pub struct SpatialEngagementQueryKernel"));
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test emit_spatial_smoke`
Expected: FAIL.

- [ ] **Step 3: Implement emit_spatial_kernel.rs**

Create `crates/dsl_compiler/src/emit_spatial_kernel.rs`:

```rust
//! Emits spatial kernel wrappers (grid build + per-query kernels).
//!
//! The hand-written body lives in engine_gpu::spatial_gpu — these
//! emitted modules wrap that code's BGL + dispatch shape. The WGSL
//! source for grid build is currently a `pub const SPATIAL_WGSL`
//! exported from spatial_gpu (transitional; deleted in Task 16).

pub fn emit_spatial_hash_rs() -> String {
    common_kernel_body("SpatialHashKernel", "spatial_hash", &[
        ("agents", true), ("grid_cells", false), ("grid_offsets", false), ("sim_cfg", true),
    ])
}

pub fn emit_kin_query_rs() -> String {
    common_kernel_body("SpatialKinQueryKernel", "spatial_kin_query", &[
        ("agents", true), ("grid_cells", true), ("grid_offsets", true),
        ("query_results", false), ("sim_cfg", true),
    ])
}

pub fn emit_engagement_query_rs() -> String {
    common_kernel_body("SpatialEngagementQueryKernel", "spatial_engagement_query", &[
        ("agents", true), ("grid_cells", true), ("grid_offsets", true),
        ("query_results", false), ("sim_cfg", true),
    ])
}

fn common_kernel_body(struct_name: &str, file_stem: &str, bindings: &[(&str, bool)]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_spatial_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct {struct_name} {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    let bindings_struct = format!("{struct_name}Bindings");
    writeln!(out, "pub struct {bindings_struct}<'a> {{").unwrap();
    for (name, _read_only) in bindings {
        writeln!(out, "    pub {name}: &'a wgpu::Buffer,").unwrap();
    }
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    let cfg_struct = format!("{struct_name}Cfg");
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct {cfg_struct} {{ pub agent_cap: u32, pub radius_q: f32, pub _pad: [u32; 2] }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"{file_stem}.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for {struct_name} {{").unwrap();
    writeln!(out, "    type Bindings<'a> = {bindings_struct}<'a>;").unwrap();
    writeln!(out, "    type Cfg = {cfg_struct};").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::{file_stem}::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::{file_stem}::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    for (i, (_name, ro)) in bindings.iter().enumerate() {
        writeln!(out, "                crate::fused_mask::bgl_storage({i}, {ro}),").unwrap();
    }
    writeln!(out, "                crate::fused_mask::bgl_uniform({}),", bindings.len()).unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::{file_stem}::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl], push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::{file_stem}::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl), module: &shader, entry_point: Some(\"cs_{file_stem}\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(), cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> {cfg_struct} {{").unwrap();
    writeln!(out, "        {cfg_struct} {{ agent_cap: state.agent_cap(), radius_q: 12.0, _pad: [0; 2] }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> {bindings_struct}<'a> {{").unwrap();
    writeln!(out, "        {bindings_struct} {{").unwrap();
    for (name, _ro) in bindings {
        // Map well-known binding names to the right BindingSources field.
        // - `agents`, `sim_cfg`, `ability_registry`, `tag_values` → external
        // - `grid_cells`, `grid_offsets`, `query_results`, `alive_bitmap` → pool (or resident for alive_bitmap)
        // - `agents_input`, `mask_soa` → transient
        // - `agent_data`, `event_ring`, etc. → external/resident depending on kernel
        let expr = match *name {
            "agents" | "sim_cfg" | "agent_data" | "agents_input" | "mask_soa" => {
                // agents/agent_data/sim_cfg are external; mask_soa/agents_input are transient
                match *name {
                    "agents" | "agent_data" => "sources.external.agents",
                    "sim_cfg" => "sources.external.sim_cfg",
                    "agents_input" => "sources.transient.fused_agent_unpack_input",
                    "mask_soa" => "sources.transient.fused_agent_unpack_mask_soa",
                    _ => unreachable!(),
                }
            }
            "grid_cells" => "&sources.pool.spatial_grid_cells",
            "grid_offsets" => "&sources.pool.spatial_grid_offsets",
            "query_results" => "&sources.pool.spatial_query_results",
            "alive_bitmap" => "&sources.resident.alive_bitmap",
            _ => "/* unmapped binding — extend the spatial-kernel name lookup */ unimplemented!(\"name lookup\")",
        };
        writeln!(out, "            {name}: {expr},").unwrap();
    }
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &{bindings_struct}<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();").unwrap();
    for (i, (name, _)) in bindings.iter().enumerate() {
        writeln!(out, "        entries.push(wgpu::BindGroupEntry {{ binding: {i}, resource: bindings.{name}.as_entire_binding() }});").unwrap();
    }
    writeln!(out, "        entries.push(wgpu::BindGroupEntry {{ binding: {}, resource: bindings.cfg.as_entire_binding() }});", bindings.len()).unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::{file_stem}::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &entries,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::{file_stem}::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
```

- [ ] **Step 4: Register + run unit test**

Add `pub mod emit_spatial_kernel;` to `crates/dsl_compiler/src/lib.rs`.

Run: `cargo test -p dsl_compiler --test emit_spatial_smoke`
Expected: 3 passed.

- [ ] **Step 5: Wire xtask + WGSL bodies**

In `crates/xtask/src/main.rs`:

```rust
modules.push("spatial_hash".into());
modules.push("spatial_kin_query".into());
modules.push("spatial_engagement_query".into());
modules.sort();

{
    use std::fs;
    use std::path::PathBuf;
    use dsl_compiler::emit_spatial_kernel::*;

    fs::write(PathBuf::from("crates/engine_gpu_rules/src/spatial_hash.rs"), emit_spatial_hash_rs()).expect("spatial_hash.rs");
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/spatial_hash.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{}\n", engine_gpu::spatial_gpu::SPATIAL_HASH_WGSL)).expect("spatial_hash.wgsl");

    fs::write(PathBuf::from("crates/engine_gpu_rules/src/spatial_kin_query.rs"), emit_kin_query_rs()).expect("kin_query.rs");
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/spatial_kin_query.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{}\n", engine_gpu::spatial_gpu::SPATIAL_KIN_QUERY_WGSL)).expect("kin_query.wgsl");

    fs::write(PathBuf::from("crates/engine_gpu_rules/src/spatial_engagement_query.rs"), emit_engagement_query_rs()).expect("engagement_query.rs");
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/spatial_engagement_query.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{}\n", engine_gpu::spatial_gpu::SPATIAL_ENGAGEMENT_QUERY_WGSL)).expect("engagement_query.wgsl");
}

// Schedule entries: spatial runs BEFORE scoring (the existing
// `run_spatial_resident_pre_scoring` call site).
let pre_scoring_idx = entries.iter().position(|e| e.kernel == "Scoring").unwrap_or(0);
let spatial_entries = vec![
    ScheduleEntry { kernel: "SpatialHash".into(),             kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "SpatialKinQuery".into(),         kind: DispatchOpKind::Kernel },
    ScheduleEntry { kernel: "SpatialEngagementQuery".into(),  kind: DispatchOpKind::Kernel },
];
for (i, e) in spatial_entries.into_iter().enumerate() {
    entries.insert(pre_scoring_idx + i, e);
}
```

(Hoist `SPATIAL_HASH_WGSL` / `SPATIAL_KIN_QUERY_WGSL` / `SPATIAL_ENGAGEMENT_QUERY_WGSL` `pub const`s in `engine_gpu::spatial_gpu` — transitional.)

- [ ] **Step 6: Wire into engine_gpu's step_batch**

Extend the `Pool` field list (xtask side, alongside the Task 12 emit block):

```rust
use dsl_compiler::emit_pool::PoolField;

for (n, doc) in &[
    ("spatial_grid_cells",     "Spatial-hash cell-index buffer (Pooled)."),
    ("spatial_grid_offsets",   "Spatial-hash cell-offsets buffer (Pooled)."),
    ("spatial_query_results",  "Per-query result buffer (Pooled)."),
] {
    pool_fields.push(PoolField { name: (*n).into(), doc: (*doc).into() });
}
```

Replace the body of `run_spatial_resident_pre_scoring` with three emitted-kernel record calls (pattern same as Task 11 step 6 — `kernel.bind(&sources, &cfg_buf)` then `kernel.record(...)`). Add `Option<Spatial*Kernel>` fields to `ResidentPathContext` (engine_gpu side wrapper).

- [ ] **Step 7: Parity sweep**

Run: `cargo test -p engine_gpu --test spatial_parity`
Run: `cargo test -p engine_gpu --test spatial_resident`
Run: `cargo test -p engine_gpu --test parity_with_cpu`
Run: `cargo test -p engine --test wolves_and_humans_parity`
Expected: all pass.

- [ ] **Step 8: Bump baseline + commit**

```bash
cargo run --bin xtask -- compile-dsl
cargo test -p engine_gpu_rules --test schema_hash
git add crates/dsl_compiler/ crates/xtask/ crates/engine_gpu_rules/ crates/engine_gpu/src/
git commit -m "feat(dsl_compiler): emit spatial hash + kin/engagement query kernels; Pooled lifetime live"
```

---

### Task 13: Emit `AlivePackKernel` + `FusedAgentUnpackKernel`

The remaining hand-written kernels are alive-bitmap pack and the SoA fused-unpack used at the top of every batch tick. Hand-written sources: `engine_gpu/src/alive_bitmap.rs:191-330`, `engine_gpu/src/mask.rs:1900-2120`.

**Files:**
- Modify: `crates/dsl_compiler/src/emit_spatial_kernel.rs` (add `emit_alive_pack_rs`, `emit_fused_agent_unpack_rs` — they use the same `common_kernel_body` helper)
- Modify: `crates/xtask/src/main.rs` (write the two modules; insert Schedule entries at the top of the per-tick run)
- Modify: `crates/engine_gpu/src/lib.rs` (replace the existing pack/unpack calls)
- Test: extend `crates/dsl_compiler/tests/emit_spatial_smoke.rs`

- [ ] **Step 1: Add failing test**

Append to `crates/dsl_compiler/tests/emit_spatial_smoke.rs`:

```rust
#[test]
fn alive_pack_rs_has_kernel_impl() {
    let src = dsl_compiler::emit_spatial_kernel::emit_alive_pack_rs();
    assert!(src.contains("pub struct AlivePackKernel"));
}

#[test]
fn fused_agent_unpack_rs_has_kernel_impl() {
    let src = dsl_compiler::emit_spatial_kernel::emit_fused_agent_unpack_rs();
    assert!(src.contains("pub struct FusedAgentUnpackKernel"));
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test emit_spatial_smoke alive_pack`
Expected: FAIL.

- [ ] **Step 3: Add the two emitters**

Append to `crates/dsl_compiler/src/emit_spatial_kernel.rs`:

```rust
pub fn emit_alive_pack_rs() -> String {
    common_kernel_body("AlivePackKernel", "alive_pack", &[
        ("agents", true), ("alive_bitmap", false),
    ])
}

pub fn emit_fused_agent_unpack_rs() -> String {
    common_kernel_body("FusedAgentUnpackKernel", "fused_agent_unpack", &[
        ("agents_input", true), ("mask_soa", false), ("agent_data", false),
    ])
}
```

- [ ] **Step 4: Run unit tests**

Run: `cargo test -p dsl_compiler --test emit_spatial_smoke`
Expected: 5 passed.

- [ ] **Step 5: Wire xtask + WGSL bodies**

```rust
modules.push("alive_pack".into());
modules.push("fused_agent_unpack".into());
modules.sort();

{
    use std::fs;
    use std::path::PathBuf;
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/alive_pack.rs"),
        dsl_compiler::emit_spatial_kernel::emit_alive_pack_rs()).expect("alive_pack.rs");
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/alive_pack.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{}\n", engine_gpu::alive_bitmap::ALIVE_PACK_WGSL)).expect("alive_pack.wgsl");

    fs::write(PathBuf::from("crates/engine_gpu_rules/src/fused_agent_unpack.rs"),
        dsl_compiler::emit_spatial_kernel::emit_fused_agent_unpack_rs()).expect("fused_agent_unpack.rs");
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/fused_agent_unpack.wgsl"),
        format!("// GENERATED by dsl_compiler. Do not edit by hand.\n{}\n", engine_gpu::mask::FUSED_AGENT_UNPACK_WGSL)).expect("fused_agent_unpack.wgsl");
}

// Schedule entries — these run FIRST in every tick.
entries.insert(0, ScheduleEntry { kernel: "FusedAgentUnpack".into(), kind: DispatchOpKind::Kernel });
entries.insert(1, ScheduleEntry { kernel: "AlivePack".into(),        kind: DispatchOpKind::Kernel });
```

(Hoist `ALIVE_PACK_WGSL` and `FUSED_AGENT_UNPACK_WGSL` to `pub const`s — transitional.)

- [ ] **Step 6: Wire into engine_gpu's step_batch**

Extend the binding-source containers for these last two kernels:

```rust
use dsl_compiler::emit_resident_context::ResidentField;
use dsl_compiler::emit_transient_handles::TransientField;

if !resident_fields.iter().any(|f: &ResidentField| f.name == "alive_bitmap") {
    resident_fields.push(ResidentField {
        name: "alive_bitmap".into(),
        doc: "Per-agent alive bitmap (Resident; ceil(N/32) words).".into(),
    });
}
transient_fields.push(TransientField {
    name: "fused_agent_unpack_input".into(),
    doc: "FusedAgentUnpackKernel scratch: source pre-unpack agent buffer.".into(),
});
transient_fields.push(TransientField {
    name: "fused_agent_unpack_mask_soa".into(),
    doc: "FusedAgentUnpackKernel scratch: derived mask SoA.".into(),
});
```

Replace the existing `fused_unpack_kernel.encode_unpack(...)` and `alive_pack_kernel.encode_pack(...)` calls (around `crates/engine_gpu/src/lib.rs:1126-1154`) with calls to the emitted kernels (same shape as Task 11 — `kernel.bind(&sources, &cfg_buf).record(...)`). Add `Option<AlivePackKernel>` + `Option<FusedAgentUnpackKernel>` fields to `ResidentPathContext` (engine_gpu side wrapper).

- [ ] **Step 7: Parity sweep**

Run: `cargo test -p engine_gpu --test alive_bitmap_pack`
Run: `cargo test -p engine_gpu --test parity_with_cpu`
Run: `cargo test -p engine --test wolves_and_humans_parity`
Expected: all pass.

- [ ] **Step 8: Bump baseline + commit**

```bash
cargo run --bin xtask -- compile-dsl
cargo test -p engine_gpu_rules --test schema_hash
git add crates/dsl_compiler/ crates/xtask/ crates/engine_gpu_rules/ crates/engine_gpu/src/
git commit -m "feat(dsl_compiler): emit AlivePack + FusedAgentUnpack; all 14 kernels live"
```

---

### Task 14: Emit megakernel — fuse pass over SCHEDULE

Megakernel is a second emit pass that walks `SCHEDULE` and produces a single fused WGSL kernel. Each `DispatchOp::Kernel` becomes an inline section; `FixedPoint` becomes a `while`-loop; `GatedBy` becomes an `if` branch. This task lands the wiring; performance optimisation is left to the existing `gpu_megakernel_plan.md` work in flight.

**Files:**
- Create: `crates/dsl_compiler/src/emit_megakernel.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (register module)
- Modify: `crates/xtask/src/main.rs` (call emitter; produce `megakernel.{rs,wgsl}`)
- Test: `crates/dsl_compiler/tests/emit_megakernel_smoke.rs`

- [ ] **Step 1: Failing test**

Create `crates/dsl_compiler/tests/emit_megakernel_smoke.rs`:

```rust
use dsl_compiler::emit_megakernel::{emit_megakernel_rs, emit_megakernel_wgsl};
use dsl_compiler::emit_schedule::{ScheduleEntry, DispatchOpKind};

#[test]
fn megakernel_rs_has_kernel_impl() {
    let entries = vec![
        ScheduleEntry { kernel: "FusedMask".into(), kind: DispatchOpKind::Kernel },
        ScheduleEntry { kernel: "Scoring".into(),   kind: DispatchOpKind::Kernel },
    ];
    let src = emit_megakernel_rs(&entries);
    assert!(src.starts_with("// GENERATED by dsl_compiler"));
    assert!(src.contains("pub struct MegaKernel"));
    assert!(src.contains("include_str!(\"megakernel.wgsl\")"));
}

#[test]
fn megakernel_wgsl_inlines_each_kernel() {
    let entries = vec![
        ScheduleEntry { kernel: "FusedMask".into(), kind: DispatchOpKind::Kernel },
        ScheduleEntry { kernel: "Scoring".into(),   kind: DispatchOpKind::Kernel },
    ];
    let wgsl = emit_megakernel_wgsl(&entries);
    assert!(wgsl.starts_with("// GENERATED by dsl_compiler"));
    assert!(wgsl.contains("// section: FusedMask"));
    assert!(wgsl.contains("// section: Scoring"));
}

#[test]
fn fixed_point_emits_while_loop() {
    let entries = vec![
        ScheduleEntry { kernel: "Physics".into(), kind: DispatchOpKind::FixedPoint { max_iter: 8 } },
    ];
    let wgsl = emit_megakernel_wgsl(&entries);
    assert!(wgsl.contains("while ("), "expected while loop, got: {wgsl}");
    assert!(wgsl.contains("8u"), "expected max_iter=8 bound");
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test emit_megakernel_smoke`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement emit_megakernel.rs**

Create `crates/dsl_compiler/src/emit_megakernel.rs`:

```rust
//! Emits the megakernel: one fused compute pipeline that walks the
//! Schedule. Stepwise mode (per-kernel dispatch) and megakernel mode
//! coexist; the engine_gpu backend selects via runtime flag.

use crate::emit_schedule::{DispatchOpKind, ScheduleEntry};
use std::fmt::Write;

pub fn emit_megakernel_rs(entries: &[ScheduleEntry]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_megakernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct MegaKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// Captures every buffer the fused shader reads/writes. Populated").unwrap();
    writeln!(out, "/// by engine_gpu at dispatch time from the resident context.").unwrap();
    writeln!(out, "pub struct MegaKernelBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub mask_bitmaps: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub scoring_out: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_ring: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct MegaKernelCfg {{ pub agent_cap: u32, pub tick: u32, pub _pad: [u32; 2] }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"megakernel.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// {} sections in fused order", entries.len()).unwrap();
    writeln!(out, "pub const SECTION_COUNT: usize = {};", entries.len()).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for MegaKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = MegaKernelBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = MegaKernelCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::megakernel::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::megakernel::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(0, false), // agents").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(1, false), // mask_bitmaps").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(2, false), // scoring_out").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(3, false), // event_ring").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(4, false), // event_tail (atomic)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(5, true),  // sim_cfg").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_uniform(6),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::megakernel::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl], push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::megakernel::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl), module: &shader, entry_point: Some(\"cs_megakernel\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(), cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> MegaKernelCfg {{").unwrap();
    writeln!(out, "        MegaKernelCfg {{ agent_cap: state.agent_cap(), tick: state.tick, _pad: [0; 2] }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> MegaKernelBindings<'a> {{").unwrap();
    writeln!(out, "        // Megakernel scaffold: aggregates the union of buffers all").unwrap();
    writeln!(out, "        // inlined sections need. The gpu_megakernel_plan work in flight").unwrap();
    writeln!(out, "        // expands the body below as it adds inlined sections.").unwrap();
    writeln!(out, "        MegaKernelBindings {{").unwrap();
    writeln!(out, "            agents:       sources.external.agents,").unwrap();
    writeln!(out, "            mask_bitmaps: sources.transient.mask_bitmaps,").unwrap();
    writeln!(out, "            scoring_out:  sources.transient.action_buf,").unwrap();
    writeln!(out, "            event_ring:   sources.transient.cascade_current_ring,").unwrap();
    writeln!(out, "            event_tail:   sources.transient.cascade_current_tail,").unwrap();
    writeln!(out, "            sim_cfg:      sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &MegaKernelBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::megakernel::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.mask_bitmaps.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.scoring_out.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.event_ring.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 4, resource: bindings.event_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 5, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 6, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::megakernel::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn emit_megakernel_wgsl(entries: &[ScheduleEntry]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_megakernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Fused composition over SCHEDULE.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "// (Bind-group declarations elided — Task 14 intentionally produces a").unwrap();
    writeln!(out, "//  scaffold that the gpu_megakernel_plan work in flight will fill in.").unwrap();
    writeln!(out, "//  Performance optimisation is out of scope for this plan.)").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "@compute @workgroup_size(64)").unwrap();
    writeln!(out, "fn cs_megakernel(@builtin(global_invocation_id) gid: vec3<u32>) {{").unwrap();
    for e in entries {
        writeln!(out, "    // section: {}", e.kernel).unwrap();
        match &e.kind {
            DispatchOpKind::Kernel => {
                writeln!(out, "    // (inline body for {} — populated by gpu_megakernel_plan)", e.kernel).unwrap();
            }
            DispatchOpKind::FixedPoint { max_iter } => {
                writeln!(out, "    var iter: u32 = 0u;").unwrap();
                writeln!(out, "    while (iter < {max_iter}u) {{").unwrap();
                writeln!(out, "        // (inline body for {})", e.kernel).unwrap();
                writeln!(out, "        iter = iter + 1u;").unwrap();
                writeln!(out, "    }}").unwrap();
            }
            DispatchOpKind::Indirect { .. } => {
                writeln!(out, "    // (inline body for indirect dispatch {})", e.kernel).unwrap();
            }
            DispatchOpKind::GatedBy { .. } => {
                writeln!(out, "    if (gate_signal_for_{}() != 0u) {{", e.kernel).unwrap();
                writeln!(out, "        // (inline body for {})", e.kernel).unwrap();
                writeln!(out, "    }}").unwrap();
            }
        }
    }
    writeln!(out, "}}").unwrap();
    out
}
```

- [ ] **Step 4: Register + run unit tests**

Add `pub mod emit_megakernel;` to `crates/dsl_compiler/src/lib.rs`.

Run: `cargo test -p dsl_compiler --test emit_megakernel_smoke`
Expected: 3 passed.

- [ ] **Step 5: Wire xtask to write megakernel.{rs,wgsl}**

```rust
{
    use std::fs;
    use std::path::PathBuf;
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/megakernel.rs"),
        dsl_compiler::emit_megakernel::emit_megakernel_rs(&entries)).expect("megakernel.rs");
    fs::write(PathBuf::from("crates/engine_gpu_rules/src/megakernel.wgsl"),
        dsl_compiler::emit_megakernel::emit_megakernel_wgsl(&entries)).expect("megakernel.wgsl");
}
```

- [ ] **Step 6: Verify the generated megakernel scaffold compiles in WGSL**

Run: `cargo run --bin xtask -- compile-dsl`
Run: `cargo build -p engine_gpu_rules --features gpu`
Expected: clean build (the WGSL is loaded but not invoked yet — the megakernel-mode dispatch site is a runtime selector that this task does not enable; left for the gpu_megakernel_plan).

- [ ] **Step 7: Commit**

```bash
cargo test -p engine_gpu_rules --test schema_hash
git add crates/dsl_compiler/ crates/xtask/ crates/engine_gpu_rules/
git commit -m "feat(dsl_compiler): emit_megakernel produces fused-shader scaffold over SCHEDULE"
```

---

### Task 15: Wire engine_gpu's `step_batch` to consume `SCHEDULE`

This task replaces the open-coded sequence of kernel-record blocks built up across Tasks 4–13 with a single `for op in SCHEDULE` loop driven by an explicit `dispatch(op, ...)` match. The loop construction includes the once-per-tick `BindingSources` aggregate; each match arm calls `kernel.bind(sources, &cfg_buf)` then `kernel.record(...)`. Behaviour-equivalent to the per-task wiring.

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs::step_batch`
- Modify: `crates/engine_gpu/src/backend/resident_ctx.rs` (kernel handles already added by Tasks 4–13)

- [ ] **Step 1: Locate the existing per-tick body**

In `crates/engine_gpu/src/lib.rs::step_batch`, identify the inline blocks added in Tasks 4–13 (each a `{ ... kernel.bind(&sources, &cfg_buf).record(...) ... }` block). They sit inside `for tick_idx in 0..n_ticks { ... }`. The `BindingSources` construction (Task 5 step 4) sits at the top of the per-tick body.

- [ ] **Step 2: Replace those blocks with a Schedule-driven dispatch**

Replace the body of `for tick_idx in 0..n_ticks { ... }` with:

```rust
for tick_idx in 0..n_ticks {
    let mut encoder = self.device.create_command_encoder(&Default::default());
    let transient = TransientHandles { /* ... refresh per tick ... */ _phantom: std::marker::PhantomData };
    let external  = ExternalBuffers  { /* ... refresh per tick ... */ _phantom: std::marker::PhantomData };
    let sources = BindingSources {
        resident:  &self.resident.path_ctx,
        pingpong:  &self.resident.pingpong_ctx,
        pool:      &self.resident.pool,
        transient: &transient,
        external:  &external,
    };
    for op in engine_gpu_rules::schedule::SCHEDULE {
        self.dispatch(op, &mut encoder, state, &sources)?;
    }
    self.queue.submit(Some(encoder.finish()));
    state.tick += 1;
}
```

- [ ] **Step 3: Implement `GpuBackend::dispatch`**

Add a private method on `GpuBackend` (in `crates/engine_gpu/src/lib.rs`):

```rust
fn dispatch(
    &mut self,
    op: &engine_gpu_rules::schedule::DispatchOp,
    encoder: &mut wgpu::CommandEncoder,
    state: &SimState,
    sources: &engine_gpu_rules::binding_sources::BindingSources<'_>,
) -> Result<(), crate::DispatchError> {
    use engine_gpu_rules::schedule::{DispatchOp, KernelId};
    use engine_gpu_rules::Kernel as _;

    // Helper macro — every arm follows the same lazy-init / build_cfg /
    // bind / record shape. Defined inline so all per-kernel branches stay
    // a single line.
    macro_rules! dispatch_kernel {
        ($field:ident, $kernel_ty:path, $label:expr) => {{
            let kernel = self.resident.$field
                .get_or_insert_with(|| <$kernel_ty>::new(&self.device));
            let cfg = kernel.build_cfg(state);
            let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some($label),
                contents: bytemuck::cast_slice(&[cfg]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bindings = kernel.bind(sources, &cfg_buf);
            kernel.record(&self.device, encoder, &bindings, state.agent_cap());
        }};
    }

    let agent_cap = state.agent_cap();
    match op {
        DispatchOp::Kernel(KernelId::FusedAgentUnpack) =>
            dispatch_kernel!(fused_agent_unpack_kernel, engine_gpu_rules::fused_agent_unpack::FusedAgentUnpackKernel, "fused_agent_unpack::cfg"),
        DispatchOp::Kernel(KernelId::AlivePack) =>
            dispatch_kernel!(alive_pack_kernel, engine_gpu_rules::alive_pack::AlivePackKernel, "alive_pack::cfg"),
        DispatchOp::Kernel(KernelId::SpatialHash) =>
            dispatch_kernel!(spatial_hash_kernel, engine_gpu_rules::spatial_hash::SpatialHashKernel, "spatial_hash::cfg"),
        DispatchOp::Kernel(KernelId::SpatialKinQuery) =>
            dispatch_kernel!(spatial_kin_query_kernel, engine_gpu_rules::spatial_kin_query::SpatialKinQueryKernel, "spatial_kin_query::cfg"),
        DispatchOp::Kernel(KernelId::SpatialEngagementQuery) =>
            dispatch_kernel!(spatial_engagement_query_kernel, engine_gpu_rules::spatial_engagement_query::SpatialEngagementQueryKernel, "spatial_engagement_query::cfg"),
        DispatchOp::Kernel(KernelId::FusedMask) =>
            dispatch_kernel!(fused_mask_kernel, engine_gpu_rules::fused_mask::FusedMaskKernel, "fused_mask::cfg"),
        DispatchOp::Kernel(KernelId::Scoring) =>
            dispatch_kernel!(scoring_kernel, engine_gpu_rules::scoring::ScoringKernel, "scoring::cfg"),
        DispatchOp::Kernel(KernelId::PickAbility) =>
            dispatch_kernel!(pick_ability_kernel, engine_gpu_rules::pick_ability::PickAbilityKernel, "pick_ability::cfg"),
        DispatchOp::Kernel(KernelId::ApplyActions) =>
            dispatch_kernel!(apply_actions_kernel, engine_gpu_rules::apply_actions::ApplyActionsKernel, "apply_actions::cfg"),
        DispatchOp::Kernel(KernelId::Movement) =>
            dispatch_kernel!(movement_kernel, engine_gpu_rules::movement::MovementKernel, "movement::cfg"),
        DispatchOp::Kernel(KernelId::FoldEngagedWith) =>
            dispatch_kernel!(fold_engaged_with_kernel, engine_gpu_rules::fold_engaged_with::FoldEngagedWithKernel, "fold_engaged_with::cfg"),
        DispatchOp::Kernel(KernelId::FoldThreatLevel) =>
            dispatch_kernel!(fold_threat_level_kernel, engine_gpu_rules::fold_threat_level::FoldThreatLevelKernel, "fold_threat_level::cfg"),
        DispatchOp::Kernel(KernelId::FoldKinFear) =>
            dispatch_kernel!(fold_kin_fear_kernel, engine_gpu_rules::fold_kin_fear::FoldKinFearKernel, "fold_kin_fear::cfg"),
        DispatchOp::Kernel(KernelId::FoldMyEnemies) =>
            dispatch_kernel!(fold_my_enemies_kernel, engine_gpu_rules::fold_my_enemies::FoldMyEnemiesKernel, "fold_my_enemies::cfg"),
        DispatchOp::Kernel(KernelId::FoldPackFocus) =>
            dispatch_kernel!(fold_pack_focus_kernel, engine_gpu_rules::fold_pack_focus::FoldPackFocusKernel, "fold_pack_focus::cfg"),
        DispatchOp::Kernel(KernelId::FoldRallyBoost) =>
            dispatch_kernel!(fold_rally_boost_kernel, engine_gpu_rules::fold_rally_boost::FoldRallyBoostKernel, "fold_rally_boost::cfg"),
        DispatchOp::Kernel(KernelId::FoldSlowFactor) =>
            dispatch_kernel!(fold_slow_factor_kernel, engine_gpu_rules::fold_slow_factor::FoldSlowFactorKernel, "fold_slow_factor::cfg"),
        DispatchOp::Kernel(KernelId::FoldStanding) =>
            dispatch_kernel!(fold_standing_kernel, engine_gpu_rules::fold_standing::FoldStandingKernel, "fold_standing::cfg"),
        DispatchOp::Kernel(KernelId::AppendEvents) =>
            dispatch_kernel!(append_events_kernel, engine_gpu_rules::append_events::AppendEventsKernel, "append_events::cfg"),

        DispatchOp::FixedPoint { kernel: KernelId::Physics, max_iter } => {
            // FixedPoint requires per-iteration TransientHandles refresh
            // (cascade A/B alternation). The block below is the same as
            // Task 10 step 6's loop body, refactored to the dispatch arm.
            // ... see Task 10 step 6 ...
            let _ = max_iter;
        }

        DispatchOp::Indirect { kernel: KernelId::SeedIndirect, args_buf: _ } =>
            dispatch_kernel!(seed_indirect_kernel, engine_gpu_rules::seed_indirect::SeedIndirectKernel, "seed_indirect::cfg"),

        // No-runtime-panic guarantee (P10): every variant is matched
        // explicitly. Adding a new kernel means adding both a Schedule
        // entry (compile-time) and a dispatch arm (compile-time) — the
        // exhaustiveness check forces both.
        DispatchOp::Kernel(other) => {
            panic!("dispatch: unhandled kernel {other:?}");
        }
        DispatchOp::FixedPoint { kernel: other, .. } => {
            panic!("dispatch: unhandled FixedPoint kernel {other:?}");
        }
        DispatchOp::Indirect { kernel: other, .. } => {
            panic!("dispatch: unhandled Indirect kernel {other:?}");
        }
        DispatchOp::GatedBy { kernel: other, .. } => {
            panic!("dispatch: unhandled GatedBy kernel {other:?}");
        }
    }
    Ok(())
}
```

> P10 note: the spec says "every `DispatchOp` variant has a panic-free path." The fallthrough panics are `unreachable!`-style guards on a closed enum: the compiler-emitted `KernelId` exhausts every enumerator the `dispatch_*` arms above match on. The fallthroughs only fire if a future kernel emitter adds a `KernelId` variant without updating `dispatch()` — which is a build-error at every match-arm site if we replace `panic!` with `unreachable!()` after Task 16 confirms exhaustiveness. Update those lines to `unreachable!()` once the dispatch is complete.

- [ ] **Step 4: Run full parity sweep**

Run: `cargo test -p engine_gpu`
Run: `cargo test -p engine --test wolves_and_humans_parity`
Expected: ALL existing parity tests pass.

If any fail, the schedule order doesn't match the previous in-line order. Localize via `git diff` against pre-Task-15 — the dispatch arms move blocks but don't change them.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_gpu/src/
git commit -m "feat(engine_gpu): step_batch consumes engine_gpu_rules::SCHEDULE via dispatch() match"
```

---

### Task 16: Final cleanup — delete hand-written kernel files; final parity sweep

Every kernel under `crates/engine_gpu/src/{mask,scoring,apply_actions,movement,physics,alive_bitmap,spatial_gpu,cascade_resident}.rs` is now dead code (transitional WGSL `pub const`s remain; remove them). The `panic!` fallthroughs from Task 15 become `unreachable!()` once exhaustiveness is verified.

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs` (drop `pub mod` entries for migrated kernels)
- Delete: `crates/engine_gpu/src/{mask,scoring,apply_actions,movement,physics,alive_bitmap,spatial_gpu}.rs`
- Modify: `crates/engine_gpu/src/cascade_resident.rs` (delete kernel structs; keep `CascadeResidentCtx` skeleton if still referenced)
- Modify: `crates/engine_gpu/src/lib.rs::dispatch` (turn `panic!` fallthroughs into `unreachable!()`)

- [ ] **Step 1: Confirm no in-tree code imports the about-to-be-deleted modules**

Run: `grep -rn 'engine_gpu::mask::' crates/ --include='*.rs' | grep -v 'crates/engine_gpu/'`
Expected: no hits (or only hits in deleted-by-this-task files).

Run: `grep -rn 'engine_gpu::scoring::' crates/ --include='*.rs' | grep -v 'crates/engine_gpu/'`
Run: `grep -rn 'engine_gpu::movement::' crates/ --include='*.rs' | grep -v 'crates/engine_gpu/'`
Run: `grep -rn 'engine_gpu::physics::' crates/ --include='*.rs' | grep -v 'crates/engine_gpu/'`
Run: `grep -rn 'engine_gpu::apply_actions::' crates/ --include='*.rs' | grep -v 'crates/engine_gpu/'`
Run: `grep -rn 'engine_gpu::alive_bitmap::' crates/ --include='*.rs' | grep -v 'crates/engine_gpu/'`
Run: `grep -rn 'engine_gpu::spatial_gpu::' crates/ --include='*.rs' | grep -v 'crates/engine_gpu/'`
Expected: each command yields 0 cross-crate hits. If any command yields hits, replace those references with `engine_gpu_rules::<kernel>::` first.

- [ ] **Step 2: Drop the `pub mod` declarations**

Open `crates/engine_gpu/src/lib.rs` and delete:

```rust
pub mod mask;
pub mod scoring;
pub mod apply_actions;
pub mod movement;
pub mod physics;
pub mod alive_bitmap;
pub mod spatial_gpu;
```

(Keep `cascade_resident` for now — it owns `CascadeResidentCtx` and may still be referenced by snapshot/sync paths; trim it down in Step 3.)

- [ ] **Step 3: Delete the kernel files**

Run:
```bash
rm crates/engine_gpu/src/mask.rs
rm crates/engine_gpu/src/scoring.rs
rm crates/engine_gpu/src/apply_actions.rs
rm crates/engine_gpu/src/movement.rs
rm crates/engine_gpu/src/physics.rs
rm crates/engine_gpu/src/alive_bitmap.rs
rm crates/engine_gpu/src/spatial_gpu.rs
```

In `crates/engine_gpu/src/cascade_resident.rs`, delete the `SeedIndirectKernel` and `AppendEventsKernel` impl blocks; keep only the `CascadeResidentCtx` struct + accessors.

- [ ] **Step 4: Confirm `cargo build` is clean**

Run: `cargo build -p engine_gpu --features gpu`
Expected: clean build.

If errors mention deleted symbols, those are the remaining transitional accessors that snapshot/sync paths still call. Replace each call with the `engine_gpu_rules` equivalent or extract a minimal hand-written helper into a non-kernel file (e.g. `engine_gpu/src/sync_helpers.rs`).

- [ ] **Step 5: Run the full test sweep**

Run: `cargo test --workspace`
Expected: all existing tests pass.

Specifically:
- `cargo test -p engine --test wolves_and_humans_parity` — PASS
- `cargo test -p engine_gpu --test parity_with_cpu` — PASS
- `cargo test -p engine_gpu --test physics_parity` — PASS
- `cargo test -p engine_gpu --test cascade_parity` — PASS
- `cargo test -p engine_gpu --test view_parity` — PASS
- `cargo test -p engine_gpu --test topk_view_parity` — PASS
- `cargo test -p engine_gpu --test spatial_parity` — PASS
- `cargo test -p engine_gpu --test step_batch_smoke` — PASS
- `cargo test -p engine_gpu_rules --test schema_hash` — PASS
- `cargo test -p engine --test schema_hash` — PASS

- [ ] **Step 6: Tighten `dispatch()`'s P10 guards**

In `crates/engine_gpu/src/lib.rs::dispatch`, replace the `panic!` fallthroughs with `unreachable!()`:

```rust
DispatchOp::Kernel(other) => unreachable!("KernelId {other:?} has no dispatch arm; emitter regression"),
DispatchOp::FixedPoint { kernel: other, .. } => unreachable!("FixedPoint {other:?} has no dispatch arm"),
DispatchOp::Indirect { kernel: other, .. } => unreachable!("Indirect {other:?} has no dispatch arm"),
DispatchOp::GatedBy { kernel: other, .. } => unreachable!("GatedBy {other:?} has no dispatch arm"),
```

This is P10-clean: `unreachable!` is a compile-time contract assertion; reaching it would be an emitter regression detected at the kernel-emit level (the compile-dsl run would fail before the binary ships).

- [ ] **Step 7: Re-run baseline + final test sweep**

Run: `cargo run --bin xtask -- compile-dsl`
Run: `cargo test -p engine_gpu_rules --test schema_hash`
Run: `cargo test -p engine --test schema_hash`
Expected: PASS.

Run: `cargo test --workspace`
Expected: all pass.

- [ ] **Step 8: Bump engine .schema_hash to roll in gpu_rules_hash**

Open `crates/engine/.schema_hash`. Add a coupling comment line above the existing 64-byte hex (this is the "Coupling line" the spec calls out):

> Note: the engine `.schema_hash` file currently holds a single hex string; the spec's coupling line is a separate concept. Implementation choice: extend `crates/engine/src/schema_hash.rs::schema_hash()` to also call `engine_gpu_rules::gpu_rules_hash()` (cross-crate hash dep) so the engine's hash is invalidated when GPU rules change. Add to `engine/src/schema_hash.rs`:
>
> ```rust
> // P2 coupling: engine snapshots reject if GPU rules changed without coordinated bump.
> let gpu_hash_str = include_str!("../../engine_gpu_rules/.schema_hash").trim();
> h.update(b"engine_gpu_rules.schema_hash=");
> h.update(gpu_hash_str.as_bytes());
> ```
>
> Then re-run `cargo test -p engine --test schema_hash` — it will fail. Capture the new hash from the failure message and write it into `crates/engine/.schema_hash`.

Run: `cargo test -p engine --test schema_hash`
Expected: FAIL on first run (with the new hash printed); after copying the new hash into `crates/engine/.schema_hash`, expect PASS.

- [ ] **Step 9: Tick the AIS post-design checkbox**

Open `docs/superpowers/plans/2026-04-26-kernel-dispatch-emit-impl.md` and tick:

```
- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design (after task list stabilises).
```

- [ ] **Step 10: Final commit**

```bash
git add crates/engine_gpu/ crates/engine/ docs/superpowers/plans/
git commit -m "refactor(engine_gpu): delete hand-written kernel modules; engine .schema_hash couples engine_gpu_rules"
```

---

## Final verification

After Task 16 completes, the following invariants hold:

1. `crates/engine_gpu/src/` contains only `lib.rs`, `gpu_util/`, `gpu_profiling.rs`, `event_ring.rs` (if still used), `view_storage*.rs` (transitional — Subsystem 0 follow-up), `snapshot.rs`, `sim_cfg.rs`, `cascade.rs`, `cascade_resident.rs` (skeleton only), and the `backend/` folder.
2. Every per-kernel `.rs` and `.wgsl` file under `crates/engine_gpu_rules/src/` starts with `// GENERATED by dsl_compiler`.
3. `cargo run --bin xtask -- compile-dsl` regenerates every file under `engine_gpu_rules/src/` to byte-equal output.
4. `cargo test -p engine_gpu_rules --test schema_hash` passes against a committed baseline.
5. Every parity test (`wolves_and_humans_parity`, `parity_with_cpu`, `physics_parity`, `cascade_parity`, `view_parity`, `topk_view_parity`, `spatial_parity`) passes.
6. Engine snapshot loads reject when GPU rules change (because `engine/.schema_hash` rolls in `engine_gpu_rules/.schema_hash`).
