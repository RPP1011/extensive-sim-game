# Kernel Dispatch-Emit Abstraction — Design

**Status:** spec (deliverable: implementation plan to follow)
**Date:** 2026-04-26
**Predecessors:**
- `docs/superpowers/research/2026-04-24-dsl-compiler-improvement-opportunities.md` (10 ranked optimization areas; Areas 3/4/7/8/9 directly relevant)
- `docs/superpowers/research/2026-04-26-kernel-dispatch-emit-research.md` (work-graph + render-graph survey)

---

## Goal

Move all GPU kernel-wrapper boilerplate (struct definition, BGL builder, pipeline construction, buffer allocation, dispatch encoding) into compiler-emitted code. Land in a new sibling crate `engine_gpu_rules/`. Migrate all 14 existing kernels in one plan.

**Constitution alignment.** Today's hand-written `Kernel` wrappers (`ScoringKernel`, `ApplyActionsKernel`, `MovementKernel`, etc.) are P1 (compiler-first) violations in disguise — every new `.sim` row that produces a kernel forces hand-edits to `engine_gpu/`. The abstraction closes that hole: a `.sim` change regenerates everything kernel-specific (struct, bindings, WGSL, schedule entries) without touching `engine_gpu/`.

## Non-goals

- **No new DSL grammar.** Buffer lifetimes are inferred from row type (Q2). DSL authors continue to write rules; they do not annotate GPU concerns.
- **No third-party WGSL→Rust binding tool.** `dsl_compiler` already holds both sides in IR; emit them from the same walk (Q6). No `wgsl_bindgen` / `naga` reflection dependency.
- **No barrier-insertion engine.** WebGPU auto-syncs dispatches within an encoder; we don't need to invent it.
- **No partial migration.** All 14 kernels move in one plan (Q5).

## Architecture

### Crate split (post-landing)

```
crates/dsl_compiler/      ← reads .sim → emits both Rust + WGSL
crates/engine_gpu_rules/  ← committed // GENERATED Rust + WGSL files (NEW)
crates/engine_gpu/        ← hand-written GPU primitives only
                            (wgpu setup, megakernel orchestration,
                             generic BufferPool, snapshot/replay)
```

The cut: `engine_gpu/` owns *resources and orchestration*; `engine_gpu_rules/` owns *kernel-specific code*. Anything per-kernel (BGL shape, WGSL body, struct layout, dispatch encoding, dependency graph entry) moves to `engine_gpu_rules/`. Anything cross-kernel (megakernel composition, the resident context skeleton, the buffer pool) stays in `engine_gpu/`.

**Invariant:** every file under `engine_gpu_rules/src/` is `// GENERATED`, including `lib.rs`. The only hand-written files in the crate are `Cargo.toml` and `build.rs`. `lib.rs` itself is regenerated each time `xtask compile-dsl` runs; its body contains a static prelude (the `Kernel` trait, the `KernelId` enum, the `BufferRef` enum) plus a varying module list — both produced by the same `emit_kernel_index` pass. The static prelude is byte-stable across regens with no `.sim` change. Enforced by a `build.rs` sentinel that scans for any `.rs`/`.wgsl` under `src/` lacking a `// GENERATED` header (same mechanism `engine_rules/` uses).

This mirrors the `engine_data` ↔ `engine_rules` split that landed in Plan B1' (Subsystem 0): "data + hand-written primitives in one crate; all emitted code in a sibling crate." `engine_gpu_rules/` is the GPU sibling.

### Emit pipeline

```
.sim files
   │
   ▼
dsl_compiler ─────── walks IR per row family
   │
   ├─ Rust binding emit  → engine_gpu_rules/src/<kernel>.rs        (// GENERATED)
   ├─ WGSL emit          → engine_gpu_rules/src/<kernel>.wgsl      (// GENERATED)
   ├─ Schedule emit      → engine_gpu_rules/src/schedule.rs        (// GENERATED)
   ├─ Resident ctx emit  → engine_gpu_rules/src/resident_context.rs (// GENERATED)
   ├─ PingPong ctx emit  → engine_gpu_rules/src/pingpong_context.rs (// GENERATED)
   └─ Megakernel emit    → engine_gpu_rules/src/megakernel.{rs,wgsl} (// GENERATED)
```

Both Rust and WGSL committed. `xtask compile-dsl` regenerates them. `cargo build` is a pure compile step — same convention as `engine_rules/`.

## Lifetime classes (the row → buffer mapping)

Five canonical classes. Every emitted buffer falls into one. Mapping is fully inferable from the DSL row type — no annotation:

| Class | Owner | Lifetime | Reset cadence |
|-------|-------|----------|---------------|
| `Transient` | `BufferPool` (engine_gpu) | one dispatch | recycled per tick |
| `Resident` | `ResidentPathContext` | batch lifetime | persists across ticks |
| `PingPong` | `CascadeResidentCtx` | one cascade iteration | A/B alternates per iter |
| `External` | engine consumer | passed-in handle | unmanaged by kernel |
| `Pooled` | shape-keyed pool | reused across compatible kernels | LRU eviction |

### Row-type → lifetime decision table

| DSL row type | Output buffer | Notable inputs |
|--------------|---------------|----------------|
| `mask` predicate | Transient (per-tick mask buf) | External (agent SoA, sim_cfg) |
| `scoring` (`target_bound`) | Transient (action buf) | External + Resident (views, cooldowns) |
| `scoring` (`per_ability`) | Resident (`chosen_ability_buf`) | External + Resident |
| `view` declaration | Resident (`view_storage`) | External + PingPong (event ring) |
| `physics` rule | PingPong (next-iter event ring) | External + PingPong (current-iter ring) |
| `apply` phase | Mutates External (agent SoA) + emits PingPong | Transient + Resident |
| `@spatial query` | Pooled | External |

**Verification.** Walked all 14 existing kernels; every buffer fits one of the 5 classes. No outliers.

### Class semantics consumed by the emitter

Each kernel's emitted Rust file declares its buffers with their class. The class determines:

- Where the binding handle lives (in `BufferPool` vs `ResidentPathContext` field vs caller arg)
- Whether the buffer is allocated lazily (Transient) or eagerly at `ensure_resident_init` (Resident, PingPong)
- Whether it's cleared/zeroed each tick (Transient = yes, Resident = no, PingPong = swap)
- Whether it participates in schema-hash via byte content (Resident yes, Transient no)

## Per-kernel emit surface

For each `.sim` row that produces a kernel, the compiler emits **one Rust file + one WGSL file** in `engine_gpu_rules/src/`. Plus shared infrastructure files.

### Per-kernel files (example: `pick_ability`)

**`engine_gpu_rules/src/pick_ability.rs`** (// GENERATED):

```rust
// GENERATED by dsl_compiler::emit_kernel — do not edit by hand.

pub struct PickAbilityKernel {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

pub struct PickAbilityBindings<'a> {
    pub agents: &'a wgpu::Buffer,            // External
    pub ability_registry: &'a wgpu::Buffer,  // External
    pub per_slot_cooldown: &'a wgpu::Buffer, // Resident
    pub chosen_ability_buf: &'a wgpu::Buffer,// Resident (this kernel's primary output)
    pub cfg: &'a wgpu::Buffer,               // Transient
    pub sim_cfg: &'a wgpu::Buffer,           // External
    pub tag_values: &'a wgpu::Buffer,        // External
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PickAbilityCfg {
    pub agent_cap: u32,
    pub ability_count: u32,
    pub num_tags: u32,
    pub tick: u32,
}

const SHADER_SRC: &str = include_str!("pick_ability.wgsl");

impl Kernel for PickAbilityKernel {
    type Bindings<'a> = PickAbilityBindings<'a>;
    type Cfg = PickAbilityCfg;

    fn new(device: &wgpu::Device) -> Self { /* emitted */ }

    fn build_cfg(&self, state: &SimState, registry: &AbilityRegistry) -> Self::Cfg { /* emitted */ }

    fn bind<'a>(
        &'a self,
        ctx: &'a ResidentPathContext,
        cfg: &'a wgpu::Buffer,
    ) -> Self::Bindings<'a> { /* emitted — wires ctx fields to binding slots */ }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &Self::Bindings<'_>,
        agent_cap: u32,
    ) { /* emitted bind-group + dispatch */ }
}
```

**`engine_gpu_rules/src/pick_ability.wgsl`** (// GENERATED) — the kernel body, formerly built at runtime as a `String`.

### Shared infrastructure files

**`engine_gpu_rules/src/lib.rs`** — module list + `Kernel` trait:

```rust
// GENERATED — module list maintained by dsl_compiler::emit_kernel_index
pub mod fused_mask;
pub mod scoring;
pub mod pick_ability;
// ... 14 modules total

pub mod schedule;
pub mod resident_context;
pub mod megakernel;

pub trait Kernel {
    type Bindings<'a>;
    type Cfg: bytemuck::Pod + bytemuck::Zeroable;
    fn new(device: &wgpu::Device) -> Self;
    fn build_cfg(&self, state: &SimState, registry: &AbilityRegistry) -> Self::Cfg;
    fn bind<'a>(&'a self, ctx: &'a ResidentPathContext, cfg: &'a wgpu::Buffer) -> Self::Bindings<'a>;
    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &Self::Bindings<'_>,
        agent_cap: u32,
    );
}
```

**No dynamic dispatch, no `Box<dyn Kernel>`.** Concrete types throughout. The trait is for type-level uniformity, testing, and contract — not for generic iteration. (Megakernel, not the trait, is the path to schedule-level uniformity.)

**`engine_gpu_rules/src/resident_context.rs`** (// GENERATED) — fields for every Resident buffer, with constructor:

```rust
pub struct ResidentPathContext {
    pub agents: wgpu::Buffer,
    pub sim_cfg: wgpu::Buffer,
    pub chosen_ability_buf: wgpu::Buffer,
    pub view_storage_engaged_with: wgpu::Buffer,
    // ... one field per Resident-class buffer
}

impl ResidentPathContext {
    pub fn new(device: &wgpu::Device, agent_cap: u32, /* ... */) -> Self { /* emitted */ }
}
```

**`engine_gpu_rules/src/pingpong_context.rs`** (// GENERATED) — A/B ring buffers for cascade physics.

## Schedule

The compiler emits a **Schedule** alongside the kernels. It is *data*, not code — a list of typed dispatch ops derived from row IR. Engine_gpu's `step_batch` is a loop over the Schedule.

### `engine_gpu_rules/src/schedule.rs` (// GENERATED)

```rust
pub enum DispatchOp {
    /// Single dispatch of one emitted kernel.
    Kernel(KernelId),
    /// Fixed-point loop over a kernel until its event ring drains.
    /// Used exclusively for cascade physics.
    FixedPoint { kernel: KernelId, max_iter: u32 },
    /// Indirect dispatch driven by a buffer.
    Indirect { kernel: KernelId, args_buf: BufferRef },
    /// Conditional: only run if a producer wrote a non-sentinel value.
    GatedBy { kernel: KernelId, gate: BufferRef },
}

pub const SCHEDULE: &[DispatchOp] = &[
    DispatchOp::Kernel(KernelId::FusedMask),
    DispatchOp::Kernel(KernelId::Scoring),
    DispatchOp::Kernel(KernelId::PickAbility),
    DispatchOp::Kernel(KernelId::ApplyActions),
    DispatchOp::Kernel(KernelId::Movement),
    DispatchOp::FixedPoint { kernel: KernelId::Physics, max_iter: 8 },
    DispatchOp::Kernel(KernelId::FoldEngagedWith),
    // ... every kernel covered
];
```

### Compiler derivation

Each row's IR declares its read/write set (already explicit in IR via `IrExpr` tree walks). The compiler:

1. Builds a DAG: row X writes buffer B; row Y reads buffer B → edge X→Y.
2. Topologically sorts the DAG.
3. Maps each row to a `DispatchOp` variant:
   - Default → `DispatchOp::Kernel`
   - Cascade physics rules → `DispatchOp::FixedPoint` (intrinsically iterative per row type)
   - Cascade indirect → `DispatchOp::Indirect`
   - Sentinel-conditional dependency → `DispatchOp::GatedBy`

### Compile-time errors (P10 / pre-commit)

The DAG construction surfaces classes of bugs the emitter rejects:

1. **Diamond writes.** Multiple rows writing the same buffer — compiler errors: "row X and row Y both write `actions_buf`."
2. **Cycles.** Row X reads what Y writes and vice versa — compiler errors with the cycle path.
3. **Unscheduled kernel.** A kernel module exists but no row produces it — compiler errors.
4. **Race within a single Schedule entry** — handled by WebGPU's auto-sync; no barrier emit needed.

### Engine_gpu's step_batch (hand-written)

```rust
fn step_batch(&mut self, state: &SimState, registry: &AbilityRegistry) {
    let mut encoder = self.device.create_command_encoder(&Default::default());
    for op in engine_gpu_rules::schedule::SCHEDULE {
        self.dispatch(op, &mut encoder, state, registry)?;
    }
    self.queue.submit(Some(encoder.finish()));
}
```

The `self.dispatch(op, ...)` is one match arm per `DispatchOp` variant — written once, never edited as new rows land.

### Megakernel as fuse pass over Schedule

Megakernel is a **second emit pass** that walks `SCHEDULE` and produces a single fused WGSL kernel:

- Each `DispatchOp::Kernel` becomes an inline section of one shader
- `FixedPoint` becomes a WGSL `while`-loop
- `GatedBy` becomes an `if` branch

Both modes coexist:

- **Stepwise mode** (debug / parity / snapshot): consume `SCHEDULE` directly, dispatch N kernels.
- **Megakernel mode** (production): consume `MEGAKERNEL`, one dispatch.

The schedule layer is the same data feeding both — there's no point at which dispatch-emit and megakernel disagree on dispatch order.

## Hot-reload via `interpreted-rules`

Today's `interpreted-rules` cargo feature routes mask + scoring through `dsl_ast::eval` instead of compiled output. After dispatch-emit lands, the same feature can extend to GPU dispatch:

```rust
#[cfg(feature = "interpreted-rules")]
fn step_batch(&mut self, state: &SimState, ...) {
    let schedule = dsl_compiler::emit_schedule(&self.cached_ir);
    for op in &schedule {
        self.dispatch_interpreted(op, state, ...);
    }
}
```

When the feature is on:
- `engine_gpu_rules/`'s committed files are not consumed.
- `dsl_compiler::emit_*` runs at boot to produce equivalent Rust + WGSL in-memory.
- `dispatch_interpreted` either runs CPU IR eval (mask, scoring, view fold — already exists) or assembles a one-shot `wgpu::ComputePipeline` from emitted-in-memory WGSL (new capability).
- `.sim` edit + `cargo run --features interpreted-rules` = live reload, no regen-and-commit cycle.

**Out of scope for the first plan.** Extending `interpreted-rules` to cover GPU dispatch is a follow-up plan. The first plan keeps scope tight on the production path.

## Schema-hash coverage (P2)

Two new artifacts:

1. **`engine_gpu_rules/.schema_hash`** — new baseline file. Hashed inputs:
   - Every `// GENERATED` `.rs` and `.wgsl` byte content
   - The `SCHEDULE` constant
   - The `MEGAKERNEL` WGSL source
2. **Coupling line in `engine/.schema_hash`**: `engine_gpu_rules.schema_hash = <hash>`. Engine snapshot loads reject if GPU rules changed without coordinated bump.

CI test: `tests/schema_hash_gpu_rules.rs` recomputes from filesystem and compares.

`xtask compile-dsl --check` already bumps `engine_rules/.schema_hash`; extend to also bump `engine_gpu_rules/.schema_hash`.

## Test strategy (P3 + P11)

End-behavior parity only — no per-kernel byte-equal tests (those over-specify what parity requires).

1. **Existing parity tests stay green.** `wolves_and_humans_parity.rs` (100-tick committed baseline), `parity_with_cpu.rs`, `physics_parity.rs`, `cascade_parity.rs`. These are end-behavior tests; they catch any kernel that changes observable output. Localization on failure: `git bisect` over migration commits, or re-run parity after each migration group.
2. **Schedule DAG equivalence test** in `dsl_compiler` — asserts compiler-inferred kernel dependency graph from current `.sim` files matches a committed reference. Catches *unexpected* changes to scheduling at compile time. Sanity rail on the DAG-inference algorithm; not parity-related.
3. **Schema-hash baselines** — catch WGSL/Rust binary content changes. Layout-relevant; not behavior-relevant. Standard P2 mechanism.

## Error handling (P10 — no runtime panic)

- **Compile-time errors only** for diamond / cycle / unscheduled (DAG construction). Bad rules fail `xtask compile-dsl --check` pre-commit; never land in committed output.
- **Runtime errors in `dispatch()` match arms**: every `DispatchOp` variant has a panic-free path. Indirect-dispatch buffer validation already exists in engine_gpu's snapshot logic; reuse.
- **Buffer-allocation failures** (OOM at backend init) → `Result<GpuBackend, GpuInitError>`. No new error types beyond what engine_gpu already exposes.

## Migration plan structure

Work units are **emitters**, not kernels. Each emitter handles a DSL row family. One emitter pass runs over every row of that family and produces N kernel modules — no per-kernel migration cost beyond the first.

### Six emitters cover all 14 kernels

| Emitter | Covers | New work |
|---------|--------|----------|
| `emit_mask_kernel` | `FusedMaskKernel`, `MaskUnpackKernel` | New |
| `emit_scoring_kernel` (target_bound) | `ScoringKernel`, `ScoringUnpackKernel`, `ApplyActionsKernel` | New |
| `emit_scoring_kernel` (per_ability) | `PickAbilityKernel` | Extend (Subsystem 3 A1+A2+A3 wrote the row body emitter + schema hash; this adds the wrapper emit on top) |
| `emit_movement_kernel` | `MovementKernel`, `PhysicsKernel`, `SeedIndirectKernel`, `AppendEventsKernel` | New |
| `emit_view_fold_kernel` | All 8 `cs_fold_<view>` kernels | New (the WGSL emit exists; this adds wrapper emit) |
| `emit_spatial_kernel` | `cs_spatial_hash`, `cs_kin_query`, `cs_engagement_query`, `AlivePackKernel`, `FusedAgentUnpackKernel` | New |

### Migration plan structure (~8 tasks)

1. **Bootstrap.** `engine_gpu_rules/` skeleton, `Kernel` trait, `Schedule` + `DispatchOp` machinery, `ResidentPathContext` emit, schema-hash baseline. No kernels migrated yet.
2. **`emit_mask_kernel`.** First real emitter. Validates the end-to-end pipeline on a leaf kernel. Parity must stay green.
3. **`emit_scoring_kernel` (both row-type variants).** The per_ability variant lights up the existing Subsystem 3 A1+A2+A3 row-body emitter.
4. **`emit_movement_kernel`.** Covers cascade + indirect dispatch. Validates `FixedPoint` and `Indirect` Schedule ops.
5. **`emit_view_fold_kernel`.** One emitter, 8 modules, in one commit. Big LOC delta, no marginal complexity beyond emitter #1.
6. **`emit_spatial_kernel`.** Covers `Pooled` lifetime class.
7. **Megakernel fuse pass.** Once individual kernels emit cleanly, walk `SCHEDULE` to fuse into one WGSL kernel. May reuse existing megakernel work in flight (`docs/superpowers/plans/gpu_megakernel_plan.md`).
8. **Cleanup.** Delete hand-written `crates/engine_gpu/src/{scoring,apply_actions,movement,physics,...}.rs`. Verify nothing in engine_gpu imports deleted code. Final parity sweep + schema-hash bump.

The migration is a `dsl_compiler/` plan with engine_gpu-side cleanup, not a kernel-by-kernel rewrite.

## Coexistence with in-flight work

- **Megakernel** (`gpu_megakernel_plan.md`): step 7 above. The fuse pass is the natural meeting point. If megakernel lands first, dispatch-emit consumes its WGSL emit interface; if dispatch-emit lands first, megakernel becomes a second emit pass over `SCHEDULE`.
- **Subsystem 3 — GPU ability evaluation**: the per_ability row body emit (Tasks A1 + A2 + A3, all landed as of 2026-04-26) is already in place. Group B (B1–B5) becomes the first consumer of dispatch-emit rather than hand-written wiring — replaced entirely by `emit_scoring_kernel` (per_ability) covering `PickAbilityKernel`. **Subsystem 3 Group B is removed from the work queue; folded into this plan.**
- **Plan 6 (`GpuBackend` foundation)**: subsumed. Plan 6 was "bridge `ComputeBackend` trait to engine_gpu primitives"; dispatch-emit makes that bridge generated, not hand-written. Plan 6 entry should be removed from the roadmap once this plan lands.
- **Cold-state replay Phases 2–4**: independent — those plans already produce hand-written kernels. They can land before or after dispatch-emit; if before, those kernels migrate as part of step 5 / 6 of the migration plan.

## Open questions deferred to the implementation plan

The brainstorm settled the architectural shape; these are concrete tactical decisions for the writing-plans pass:

- File-naming convention inside `engine_gpu_rules/src/` (per-row vs per-row-family).
- Exact `DispatchOp` variants beyond the four listed (any cases the survey missed).
- `interpreted-rules` extension — sequencing in a follow-up plan.
- LRU eviction policy for `Pooled` buffers — small detail; can be hardcoded initially.
