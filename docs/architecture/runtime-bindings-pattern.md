# Runtime Bindings Pattern

How per-runtime `step()` code constructs and dispatches compiler-emitted GPU
kernels. Adopted as of the Phase 4 migration (PRs #62-#78), when most
per-fixture `crates/*_runtime` crates still existed independently; today
almost all of that generated code lives inside the `crates/sims` mega-crate
(one `OUT_DIR` module per fixture) instead, with `crates/tom_probe_runtime`
and `crates/viewer_runtime` the last standalone holdouts — see
`crates/sims/CLAUDE.md`. The pattern itself is unchanged by that
consolidation; only the crate each `step()` lives in has moved.

## Goal

The dsl_compiler emits, into each runtime's `OUT_DIR/generated.rs`, a
`<KernelName>Bindings` struct per kernel and a sibling
`<KernelName>Bindings::from_context(...)` constructor (or
`from_context_with_extras(...)` when the kernel binds fixture-specific
buffers). Per-runtime dispatch uses the constructor — never a hand-written
`Bindings { ... }` literal.

Hand-written literals would fan out across every fixture's `step()` (100+
today, mostly generated modules inside `crates/sims`, plus the two
standalone legacy runtime crates): every new SoA column or compiler-emitted
buffer would have to be threaded through every dispatch site. The
constructor pattern keeps that fan-out inside the compiler.

## The pattern

Per-kernel dispatch is three statements:

```rust
let agent_buffers = engine::gpu::AgentBuffers {
    hp_buf: Some(&self.agent_hp_buf),
    alive_buf: Some(&self.agent_alive_buf),
    // ...standard SoA buffers this kernel reads...
    ..Default::default()
};
let ctx = engine::gpu::KernelBindingsContext {
    state: &agent_buffers,
    event_ring: &self.event_ring,
    registry: &self.registry_gpu,
    voxel_grid: None,                 // Some(&buf) only for voxel-aware fixtures
};
let bindings = SomeKernelBindings::from_context_with_extras(
    &ctx,
    &SomeKernelExtras {
        cfg: &self.some_cfg_buf,      // fixture-specific uniforms / SoA / mask buffers
    },
);
dispatch::dispatch_some_kernel(&mut self.cache, &bindings, &device, &mut encoder, count);
```

Use plain `from_context(&ctx)` when the kernel binds **only** standard sources.
Use `from_context_with_extras(&ctx, &extras)` whenever a kernel binds anything
the naming convention can't classify (fixture-specific cfg uniforms, mask
bitmaps, scoring output buffers, fixture-only SoA like
`agent_creature_type`).

## Naming convention

The compiler classifies each emitted Bindings field name and routes it via:

| Field name pattern              | Source           | Resolves to                   |
|---------------------------------|------------------|-------------------------------|
| `event_ring`                    | `ctx.event_ring` | `ctx.event_ring.ring()`       |
| `event_tail`                    | `ctx.event_ring` | `ctx.event_ring.tail()`       |
| `agent_<col>` (standard SoA)    | `ctx.state`      | `ctx.state.<col>_buf`         |
| `ability_registry_<col>`        | `ctx.registry`   | `&ctx.registry.<col>`         |
| `voxel_grid`                    | `ctx.voxel_grid` | `ctx.voxel_grid.expect(...)`  |
| (anything else)                 | `extras`         | `extras.<name>`               |

Source of truth: `crates/engine/src/gpu/bindings_context.rs` and
`crates/dsl_compiler/src/cg/emit/program.rs::classify_binding`.

## Empty-placeholder rule

Some runtimes don't actually use one of the shared sources (a tiny probe
fixture may not consume the chronicle ring; a non-cast fixture may not need
the ability registry). The constructor still requires every field of
`KernelBindingsContext` — supply a placeholder:

- **No `EventRing`**: keep an empty `engine::gpu::EventRing::new(&device, 0)`
  on the runtime struct anyway and pass `&self.event_ring`.
- **No `PackedAbilityRegistryGpu`**: keep an empty
  `PackedAbilityRegistryGpu::empty(&device)` and pass `&self.registry_gpu`.
- **No voxel terrain**: pass `voxel_grid: None`.

The placeholders compile to a few hundred bytes of unused buffer; the win is
that every runtime constructs `ctx` the same way.

## The `note_emits` ctx-rebuild gotcha

`KernelBindingsContext` borrows `&self.event_ring` immutably. If the runtime
needs to call `self.event_ring.note_emits(count)` (mutable borrow) **between**
kernel dispatches, the compiler will reject the second mutable borrow while
`ctx` still holds the immutable one.

**Fix**: scope the dispatch in a fresh `{ ... }` block so `ctx` drops before
the `note_emits` call:

```rust
{
    let agent_buffers = AgentBuffers { /* ... */ ..Default::default() };
    let ctx = KernelBindingsContext { /* ... */ };
    let bindings = SomeKernelBindings::from_context_with_extras(&ctx, &extras);
    dispatch::dispatch_some_kernel(&mut self.cache, &bindings, &device, &mut encoder, count);
}
self.event_ring.note_emits(count);
```

Affects fixtures that drive multi-stage pipelines with chronicle bookkeeping
between stages: `diplomacy_probe`, `tom_probe`, `stochastic_probe`,
`stdlib_math_probe`. Same pattern applies anywhere `note_emits` lives between
two compiler-emitted dispatches.

## When to add a new field to a shared source

The compiler classifies by **field-name prefix**, not by which struct holds the
buffer. Adding a new shared source means:

1. **Standard agent SoA column?** Add `Option<&'a wgpu::Buffer>` field on
   `AgentBuffers` and a row to `classify_binding`'s `agent_<col>` table.
2. **New shared infrastructure** (e.g. a global LUT)? Add a field on
   `KernelBindingsContext` and a row in `classify_binding` that routes the
   matching binding name to it.
3. **Fixture-specific?** Don't touch shared types. The compiler will route
   the field through the per-kernel `<KernelName>Extras` struct
   automatically.

Standard SoA columns shared across many fixtures (hp, alive, pos, ...) earn a
slot on `AgentBuffers`. Fixture-only columns (e.g. `agent_creature_type` in
spy_network) ride extras.

## Migrating a new runtime author's checklist

1. For every `let bindings = <KernelName>Bindings { ... }` literal, replace
   with a `from_context()` or `from_context_with_extras()` call.
2. Hoist any per-step shared `agent_buffers` + `ctx` to the top of `step()`
   (or per scope if `note_emits` is involved).
3. Anything the compiler refuses to classify ("no field `foo` on
   `KernelBindingsContext`") goes into `<KernelName>Extras`.
4. Add empty placeholders for any shared source the runtime doesn't actually
   use.
5. `cargo build -p <crate> --release` until clean.
6. `cargo test -p <crate> --release` — no behavioral change expected; this
   refactor must not move tests.

## See also

- `crates/engine/src/gpu/bindings_context.rs` — shared types + extended
  doc comments.
- `crates/dsl_compiler/src/cg/emit/program.rs::classify_binding` — the
  routing table the compiler uses.
- `docs/superpowers/plans/2026-05-09-compiler-emitted-bindings-constructors.md`
  — the design plan, with rationale and per-phase breakdown.
