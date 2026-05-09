# Compiler-Emitted Bindings Constructors — Eliminate the Per-Runtime Fanout Tax

> Goal: every per-kernel `Bindings { ... }` literal in every runtime's `step()` becomes a single `Bindings::from_context(&ctx)` call. The compiler already knows which buffers each kernel reads — emit the constructor that wires them up.

## Goal

The dsl_compiler emits `OUT_DIR/generated.rs` with one `Bindings` struct per kernel (`PhysicsDispatchAoePulseBindings`, `FusedMaskVerbDisguiseBindings`, `ScoringBindings`, etc.). It also knows exactly which buffers each kernel reads — the WGSL it generates references `agent_hp[i]`, `event_ring[slot * 10u]`, `mask_0_bitmap[word]`, `ability_registry_when_pred_field[...]`, etc.

What it **doesn't** emit today is the **construction** of those Bindings literals. Per-runtime `step()` code spells out every field by hand:

```rust
// 30 lines × 8 dispatches × ~40 runtimes = ~9 600 hand-written binding lines
let bindings = PhysicsDispatchAoePulseBindings {
    event_ring: self.event_ring.ring(),
    event_tail: self.event_ring.tail(),
    agent_pos: &self.agent_pos_buf,
    agent_hp: &self.agent_hp_buf,
    // ... 26 more
};
```

Every new SoA column (PR #46 AbilityPower, PR #55 fanout, commit bcb490de stress_cast_density miss) and every new compiler-emitted buffer (PR #58 Phase 2 instrumentation) repeats the fanout. **The compiler should emit the constructor.**

After this slice:

```rust
let bindings = PhysicsDispatchAoePulseBindings::from_context(&self.kernel_ctx());
```

Per-fixture extras (mask bitmaps with fixture-specific names, scoring output, ad-hoc per-fixture buffers) ride a small `Extras` struct or trailing args. Mostly the migrating fixture replaces a 30-line literal with a 1-3 line call.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/spy_network_runtime/src/lib.rs` — 8 hand-written `*Bindings { ... }` literals (search line 669 onward), each 25-30 lines.
  - `crates/village_economy_runtime/src/lib.rs::step()` — same shape, 6-7 dispatch sites.
  - `crates/engine/src/gpu/event_ring.rs::EventRing` — already centralized; provides `.ring()` + `.tail()` + lifecycle helpers.
  - `crates/engine/src/state/mod.rs::SimState` — owns per-agent SoA buffers + accessors (`agent_hp_buf` is a field; PR #46 added `hot_ability_power` mirror buffer).
  - `crates/engine/src/ability/registry_gpu.rs::PackedAbilityRegistryGpu` — owns per-registry buffers (`when_pred_field` etc.).
  - `OUT_DIR/generated.rs` per runtime — emits `<KernelName>Bindings` structs with `&'a wgpu::Buffer` fields named after the kernel's bound names.

  Search method: `rg`.

- **Decision:** extend the compiler's BGL emitter to also emit a `Bindings::from_context(&KernelBindingsContext)` constructor for every kernel. Define `KernelBindingsContext<'a>` in `engine` as the canonical bundle of shared buffer sources: `&SimState`, `&EventRing`, `&PackedAbilityRegistryGpu`. Per-fixture extras (custom mask bitmaps, scoring output buffers, fixture-specific SoA columns like `agent_creature_type` that don't live on `SimState` today) ride a per-kernel `<KernelName>Extras` struct as a trailing argument: `Bindings::from_context(&ctx, &extras)`.

  Why constructor not trait: simpler to read at the call site, no generic plumbing, and the compiler can emit it via the same template that emits the struct definition.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: NONE.
  - Generated outputs re-emitted: every per-runtime `OUT_DIR/generated.rs` (the `Bindings` structs gain a `from_context()` impl). All ~40 runtime crates regenerate, but the per-runtime hand-written code shrinks dramatically.
  - Engine-side: new `KernelBindingsContext<'a>` struct in `crates/engine/src/gpu/`.

- **Hand-written downstream code:** per-fixture migrations replace 30-line literals with `from_context(&ctx, &extras)` calls. Per-fixture `Extras` structs stay handrolled (small — typically 3-5 fields) but they're scoped to per-fixture quirks rather than re-spelling shared buffers every time.

- **Constitution check:**
  - P1 (Compiler-First): PASS — the constructor is compiler-emitted alongside the Bindings struct it constructs. No new hand-written kernel code; less hand-written dispatch code.
  - P2 (Schema-Hash on Layout): PASS — pure host-side dispatch refactor; no SoA / event / mask-predicate changes.
  - P3 (Cross-Backend Parity): N/A — host-side only.
  - P4: N/A.
  - P5: PASS — deterministic; just buffer reference plumbing.
  - P6: N/A.
  - P7: N/A.
  - P8: PASS — this section.
  - P9: PASS.
  - P10: PASS — compile-time errors if a kernel needs a buffer the context doesn't expose. No new panic surface.
  - P11: N/A.

- **Runtime gate:**
  - Each migrated runtime's existing behavioral pin (e.g., `spy_network::noble_dies_after_slander_cascade`) MUST still pass after migration.
  - A new compiler test `bindings_from_context_emits_for_every_kernel` asserts every emitted `*Bindings` struct has a corresponding `from_context` impl.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design.

---

## Design

### `KernelBindingsContext<'a>` — the shared bundle

```rust
// In crates/engine/src/gpu/mod.rs (or a new bindings_context.rs)
pub struct KernelBindingsContext<'a> {
    pub state: &'a SimState,
    pub event_ring: &'a EventRing,
    pub registry: &'a PackedAbilityRegistryGpu,
}
```

Three references = covers the bulk of every existing dispatch's bindings. The compiler's `from_context()` walks the kernel's binding list and pulls each buffer from the appropriate source by naming convention:

- `event_ring` / `event_tail` → `ctx.event_ring.ring()` / `ctx.event_ring.tail()`
- `agent_*` (e.g. `agent_hp`, `agent_pos`, `agent_ability_power`) → `&ctx.state.<field>_buf` (per-agent SoA, lives on SimState)
- `ability_registry_*` → `&ctx.registry.<field>` (lives on PackedAbilityRegistryGpu)
- `spatial_grid_*` → `&ctx.state.spatial.<field>` (per-fixture spatial hash, lives on SimState)

The naming convention is **already** the contract today (the compiler emits `agent_hp` field name; the runtime supplies `agent_hp_buf`). Formalizing it via the compiler-side getter avoids per-runtime divergence.

### `<KernelName>Extras` — per-fixture quirks

Some buffers don't live on the shared sources:
- Per-fixture mask bitmaps (`mask_0_bitmap_buf`, `mask_1_bitmap_buf`, ...)
- Per-fixture scoring output buffer (`scoring_output_buf`)
- Fixture-specific SoA columns (e.g., `agent_creature_type_buf` in spy_network)
- Per-runtime cfg uniform buffers (`mask_cfg_buf`, `scoring_cfg_buf`)

For these, the compiler emits a sibling `<KernelName>Extras` struct + a `from_context_with_extras(ctx, extras)` constructor:

```rust
// Compiler-emitted:
pub struct PhysicsDispatchAoePulseExtras<'a> {
    pub mask_0_bitmap: &'a wgpu::Buffer,
    pub mask_1_bitmap: &'a wgpu::Buffer,
    pub scoring_output: &'a wgpu::Buffer,
    pub mask_cfg: &'a wgpu::Buffer,
}

impl<'a> PhysicsDispatchAoePulseBindings<'a> {
    pub fn from_context_with_extras(
        ctx: &KernelBindingsContext<'a>,
        extras: &PhysicsDispatchAoePulseExtras<'a>,
    ) -> Self { /* ... */ }
}
```

When a kernel has zero extras, the simpler `from_context()` impl is generated instead.

### Naming-convention rules (the compiler's getter table)

The compiler scans each emitted Bindings field and maps to a source via:

| Field name pattern | Source | Getter |
|---|---|---|
| `event_ring` | ctx.event_ring | `.ring()` |
| `event_tail` | ctx.event_ring | `.tail()` |
| `agent_<col>` | ctx.state | `&.{<col>}_buf` (or via `.{col}_buf()` accessor if private) |
| `ability_registry_<col>` | ctx.registry | `&.{<col>}` |
| `spatial_grid_<col>` | ctx.state.spatial | `&.{<col>}` |
| `cfg` (or any name not matching above) | extras | `.{name}` |

The "extras" bucket catches anything the convention doesn't cover. Per-fixture authors writing new fixtures don't need to invent: if a buffer doesn't fit the standard sources, it goes in extras.

### Future-proofing for `DebugWgslFlags`

Phase 2's instrumentation buffers (`event_kind_counts`, `mask_total`, `mask_passed`, `score_kernel_visits`) are already named consistently. Add to the convention table:

| `event_kind_counts` | ctx.debug | `&.event_kind_counts` |
| `mask_total` / `mask_passed` | ctx.debug | `&.mask_total[<id>]` / `&.mask_passed[<id>]` |
| `score_kernel_visits` | ctx.debug | `&.score_kernel_visits` |

Add an optional `pub debug: Option<&'a DebugBuffers>` field to `KernelBindingsContext`. Constructors generated when the flag is set assert `debug.is_some()` at compile time (or runtime panic with clear message).

---

## Tasks

| # | Task | Files | Description |
|---|---|---|---|
| 1 | Define `KernelBindingsContext` in engine | `crates/engine/src/gpu/{mod.rs,bindings_context.rs}` | New struct + accessor methods. Three required fields (state, event_ring, registry). Optional `debug` field for Phase 2 buffers. |
| 2 | Compiler emits `from_context()` per Bindings struct | `crates/dsl_compiler/src/cg/emit/bindings.rs` (or wherever `*Bindings` structs are emitted today — search for `<KernelName>Bindings`) | Walk each kernel's binding list; emit a `from_context(ctx)` (or `from_context_with_extras(ctx, extras)`) impl per the naming convention table. Emit `<KernelName>Extras` struct when extras are needed. |
| 3 | Compiler test for completeness | `crates/dsl_compiler/src/cg/emit/bindings.rs::tests` | `bindings_from_context_emits_for_every_kernel` — assert every emitted `*Bindings` struct has either `from_context` or `from_context_with_extras`. |
| 4 | Pilot migration: `debug_probe_runtime` | `crates/debug_probe_runtime/src/lib.rs` | Smallest, most recent runtime — easiest to verify the migration. Replace each `Bindings { ... }` literal with `Bindings::from_context_with_extras(&ctx, &extras)`. Behavioral pin must still pass. |
| 5 | Migrate the 2 stress runtimes | `crates/stress_agent_count_runtime/src/lib.rs`, `crates/stress_cast_density_runtime/src/lib.rs` | Mechanical. Behavioral pins must still pass. NDJSON output unchanged. |
| 6 | Migrate the 4 "real" production runtimes | `crates/spy_network_runtime/src/lib.rs`, `crates/village_economy_runtime/src/lib.rs`, `crates/duel_25v25_runtime/src/lib.rs`, `crates/wave_defense_runtime/src/lib.rs` (after #243 lands) | These have the most binding sites; biggest LOC reduction. All behavioral pins must still pass. |
| 7 | Migrate the remaining ~30 runtime crates | every other `crates/*_runtime/src/lib.rs` | Mechanical fanout. Most are smoke / probe runtimes with 1-2 dispatch sites. Can be parallelized across multiple agents (each touching disjoint runtime crates → no merge conflicts). |
| 8 | Doc update + CONTRIBUTING note | `docs/spec/engine.md` (the dispatch section) + a new `docs/architecture/runtime-bindings-pattern.md` | New runtime authors should construct `Bindings::from_context()` calls; only fall through to handrolled literals for genuinely fixture-specific buffers (which go in `<Kernel>Extras`). |

Tasks 1+2+3 are sequential — the compiler change must land first. Tasks 4+5+6 then run sequentially against the new API. Task 7 parallelizes across many agents (one agent per ~5 runtime crates is fine since they touch disjoint files). Task 8 lands last.

Total: substantial. Probably 3-5 PRs (tasks 1-3 = one PR; task 4 = one PR proving the migration shape; tasks 5+6 = one PR; task 7 = one or two PRs; task 8 = one PR).

## Pilot-first risk mitigation

Task 4 (migrating `debug_probe_runtime`) is the explicit pilot before fanning out. Reasons:
- Smallest fixture (one kernel-bearing dispatch); easiest to verify the new API works end-to-end.
- Recently authored — the binding shape is fresh in author memory.
- Already opts in to Phase 2 instrumentation, so it exercises the optional `debug` context field too.

If task 4 surfaces a design flaw (e.g., the convention table doesn't cover some buffer kind, or `from_context` has lifetime trouble in practice), iterate on tasks 1-3 before fanning out. Don't migrate 40 runtimes against a bad API.

## Out of scope (deferred)

- Migrating buffer ownership: keep per-fixture buffers like `mask_*_bitmap_buf` and `scoring_output_buf` on the runtime struct. Moving them onto `SimState` would eliminate even the Extras struct, but it's a much bigger refactor (changes SimState's shape + per-fixture init).
- Generic over wgpu Backend: the constructor is concrete `&'a wgpu::Buffer`; CPU backend (if it ever lands) needs its own context type.
- Cargo workspace test that runs every runtime's behavioral pin in CI: the existing `cargo test --workspace --release` already does this; no extra infra needed.
- Auto-deriving Debug / Clone on the new structs (not needed for dispatch — references are already cheap).

## Why now

Five recent slices each paid this fanout tax:

| PR / commit | What fanned out | Cost |
|---|---|---|
| #46 (AbilityPower SoA) | New per-agent f32 column | 11 runtimes × ~3 lines each |
| #55 (AbilityPower GPU bindings) | Per-runtime binding fanout | 11 runtimes × ~8 dispatch sites |
| `bcb490de` (cast_density miss) | Same fanout, missed target | 1 runtime, found post-merge |
| #58 (Phase 2 wiring) | 3 instrumentation buffer types | 3 runtimes × ~12 lines each |
| #243 (wave_defense, in flight) | Hand-spelling all bindings for a brand-new runtime | ~200 lines for the new runtime |

Each future engine extension that adds a buffer pays the same tax until this slice lands. Rough estimate: every ~2-3 weeks of engine work has shipped one fanout-causing change. ROI of compiler-emitted constructors compounds with every future buffer addition.
