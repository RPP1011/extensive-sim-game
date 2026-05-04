# Task 5.7 — Patch Specification: CG-pipeline switchover

**HEAD at dispatch:** `1dd20f9c`
**Spec author:** dispatch agent (no code changes)
**Apply target:** a separate apply agent reads this file and applies edits
sequentially. **Do not skip the `Apply order` section** — Patch 4 consumes
Patches 1, 2, and 3.

---

## Overview

Task 5.7 makes the CG pipeline canonical for `crates/engine_gpu_rules/src/`
output. Before flipping the switch, three pre-existing wire-incompatibility
gaps between CG-emitted artifacts and the legacy emitter (the consumer
runtime in `crates/engine_gpu/`) must be closed. After they close, a fourth
patch flips the xtask emitter from "legacy canonical, CG side-channel" to
"CG canonical (opt-in via `--cg-canonical`)" and a fifth patch gates the
result on `gpu_pipeline_smoke`.

**End-state (realistic):** `gpu_pipeline_smoke` passes when
`crates/engine_gpu_rules/src/` is regenerated via `xtask compile-dsl
--cg-canonical`. **`parity_with_cpu` is NOT a goal of 5.7** — it requires
AST coverage (Match/Fold/Quantifier/PerUnit/For for scoring + physics)
that's still deferred. parity_with_cpu is a separate iteration.

**Reversibility:** Patch 4 introduces an opt-in flag. Reverting the flag
returns to the legacy emit path. No engine_gpu code changes are made
unconditionally — the only engine_gpu touch is symbol-renames forced by
the CG emit naming choice (see Patch 4 §"Symbol contract").

---

## Pre-conditions

### HEAD SHA
`1dd20f9c1b1cb3b9815df26907e9e8bd0cd3928e` ("chore: disable critic skills +
hooks (drag > signal)")

### Baseline test status (read before any patch is applied)
- `cargo test -p dsl_compiler` — assumed passing at HEAD per dispatch note.
- `cargo test -p xtask` — 10 tests passing per dispatch note.
- `cargo test -p engine_gpu --features gpu --test gpu_pipeline_smoke` —
  **the apply agent MUST run this once before applying any patch and
  record the verdict.** The dispatch note flags one failure on a
  pre-existing test in `crates/dsl_compiler/tests/annotation_per_entity_ring.rs`
  (`per_entity_ring_emits_wgsl_fold_kernel`); that's unrelated to
  `gpu_pipeline_smoke`. If `gpu_pipeline_smoke` itself fails on the
  legacy emit at HEAD, the switchover gate (Patch 5) is automatically a
  no-op pass-through and the apply agent must record the pre-existing
  failure for follow-up rather than rolling back.

### Files to read

Cross-cutting CG synth:
- `crates/dsl_compiler/src/cg/emit/cross_cutting.rs` — full file.
  Patches 1–3 modify this file.
- `crates/dsl_compiler/src/cg/emit/program.rs:940-1033`
  (`synthesize_lib_rs`) — the CG-emitted `lib.rs` declares
  `pub enum KernelKind` (line 975); the legacy `lib.rs` declares
  `pub enum KernelId`. This is the symbol-contract divergence Patch 4
  must reconcile.

Legacy emitters (the contract CG must match):
- `crates/dsl_compiler/src/emit_resident_context.rs` — full file
  (264 lines). The `FoldViewSpec.primary_field` field is the alias hook
  Patch 1 mirrors via a hardcoded table.
- `crates/dsl_compiler/src/emit_schedule.rs` — full file (215 lines).
  The `DispatchOpKind` enum (lines 13–22) is the data shape Patch 3
  reproduces.

xtask:
- `crates/xtask/src/cli/mod.rs:160-276` — `CompileDslArgs`. Patch 4 adds
  one new flag here.
- `crates/xtask/src/compile_dsl_cmd.rs` — full file (2748 lines). The
  legacy emit driver (lines 198–2295) is the entry-point Patch 4 gates;
  the side-channel helper `emit_cg_side_channel` (lines 2378–2483) is
  what Patch 4 reuses.

Consumers (do not modify in 1–3, but Patch 4 must land symbol-renames
here):
- `crates/engine_gpu/src/lib.rs:644-918` — the
  `for op in engine_gpu_rules::schedule::SCHEDULE` dispatch loop.
  Today it matches on `KernelId::*` (line 802 onward). Patch 4 §
  "Symbol contract" decides whether CG emits `KernelId` or this consumer
  is renamed to `KernelKind`.
- `crates/engine_gpu/src/backend/resident_ctx.rs:58` —
  `pub path_ctx: engine_gpu_rules::resident_context::ResidentPathContext`.
- `crates/engine_gpu/src/physics.rs:89-90` — direct field access on
  `sources.resident.standing_primary` and `memory_primary`. The aliasing
  contract in Patch 1 keeps these field names alive.
- `crates/engine_gpu/tests/gpu_pipeline_smoke.rs` — full file (50 lines).
  Patch 5's gate target.

### Runtime contract refs

The `engine_gpu` runtime expects these load-bearing symbols on
`engine_gpu_rules`:

1. `engine_gpu_rules::schedule::SCHEDULE: &[DispatchOp]` — the
   schedule loop iterates this.
2. `engine_gpu_rules::schedule::DispatchOp::{Kernel, FixedPoint,
   Indirect, GatedBy}` — the four arms the dispatch matches.
3. The `KernelId` enum (current name) — Patch 4 decides the CG-side
   resolution.
4. `engine_gpu_rules::resident_context::ResidentPathContext` with fields
   `standing_primary`, `memory_primary` (direct field access at
   `engine_gpu/src/physics.rs:89-90`); fold accessor methods
   `fold_view_<name>_handles()` for every materialised view; and
   `scoring_view_buffers_slice()`.

---

## Patch 1: ResidentPathContext aliasing

### Goal

Make CG-emitted `fold_view_<name>_handles()` accessors return the same
buffer references the legacy emitter does for the three aliased views
(`standing`, `memory`, `engaged_with`). After this patch the legacy and
CG accessor bodies are byte-equal for all materialised views (modulo
header comment phrasing, which is not load-bearing).

### Approach: option (b) — driver-level alias table

A hardcoded table in `cross_cutting.rs` maps each materialised-view
snake-name to the resident field its `fold_view_<name>_handles()`
accessor reads. The table mirrors the legacy `match name.as_str()` at
`crates/xtask/src/compile_dsl_cmd.rs:1436-1440`:

```rust
match name {
    "standing"     => "standing_primary",
    "memory"       => "memory_primary",
    "engaged_with" => "standing_primary",   // intentional alias
    other          => format!("view_storage_{other}"),
}
```

This is the pragmatic patch for 5.7. Options (a) IR-level alias hint and
(c) update engine_gpu to expect distinct view buffers per view are
documented as future refinements in the limitations docstring.

### File: `crates/dsl_compiler/src/cg/emit/cross_cutting.rs`

#### Edit 1.1 — add the alias resolver helper

**Anchor:** insert immediately after the `RESIDENT_FIXED_FIELDS`
definition (after line 88, before the
`// ---------------------------------------------------------------------------`
divider that introduces "Public API" at line 90).

**Add:**

```rust
/// Resolve a materialised-view snake-name to the resident-field it
/// aliases for the `fold_view_<name>_handles()` accessor's primary
/// return.
///
/// Mirrors the legacy aliasing in `crates/xtask/src/compile_dsl_cmd.rs`
/// (the `match name.as_str()` near line 1436): the `standing`,
/// `memory`, and `engaged_with` views all live in two pre-existing
/// resident fields (`standing_primary`, `memory_primary`); the
/// remainder live under `view_storage_<name>`.
///
/// **Why this is a hardcoded table and not an IR property.** The legacy
/// emitter folds three concerns into one alias decision:
///   1. `standing` and `memory` use specialised storage shapes
///      (SymmetricPairTopK, PerEntityRing) the CG IR doesn't yet
///      surface as per-view storage-hint metadata.
///   2. `engaged_with`'s `Agent`/`AgentId` return-type mismatch causes
///      `classify_view` to reject it; the legacy fold kernel for it
///      is dead code (gated off by default).
///   3. Future plans (the `(a)` IR-level alias hint and `(c)` distinct
///      buffers per view in `engine_gpu`) supersede this table.
///
/// This patch matches the legacy contract byte-for-byte so the
/// switchover (Task 5.7 Patch 4) doesn't require an engine_gpu code
/// change. The table is the load-bearing alias surface until one of
/// the future plans lands.
fn resident_primary_field_for_view(view_name: &str) -> String {
    match view_name {
        "standing" => "standing_primary".to_string(),
        "memory" => "memory_primary".to_string(),
        // `engaged_with`'s legacy fold falls back to `standing_primary`
        // because `classify_view` rejects it (return-type mismatch).
        // The dispatch is gated off by default, so this never runs in
        // production today; the alias keeps emitted code compiling.
        "engaged_with" => "standing_primary".to_string(),
        other => format!("view_storage_{other}"),
    }
}
```

#### Edit 1.2 — use the resolver in the per-view accessor body

**Anchor:** in `synthesize_resident_context`, the per-view accessor
emit loop. The current line at the relevant spot is:

```rust
        writeln!(
            out,
            "        (&self.view_storage_{view}, None, None)"
        ).expect("write to String");
```

(this is at approximately line 248–251 of `cross_cutting.rs`).

**Replace with:**

```rust
        let primary_field = resident_primary_field_for_view(view);
        writeln!(
            out,
            "        (&self.{primary_field}, None, None)"
        ).expect("write to String");
```

#### Edit 1.3 — guard against alias collisions in the field-emit loop

**Context.** The per-view storage-field emit loop in
`synthesize_resident_context` (the loop that runs from approximately
line 164 onward, immediately after the `RESIDENT_FIXED_FIELDS` loop)
emits `pub view_storage_<view>: wgpu::Buffer` for every materialised
view. The aliased views (`standing`, `memory`, `engaged_with`) must NOT
get a `view_storage_<view>` field — those views' primary buffers are
the existing `standing_primary` / `memory_primary` fields already
declared in `RESIDENT_FIXED_FIELDS`. Without this guard, the emitted
struct would carry `view_storage_standing` AND `standing_primary`,
both 256 bytes, both initialised — wasteful but more importantly,
inconsistent with the legacy contract.

**Anchor:** the per-view storage-field loop. Before the patch it reads:

```rust
    for view in &view_names {
        writeln!(
            out,
            "    /// Resident view storage for `{view}`."
        ).expect("write to String");
        writeln!(out, "    pub view_storage_{view}: wgpu::Buffer,").expect("write to String");
    }
```

**Replace with:**

```rust
    for view in &view_names {
        // Skip per-view storage fields for views that alias to a
        // pre-existing resident field (see
        // `resident_primary_field_for_view`). Emitting both would
        // double-allocate the placeholder and diverge from the legacy
        // resident-context shape consumed by `engine_gpu`.
        if !resident_primary_field_for_view(view).starts_with("view_storage_") {
            continue;
        }
        writeln!(
            out,
            "    /// Resident view storage for `{view}`."
        ).expect("write to String");
        writeln!(out, "    pub view_storage_{view}: wgpu::Buffer,").expect("write to String");
    }
```

#### Edit 1.4 — guard the per-view init loop in `new()`

**Anchor:** the per-view init loop inside `pub fn new(...) -> Self`.
Before the patch it reads:

```rust
    for view in &view_names {
        let field = format!("view_storage_{view}");
        emit_buffer_init(&mut out, &field);
    }
```

**Replace with:**

```rust
    for view in &view_names {
        // Aliased views (standing, memory, engaged_with) get their
        // buffer from `RESIDENT_FIXED_FIELDS` — no per-view init.
        // Mirrors the field-emit guard above.
        let field = resident_primary_field_for_view(view);
        if !field.starts_with("view_storage_") {
            continue;
        }
        emit_buffer_init(&mut out, &field);
    }
```

#### Edit 1.5 — guard the scoring-view-cache list

**Anchor:** the `for view in &view_names` loop inside the
`scoring_view_buffers_slice()` synth (around lines 209–214 today):

```rust
        for view in &view_names {
            writeln!(
                out,
                "                unsafe {{ &*((&self.view_storage_{view}) as *const wgpu::Buffer) }},"
            ).expect("write to String");
        }
```

**Replace with:**

```rust
        for view in &view_names {
            let primary_field = resident_primary_field_for_view(view);
            writeln!(
                out,
                "                unsafe {{ &*((&self.{primary_field}) as *const wgpu::Buffer) }},"
            ).expect("write to String");
        }
```

This produces the same ordering as before (interner order); Patch 2
narrows the *list* of views included.

### Tests for Patch 1

In `cross_cutting.rs`'s `mod tests` section add:

```rust
#[test]
fn resident_context_aliases_standing_to_standing_primary() {
    let (prog, _) = one_view_fold_program(11, "standing");
    let src = synthesize_resident_context(&prog);
    assert!(
        src.contains("pub fn fold_view_standing_handles<'a>"),
        "accessor must exist: {src}"
    );
    assert!(
        src.contains("(&self.standing_primary, None, None)"),
        "standing must alias to standing_primary: {src}"
    );
    assert!(
        !src.contains("pub view_storage_standing: wgpu::Buffer"),
        "standing must NOT get its own view_storage field: {src}"
    );
}

#[test]
fn resident_context_aliases_memory_to_memory_primary() {
    let (prog, _) = one_view_fold_program(12, "memory");
    let src = synthesize_resident_context(&prog);
    assert!(
        src.contains("(&self.memory_primary, None, None)"),
        "memory must alias to memory_primary: {src}"
    );
    assert!(
        !src.contains("pub view_storage_memory: wgpu::Buffer"),
        "memory must NOT get its own view_storage field: {src}"
    );
}

#[test]
fn resident_context_aliases_engaged_with_to_standing_primary() {
    let (prog, _) = one_view_fold_program(13, "engaged_with");
    let src = synthesize_resident_context(&prog);
    assert!(
        src.contains("(&self.standing_primary, None, None)"),
        "engaged_with must alias to standing_primary: {src}"
    );
    assert!(
        !src.contains("pub view_storage_engaged_with: wgpu::Buffer"),
        "engaged_with must NOT get its own view_storage field: {src}"
    );
}

#[test]
fn resident_context_non_aliased_view_keeps_view_storage_field() {
    let (prog, _) = one_view_fold_program(14, "threat_level");
    let src = synthesize_resident_context(&prog);
    assert!(
        src.contains("pub view_storage_threat_level: wgpu::Buffer"),
        "non-aliased view must keep its view_storage_<name> field: {src}"
    );
    assert!(
        src.contains("(&self.view_storage_threat_level, None, None)"),
        "non-aliased view's accessor body unchanged: {src}"
    );
}
```

The two pre-existing tests
(`resident_context_emits_one_view_storage_field_per_materialised_view`
and `resident_context_emits_fold_view_handles_accessor_per_view`,
~line 684 and 694) continue to use `view_id=3, name="threat_level"`,
which is non-aliased. They stay green without changes.

The pre-existing
`resident_context_emits_fixed_fields_with_no_views` test (~line 660)
asserts `!src.contains("view_storage_")`. With this patch in place it
still holds for the empty-view case. Stays green.

---

## Patch 2: scoring_view_buffers_slice ordering

### Goal

CG-emitted `scoring_view_buffers_slice()` returns the same ordered
subset of view buffers the legacy emitter does — namely the view names
in `scoring_view_binding_order` order, NOT every materialised view in
interner order.

### Background

The legacy ordering, derived from
`crates/xtask/src/compile_dsl_cmd.rs:719-743`, is the alphabetical sort
of materialised non-Lazy view names that successfully classify under
`emit_view_wgsl::classify_view` (a subset of `prog.interner.views`).

### Approach

Add a hardcoded ordering list in `cross_cutting.rs` that matches the
five views the legacy emitter actually emits to
`scoring_view_buffers_slice`: `kin_fear`, `my_enemies`, `pack_focus`,
`rally_boost`, `threat_level`. The aliased views (`standing`, `memory`,
`engaged_with`) are NOT in the slice — that's the legacy contract.

(Verification: `crates/engine_gpu_rules/src/resident_context.rs:138-153`
at HEAD lists exactly those five.)

### File: `crates/dsl_compiler/src/cg/emit/cross_cutting.rs`

#### Edit 2.1 — define the binding order list

**Anchor:** insert immediately after the `RESIDENT_FIXED_FIELDS` block
and before `resident_primary_field_for_view` (Patch 1). If applying in
the recommended order, this lands right after the table from Patch 1.

**Add:**

```rust
/// View names the scoring kernel binds in `bind()` order. Mirrors the
/// legacy `scoring_view_binding_order` (the alphabetical-by-snake-name
/// sort of materialised non-Lazy views that pass
/// `emit_view_wgsl::classify_view`).
///
/// **Why hardcoded.** The CG IR's `prog.interner.views` includes
/// every view the lowering pass touched, in interner order — a
/// superset that includes aliased views (`standing`, `memory`,
/// `engaged_with`) the legacy emitter excludes. Threading the
/// classify-and-sort logic through the CG pipeline is a future
/// refinement (the IR would carry a `materialised_for_scoring: bool`
/// per-view flag); this hardcoded list closes the gap for Task 5.7
/// without that plumbing.
///
/// Adding a new scoring view to the DSL: append the snake-name here
/// AND ensure the view either (a) gets a `view_storage_<name>`
/// resident field (the default) or (b) is added to
/// `resident_primary_field_for_view`'s alias table. Failing to do
/// either yields an emit-time `compile_error` or a wgpu validation
/// panic at runtime — both are loud.
const SCORING_VIEW_BINDING_ORDER: &[&str] = &[
    "kin_fear",
    "my_enemies",
    "pack_focus",
    "rally_boost",
    "threat_level",
];
```

#### Edit 2.2 — drive the slice from the ordered list, not interner order

**Anchor:** the `scoring_view_buffers_slice()` synth body. Currently
the relevant fragment is:

```rust
    if view_names.is_empty() {
        out.push_str("        &[]\n");
    } else {
        out.push_str("        let v = self.scoring_view_buffers_cache.get_or_init(|| {\n");
        // ...
        for view in &view_names {
            writeln!(
                out,
                "                unsafe {{ &*((&self.view_storage_{view}) as *const wgpu::Buffer) }},"
            ).expect("write to String");
        }
        // ...
    }
```

(after Patch 1.5 the inner write uses `primary_field` rather than
`view_storage_{view}` — see Patch 1.5).

**Replace with:**

```rust
    // Compute the intersection of `SCORING_VIEW_BINDING_ORDER` and the
    // views actually materialised in `prog`. The legacy emitter walks
    // `combined.views` and filters by `classify_view`; here we approximate
    // by intersecting the hardcoded order with the IR's
    // `view_names` set so the slice never references a view the program
    // didn't materialise.
    let materialised_set: std::collections::BTreeSet<&str> =
        view_names.iter().map(String::as_str).collect();
    let scoring_views: Vec<&str> = SCORING_VIEW_BINDING_ORDER
        .iter()
        .copied()
        .filter(|n| materialised_set.contains(n))
        .collect();

    if scoring_views.is_empty() {
        out.push_str("        &[]\n");
    } else {
        out.push_str("        let v = self.scoring_view_buffers_cache.get_or_init(|| {\n");
        out.push_str("            // SAFETY: the &wgpu::Buffer references inside `Self` live as long as\n");
        out.push_str("            // `Self` does. The OnceLock is dropped together with `Self`,\n");
        out.push_str("            // so the 'static cast is sound for as long as the cache exists.\n");
        out.push_str("            let raw: Vec<&'static wgpu::Buffer> = vec![\n");
        for view in &scoring_views {
            let primary_field = resident_primary_field_for_view(view);
            writeln!(
                out,
                "                unsafe {{ &*((&self.{primary_field}) as *const wgpu::Buffer) }},"
            ).expect("write to String");
        }
        out.push_str("            ];\n");
        out.push_str("            raw\n");
        out.push_str("        });\n");
        out.push_str("        // Re-borrow at the caller's lifetime.\n");
        out.push_str("        // Vec<&'static T> coerces to &[&T] here.\n");
        out.push_str("        unsafe { std::mem::transmute::<&[&'static wgpu::Buffer], &'a [&'a wgpu::Buffer]>(v.as_slice()) }\n");
    }
```

The empty-views fallback (`&[]`) and the `OnceLock` shape are unchanged.

### Tests for Patch 2

Append:

```rust
#[test]
fn scoring_view_buffers_slice_uses_binding_order_subset() {
    // Fixture: program materialises kin_fear (id=1), threat_level
    // (id=2), and a non-scoring view (id=99, name="some_other"). Slice
    // must list kin_fear before threat_level (binding order) and skip
    // some_other entirely (not in SCORING_VIEW_BINDING_ORDER).
    let mut prog = CgProgram::default();
    for (id, name) in [(1, "kin_fear"), (2, "threat_level"), (99, "some_other")] {
        prog.interner.views.insert(id, name.to_string());
    }
    // Use the same fold-op pattern as `one_view_fold_program` — copy
    // the body or extract a helper that takes a list of (id, name).
    // Spec note: implementer chooses the cleanest path.
    for (id, _) in [(1u32, "kin_fear"), (2, "threat_level"), (99, "some_other")] {
        // (Construct one ViewFold op per id; see `one_view_fold_program`
        // for the per-op shape — same pattern, just three iterations.)
        // ...
    }

    let src = synthesize_resident_context(&prog);
    let kin_pos = src.find("view_storage_kin_fear")
        .or_else(|| src.find("kin_fear"))
        .expect("kin_fear must appear in slice");
    let threat_pos = src.find("view_storage_threat_level")
        .or_else(|| src.find("threat_level"))
        .expect("threat_level must appear in slice");
    assert!(kin_pos < threat_pos, "kin_fear must precede threat_level");
    assert!(
        !src.contains("view_storage_some_other"),
        "some_other must NOT appear: {src}"
    );
}

#[test]
fn scoring_view_buffers_slice_emits_empty_when_no_scoring_views_materialised() {
    // Only an aliased view materialised — slice should be empty
    // (standing is not in SCORING_VIEW_BINDING_ORDER).
    let (prog, _) = one_view_fold_program(11, "standing");
    let src = synthesize_resident_context(&prog);
    // Find the scoring_view_buffers_slice body and check it's `&[]`.
    let slice_start = src.find("pub fn scoring_view_buffers_slice").expect("slice fn must exist");
    let after = &src[slice_start..];
    assert!(
        after.contains("        &[]"),
        "empty slice must be emitted when only aliased views materialised: {src}"
    );
}
```

The pre-existing test
`resident_context_emits_one_view_storage_field_per_materialised_view`
(uses view_id=3, name="threat_level", in binding order) still passes —
threat_level is in `SCORING_VIEW_BINDING_ORDER`, so the slice still
references its `view_storage_threat_level` field.

---

## Patch 3: SCHEDULE DispatchOp wrappers

### Goal

CG-emitted `SCHEDULE` constant emits the same `FixedPoint` and
`Indirect` wrappers the legacy emitter does:

- Physics-rule kernels → `DispatchOp::FixedPoint { kernel: ..., max_iter: 8 }`
- Indirect-topology kernels (SeedIndirect) →
  `DispatchOp::Indirect { kernel: ..., args_buf: BufferRef::ResidentIndirectArgs }`
- Everything else → `DispatchOp::Kernel(...)`

Without this, runtime correctness breaks: physics never fixpoint-loops,
seed-indirect never triggers indirect dispatch.

### Approach

In `synthesize_schedule`, classify each topology / kernel name pair:

1. `KernelTopology::Indirect { producer, .. }` — emit
   `DispatchOp::Indirect { kernel, args_buf: BufferRef::ResidentIndirectArgs }`.
2. `KernelTopology::{Fused,Split}` whose primary op is a
   `ComputeOpKind::PhysicsRule` AND whose semantic name resolves to
   the kernel currently named `"physics"` — emit
   `DispatchOp::FixedPoint { kernel, max_iter: 8 }`.
3. Otherwise — `DispatchOp::Kernel(kernel)` (current behaviour).

The kernel-name check in (2) is the bridge between IR-level
`PhysicsRule` ops and the legacy emit's hand-pinned "Physics"
identity. Today every PhysicsRule op currently lowers to the kernel
named `"physics"` (see `semantic_kernel_name_for_topology` in
`crates/dsl_compiler/src/cg/emit/kernel.rs`). The check is a future-proof
guard: if a future plan splits PhysicsRule across multiple kernels, the
emit needs explicit metadata about which is FixedPoint-eligible.

### File: `crates/dsl_compiler/src/cg/emit/cross_cutting.rs`

#### Edit 3.1 — replace the schedule-entry construction

**Anchor:** in `synthesize_schedule`, the loop that today reads:

```rust
    let mut entries: Vec<String> = Vec::new();
    for stage in &schedule.stages {
        for topology in &stage.kernels {
            match semantic_kernel_name_for_topology(topology, prog) {
                Some(name) => entries.push(name),
                None => continue,
            }
        }
    }

    if entries.is_empty() {
        out.push_str("pub const SCHEDULE: &[DispatchOp] = &[];\n");
        return out;
    }

    out.push_str("pub const SCHEDULE: &[DispatchOp] = &[\n");
    for name in &entries {
        let pascal = snake_to_pascal(name);
        writeln!(out, "    DispatchOp::Kernel(KernelKind::{pascal}),")
            .expect("write to String");
    }
    out.push_str("];\n");
    out
```

**Replace with:**

```rust
    /// One entry in the synthesised schedule, paired with the dispatch-op
    /// shape it should emit. Kept local to the synth body — public callers
    /// see only the rendered `&str`.
    enum ScheduleEntry<'a> {
        Kernel(String),
        FixedPoint { kernel: String, max_iter: u32 },
        Indirect { kernel: String, args_buf: &'a str },
    }

    let mut entries: Vec<ScheduleEntry> = Vec::new();
    for stage in &schedule.stages {
        for topology in &stage.kernels {
            let name = match semantic_kernel_name_for_topology(topology, prog) {
                Some(n) => n,
                None => continue,
            };
            let entry = classify_topology_for_schedule(topology, &name, prog);
            entries.push(entry);
        }
    }

    if entries.is_empty() {
        out.push_str("pub const SCHEDULE: &[DispatchOp] = &[];\n");
        return out;
    }

    out.push_str("pub const SCHEDULE: &[DispatchOp] = &[\n");
    for entry in &entries {
        match entry {
            ScheduleEntry::Kernel(name) => {
                let pascal = snake_to_pascal(name);
                writeln!(out, "    DispatchOp::Kernel(KernelKind::{pascal}),")
                    .expect("write to String");
            }
            ScheduleEntry::FixedPoint { kernel, max_iter } => {
                let pascal = snake_to_pascal(kernel);
                writeln!(
                    out,
                    "    DispatchOp::FixedPoint {{ kernel: KernelKind::{pascal}, max_iter: {max_iter} }},"
                )
                .expect("write to String");
            }
            ScheduleEntry::Indirect { kernel, args_buf } => {
                let pascal = snake_to_pascal(kernel);
                writeln!(
                    out,
                    "    DispatchOp::Indirect {{ kernel: KernelKind::{pascal}, args_buf: BufferRef::{args_buf} }},"
                )
                .expect("write to String");
            }
        }
    }
    out.push_str("];\n");
    out
```

(A free-standing `ScheduleEntry` enum is fine — the helper is only used
inside this fn. If the apply agent prefers a top-of-module type, that's
also acceptable, but keeping it local matches the existing CG-emit
style.)

#### Edit 3.2 — add the classifier helper

**Anchor:** insert immediately after `synthesize_schedule` (and before
the `// Helpers` divider near line 491).

**Add:**

```rust
/// Decide which `DispatchOp` variant the synthesised SCHEDULE should
/// emit for `topology` (already named `kernel_name`).
///
/// Today three rules apply:
///
/// 1. [`KernelTopology::Indirect`] producer/consumer pairs → the
///    consumer-side dispatch is `DispatchOp::Indirect`, with
///    `args_buf` pinned to `BufferRef::ResidentIndirectArgs` (the only
///    `BufferRef` variant the runtime currently routes through). The
///    producer (`SeedIndirectArgs` plumbing) emits its own kernel
///    entry one stage earlier and keeps `DispatchOp::Kernel(...)`
///    classification.
///
/// 2. [`KernelTopology::Split`] / [`KernelTopology::Fused`] whose body
///    is a `ComputeOpKind::PhysicsRule` AND whose semantic name is
///    `"physics"` → `DispatchOp::FixedPoint { max_iter: 8 }`. The
///    `max_iter: 8` matches the legacy
///    `crates/xtask/src/compile_dsl_cmd.rs:1283` value; threading
///    a per-rule `@cascade(max_iter=N)` annotation through the IR is
///    a future refinement.
///
/// 3. Everything else → `DispatchOp::Kernel(...)`.
///
/// **Why the kernel-name guard in rule 2.** Today every `PhysicsRule`
/// op lowers to the kernel named `"physics"`. The guard is forward-
/// looking: a future plan that splits PhysicsRule across multiple
/// kernels would need explicit per-kernel FixedPoint metadata, and
/// the guard surfaces that future work as an emit-time miss rather
/// than a runtime correctness drift.
fn classify_topology_for_schedule<'a>(
    topology: &crate::cg::schedule::synthesis::KernelTopology,
    kernel_name: &str,
    prog: &CgProgram,
) -> ScheduleEntry<'a> {
    use crate::cg::schedule::synthesis::KernelTopology;
    match topology {
        KernelTopology::Indirect { .. } => ScheduleEntry::Indirect {
            kernel: kernel_name.to_string(),
            args_buf: "ResidentIndirectArgs",
        },
        KernelTopology::Split { op, .. } | KernelTopology::Fused { ops: _, .. } => {
            // For Fused, classify on the FIRST op (today every fused
            // kernel is single-op or has homogeneous classification —
            // mixing PhysicsRule with non-PhysicsRule in one fused
            // kernel would be a structural mismatch).
            let primary_op_id = match topology {
                KernelTopology::Split { op, .. } => *op,
                KernelTopology::Fused { ops, .. } => match ops.first() {
                    Some(o) => *o,
                    None => return ScheduleEntry::Kernel(kernel_name.to_string()),
                },
                KernelTopology::Indirect { .. } => unreachable!(),
            };
            let _ = op; // silence unused-binding warning from Split arm above
            let is_physics_rule = prog
                .ops
                .get(primary_op_id.0 as usize)
                .map(|op| matches!(op.kind, ComputeOpKind::PhysicsRule { .. }))
                .unwrap_or(false);
            if is_physics_rule && kernel_name == "physics" {
                ScheduleEntry::FixedPoint {
                    kernel: kernel_name.to_string(),
                    max_iter: 8,
                }
            } else {
                ScheduleEntry::Kernel(kernel_name.to_string())
            }
        }
    }
}
```

(The apply agent should adjust the `let _ = op` line if their match
binding-pattern preference produces clean unused-warning-free code.
The intent is: extract the primary OpId, look it up in `prog.ops`,
check `ComputeOpKind::PhysicsRule`.)

### Tests for Patch 3

The pre-existing `schedule_synthesises_dispatch_op_enum_and_const_from_stages`
(line 715) and `schedule_with_empty_stages_emits_empty_const` (line 738)
both stay green — the first uses a `ViewFold` op (Split topology, not
PhysicsRule), so it still emits as `DispatchOp::Kernel(...)`.

Add:

```rust
#[test]
fn schedule_emits_indirect_for_indirect_topology() {
    use crate::cg::schedule::synthesis::{ComputeStage, KernelTopology};
    use crate::cg::data_handle::EventRingId;
    use crate::cg::op::{ComputeOp, ComputeOpKind, OpId, PlumbingKind, Span};
    use crate::cg::dispatch::DispatchShape;

    let mut prog = CgProgram::default();
    prog.interner
        .event_kinds
        .insert(7, "AgentAttacked".to_string());

    // Producer: Plumbing::SeedIndirectArgs.
    let producer_kind = ComputeOpKind::Plumbing {
        kind: PlumbingKind::SeedIndirectArgs {
            ring: EventRingId(0),
        },
    };
    let producer_op = ComputeOp::new(
        OpId(0),
        producer_kind,
        DispatchShape::OneShot,
        Span::dummy(),
        &prog, &prog, &prog,
    );
    let producer_id = OpId(prog.ops.len() as u32);
    prog.ops.push(producer_op);

    // Consumer: a PerEvent op (use any kind — emit treats Indirect
    // by topology, not op kind, on the consumer side).
    // ... construct one. For brevity in this spec, the apply agent
    // may reuse `one_view_fold_program`'s ViewFold construction and
    // stitch it under an Indirect topology.

    let topology = KernelTopology::Indirect {
        producer: producer_id,
        consumers: vec![/* fill in */],
    };
    let schedule = ComputeSchedule {
        stages: vec![ComputeStage { kernels: vec![topology] }],
    };
    let src = synthesize_schedule(&schedule, &prog);
    assert!(
        src.contains("DispatchOp::Indirect"),
        "Indirect topology must emit DispatchOp::Indirect: {src}"
    );
    assert!(
        src.contains("BufferRef::ResidentIndirectArgs"),
        "Indirect emit must reference ResidentIndirectArgs: {src}"
    );
}

#[test]
fn schedule_emits_fixed_point_for_physics_rule() {
    use crate::cg::schedule::synthesis::{ComputeStage, KernelTopology};
    use crate::cg::data_handle::{EventKindId, EventRingId};
    use crate::cg::op::{ComputeOp, ComputeOpKind, OpId, PhysicsRuleId, ReplayabilityFlag, Span};
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::stmt::{CgStmtList, CgStmtListId};

    let mut prog = CgProgram::default();
    prog.interner
        .event_kinds
        .insert(7, "AgentAttacked".to_string());
    let empty_list = CgStmtList { stmts: vec![] };
    let list_id = CgStmtListId(prog.stmt_lists.len() as u32);
    prog.stmt_lists.push(empty_list);

    let physics_kind = ComputeOpKind::PhysicsRule {
        rule: PhysicsRuleId(0),
        on_event: EventKindId(7),
        body: list_id,
        // The `replayable` field's exact constructor depends on the
        // `ReplayabilityFlag` type's API; the apply agent uses the
        // default-equivalent variant. The flag is irrelevant to the
        // FixedPoint classification.
        replayable: ReplayabilityFlag::default(),
    };
    let physics_op = ComputeOp::new(
        OpId(0),
        physics_kind,
        DispatchShape::PerEvent { source_ring: EventRingId(0) },
        Span::dummy(),
        &prog, &prog, &prog,
    );
    let op_id = OpId(prog.ops.len() as u32);
    prog.ops.push(physics_op);

    let topology = KernelTopology::Split {
        op: op_id,
        dispatch: DispatchShape::PerEvent { source_ring: EventRingId(0) },
    };
    let schedule = ComputeSchedule {
        stages: vec![ComputeStage { kernels: vec![topology] }],
    };
    let src = synthesize_schedule(&schedule, &prog);
    // The kernel name will resolve to "physics" via
    // semantic_kernel_name_for_topology if the IR is shaped like the
    // production lowering; if the test fixture's name doesn't resolve
    // to "physics", the test should be adjusted to either inject the
    // name or exercise just the classifier helper directly.
    assert!(
        src.contains("DispatchOp::FixedPoint")
            && src.contains("max_iter: 8"),
        "PhysicsRule named `physics` must emit FixedPoint with max_iter: 8: {src}"
    );
}
```

If wiring `semantic_kernel_name_for_topology` to return `"physics"`
in the test fixture is friction, the apply agent has two choices:

(a) Drop the second test and rely on integration-level coverage from
    `gpu_pipeline_smoke` (Patch 5). The first test (Indirect) is
    already a robust unit-level guard.
(b) Refactor `classify_topology_for_schedule` to accept the kernel
    name as `&str` (already does in this spec) and unit-test it
    directly without going through `synthesize_schedule`.

Option (b) is cleaner. The apply agent picks.

---

## Patch 4: xtask `--cg-canonical` flag

### Goal

Add a `--cg-canonical` flag to `compile-dsl`. When set:
1. The CG pipeline runs and writes `EmittedArtifacts` directly into
   `crates/engine_gpu_rules/src/`.
2. The legacy emit path is **skipped** for the per-kernel + cross-cutting
   files (otherwise it would clobber the CG output).
3. The shared-data emissions (`engine_data` events / schema, `engine_rules`
   step / backend / mask_fill / cascade_reg, Python events / enums) STILL
   run from the legacy path — the CG pipeline doesn't cover those today
   and dropping them would break dependent crates.

When `--cg-canonical` is NOT set: behaviour is unchanged. Legacy emit is
canonical; `--cg-emit-into <dir>` (existing) writes a side-channel diff.

### Symbol contract

The CG-emitted `lib.rs` declares `pub enum KernelKind` (not `KernelId`).
The engine_gpu dispatch loop (`crates/engine_gpu/src/lib.rs:644-918`)
matches on `KernelId`. To make the switchover compile:

**Option A:** make CG emit `KernelId` instead of `KernelKind`. Pure
mechanical rename in `cross_cutting.rs` (lines 448, 452–455, 483) and
`program.rs` (line 975 onwards in `synthesize_lib_rs`).

**Option B:** rename the engine_gpu dispatch sites from `KernelId` to
`KernelKind`. Mechanical rename, ~22 lines under `crates/engine_gpu/src/lib.rs`.

**Decision: Option A.** The legacy emitter is older; engine_gpu's
identifier was set from the legacy contract. CG emit hasn't shipped a
runtime consumer yet, so renaming on the CG side is the lower-risk move
and leaves engine_gpu untouched.

#### Edit 4.0a — rename `KernelKind` → `KernelId` in `synthesize_schedule`

**Anchor:** in `cross_cutting.rs::synthesize_schedule`, the four lines
that hard-code `KernelKind`:

- Line 448: `out.push_str("use crate::{KernelKind, BufferRef};\n");`
- Line 452: `    Kernel(KernelKind),\n`
- Line 453: `    FixedPoint { kernel: KernelKind, max_iter: u32 },\n`
- Line 454: `    Indirect { kernel: KernelKind, args_buf: BufferRef },\n`
- Line 455: `    GatedBy { kernel: KernelKind, gate: BufferRef },\n`
- Line 483 (inside the schedule entry loop): `KernelKind::{pascal}`

**Replace `KernelKind` with `KernelId` in all six positions.**

After Patch 3's edits, the new emit paths (`FixedPoint`/`Indirect` arms
in Patch 3 §3.1) also reference `KernelKind::` — those must use
`KernelId::` too. The apply agent applies this rename across the entire
post-Patch-3 `synthesize_schedule` body.

The pre-existing tests that assert on `KernelKind` strings:
- `cross_cutting.rs:730` — `assert!(src.contains("Kernel(KernelKind),")`
- `cross_cutting.rs:733` — `assert!(src.contains("DispatchOp::Kernel(KernelKind::"))`

Update these to assert on `KernelId` instead.

#### Edit 4.0b — rename `KernelKind` → `KernelId` in `synthesize_lib_rs`

**Anchor:** in `crates/dsl_compiler/src/cg/emit/program.rs:975`:

```rust
    out.push_str("pub enum KernelKind {\n");
```

**Replace with:**

```rust
    out.push_str("pub enum KernelId {\n");
```

Also update the related docstrings in that fn (lines 910, 932, 973)
and the test assertions at lines 1657-1658 and the line 1694 area
that reference `KernelKind`.

(These docstring references and test assertions are listed in the
`grep -rn "KernelKind"` output collected in pre-conditions; the apply
agent searches and replaces.)

#### Edit 4.0c — verify no other `KernelKind` references in CG-emit output

After 4.0a + 4.0b, run:

```bash
cd /home/ricky/Projects/game
grep -rn "KernelKind" crates/dsl_compiler/src/cg/
```

Every remaining match should be in IR-side type names (e.g.
`KernelKind` enum in `kernel_binding_ir.rs`) — those are internal IR
types, not what the EMITTED files reference. If any remaining match is
in a string literal or `.push_str(...)` call within the `cg/emit/`
subdirectory, the apply agent updates it the same way.

(`crates/dsl_compiler/src/kernel_binding_ir.rs` defines
`pub enum KernelKind { Generic, ViewFold }` — that's an IR-side type,
NOT the runtime registry enum. Leave it alone.)

### File: `crates/xtask/src/cli/mod.rs`

#### Edit 4.1 — add the flag to `CompileDslArgs`

**Anchor:** immediately after the `pub cg_emit_into: Option<PathBuf>`
field (line 275), before the closing `}` at line 276.

**Add:**

```rust
    /// Switchover: write CG-emitted `EmittedArtifacts` directly into
    /// `crates/engine_gpu_rules/src/` instead of the legacy emit path.
    ///
    /// When set:
    /// - The CG pipeline (lower → synthesize_schedule → emit_cg_program)
    ///   runs once and writes per-kernel WGSL + Rust modules,
    ///   `lib.rs`, and the cross-cutting modules (binding_sources,
    ///   resident_context, schedule, etc.) into
    ///   `crates/engine_gpu_rules/src/`.
    /// - The legacy `engine_gpu_rules` emit blocks (per-kernel files +
    ///   schedule + resident_context + lib.rs) are SKIPPED.
    /// - All other emissions (engine_data events / schema /
    ///   scoring / entities / enums / configs, engine_rules
    ///   step / backend / mask_fill / cascade_reg, Python events /
    ///   enums) run from the legacy path unchanged — the CG pipeline
    ///   doesn't cover those.
    ///
    /// When NOT set: behaviour unchanged from pre-Task-5.7 (legacy
    /// emit is canonical; `--cg-emit-into` produces an optional
    /// side-channel diff).
    ///
    /// **Reversibility.** Reverting to the legacy path is just
    /// dropping the flag and re-running `compile-dsl` — the legacy
    /// emit path overwrites the CG-written files. No engine_gpu
    /// code change is required to switch back.
    ///
    /// **Mutual exclusion.** `--cg-canonical` and `--cg-emit-into`
    /// can both be set simultaneously, but the `--cg-emit-into` path
    /// MUST NOT resolve under `crates/engine_gpu_rules/` (the existing
    /// `path_resolves_into_engine_gpu_rules` rejection still applies).
    /// The flags answer different questions: `--cg-emit-into` writes
    /// a SIDE-CHANNEL diff to a scratch dir; `--cg-canonical` flips
    /// the production overlay.
    #[arg(long)]
    pub cg_canonical: bool,
```

### File: `crates/xtask/src/compile_dsl_cmd.rs`

#### Edit 4.2 — early-return path when `--cg-canonical` is set

**Anchor:** in `run_compile_dsl`, immediately after the `compile_all`
result is bound (around line 33). Before the existing `if let Some(cg_dir)
= args.cg_emit_into.as_ref()` block (line 48).

**Reasoning.** The CG-canonical branch must run BEFORE the existing
`emit_cg_side_channel` path (so the user can supply both flags) but
AFTER `compile_all` has produced a `combined: Compilation` (which the
CG lowering needs). The legacy emit blocks (lines 198–2295) are
inside the `else` branch of `if args.check { ... } else { ... }`;
when `cg_canonical` is set we want to:

1. Run shared-data emissions (engine_data events, engine_rules step etc).
2. Run the CG pipeline against `crates/engine_gpu_rules`.
3. SKIP the per-kernel + cross-cutting emit blocks.

The cleanest implementation is a guard inside the `else` branch that
short-circuits the per-kernel emit blocks. **Approach:** introduce a
local `let cg_canonical = args.cg_canonical;` near the top of the `else`
block, and wrap the per-kernel emit blocks (the long sequence starting
"Per-kernel emit accumulators" at line 248 through the schedule write
at line 1478) in `if !cg_canonical { ... }`. After the wrapped block,
add the CG-canonical write.

#### Edit 4.3 — concrete edits in compile_dsl_cmd.rs

The full diff is mechanical but spans hundreds of lines. The apply
agent does two text edits:

**Edit 4.3a:** at line 247 (just before the per-kernel emit
accumulators), insert a marker:

```rust
        // ----- BEGIN: legacy per-kernel emit (skipped when
        //              --cg-canonical is set; replaced by the CG
        //              pipeline write below).
        if !args.cg_canonical {
```

**Edit 4.3b:** at line 1479 (after the schedule write, the closing of
the per-kernel emit cluster), insert:

```rust
        }
        // ----- END: legacy per-kernel emit.

        if args.cg_canonical {
            // CG-canonical: run lower → synthesize_schedule → emit_cg_program
            // and write the resulting EmittedArtifacts into
            // crates/engine_gpu_rules/src/. Mirrors `emit_cg_side_channel`
            // but skips its production-path-rejection guard (this IS the
            // production path).
            use dsl_compiler::cg::emit::emit_cg_program;
            use dsl_compiler::cg::lower::lower_compilation_to_cg;
            use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};

            let prog = match lower_compilation_to_cg(&combined) {
                Ok(p) => p,
                Err(outcome) => {
                    eprintln!(
                        "compile-dsl: --cg-canonical: lowering produced {} diagnostic(s):",
                        outcome.diagnostics.len()
                    );
                    for diag in &outcome.diagnostics {
                        eprintln!("  - {diag}");
                    }
                    outcome.program
                }
            };
            let synthesis = synthesize_schedule(&prog, ScheduleStrategy::Default);
            if !synthesis.fusion_diagnostics.is_empty()
                || !synthesis.schedule_diagnostics.is_empty()
            {
                eprintln!(
                    "compile-dsl: --cg-canonical: schedule synthesis produced \
                     {} fusion + {} schedule diagnostic(s)",
                    synthesis.fusion_diagnostics.len(),
                    synthesis.schedule_diagnostics.len(),
                );
                for d in &synthesis.fusion_diagnostics {
                    eprintln!("  - fusion: {}", d.message);
                }
                for d in &synthesis.schedule_diagnostics {
                    eprintln!("  - schedule: {}", d.message);
                }
            }

            let artifacts = match emit_cg_program(&synthesis.schedule, &prog) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("compile-dsl: --cg-canonical: emit_cg_program failed: {e}");
                    return ExitCode::FAILURE;
                }
            };

            let target = PathBuf::from("crates/engine_gpu_rules/src");
            if let Err(e) = fs::create_dir_all(&target) {
                eprintln!(
                    "compile-dsl: --cg-canonical: mkdir {}: {e}",
                    target.display()
                );
                return ExitCode::FAILURE;
            }
            if let Err(e) = write_cg_artifacts(&target, &artifacts) {
                eprintln!("compile-dsl: --cg-canonical: {e}");
                return ExitCode::FAILURE;
            }
            println!(
                "compile-dsl: --cg-canonical: wrote {} wgsl + {} rust file(s) to {} \
                 ({} kernel(s) in index)",
                artifacts.wgsl_files.len(),
                artifacts.rust_files.len(),
                target.display(),
                artifacts.kernel_index.len(),
            );
        }
```

The `write_cg_artifacts` helper at line 2575 is reused as-is.

#### Edit 4.4 — `--check` mode handling

**Decision: `--check` ignores `--cg-canonical`.** When the user runs
`cargo run --bin xtask -- compile-dsl --check`, that's the CI gate
verifying committed legacy output matches a re-run of the legacy
emitter. The CG pipeline doesn't have a `--check` mode today; adding
one is out of scope for 5.7.

**Action:** in `run_compile_dsl`, immediately after `args` is bound and
before any work happens, add:

```rust
    if args.check && args.cg_canonical {
        eprintln!(
            "compile-dsl: --check and --cg-canonical are mutually \
             exclusive (the CG pipeline does not have a --check mode \
             yet). Drop one of the flags."
        );
        return ExitCode::FAILURE;
    }
```

Insert near the top of `run_compile_dsl` (after `args` is bound, before
`discover_sim_files` at line 17 — or as the first statement after the
`pub fn` signature).

### Tests for Patch 4

In `crates/xtask/src/compile_dsl_cmd.rs::cg_side_channel_tests` (or a
new sibling module), add:

```rust
#[test]
fn cg_canonical_flag_is_visible_in_help() {
    // Smoke test: parse the args and verify cg_canonical is a real
    // field. (Doesn't exercise the write path — that needs a full
    // cargo run round-trip.)
    use clap::Parser;
    let args = crate::cli::CompileDslArgs::parse_from(["compile-dsl", "--cg-canonical"]);
    assert!(args.cg_canonical);
}

#[test]
fn cg_canonical_and_check_are_mutually_exclusive() {
    use clap::Parser;
    use std::process::ExitCode;
    let args = crate::cli::CompileDslArgs::parse_from([
        "compile-dsl",
        "--cg-canonical",
        "--check",
    ]);
    let exit = run_compile_dsl(args);
    assert_eq!(format!("{exit:?}"), format!("{:?}", ExitCode::FAILURE));
}
```

The mutual-exclusion test needs `run_compile_dsl` to be reachable
from the test scope. If it's currently `pub(crate)`, that works; if
private, the apply agent either widens its visibility or skips this
test (the early-exit path is small enough that integration-level
coverage from running `cargo run --bin xtask -- compile-dsl
--cg-canonical --check` is sufficient).

---

## Patch 5: gpu_pipeline_smoke gate

### Goal

After Patches 1–4 are applied, run `gpu_pipeline_smoke` against the
CG-canonical output and verify it passes. If it doesn't, document the
failures for follow-up rather than rolling back — the switchover is
reversible per Patch 4's contract.

### Apply-agent procedure

Run sequentially:

```bash
cd /home/ricky/Projects/game

# 1. Build with the legacy output committed (sanity).
cargo build -p engine_gpu --features gpu

# 2. Re-emit via the CG-canonical path. This OVERWRITES every file
#    under crates/engine_gpu_rules/src/ that the CG pipeline produces.
cargo run --bin xtask -- compile-dsl --cg-canonical

# 3. Diff the changes (so the apply agent sees what changed).
git diff --stat crates/engine_gpu_rules/src/

# 4. Run the smoke test.
cargo test -p engine_gpu --features gpu --test gpu_pipeline_smoke
```

### Expected output (success)

```
running 1 test
test step_batch_instantiates_every_kernel ... ok
```

### Expected output (skipped — no GPU adapter)

The test starts with:
```rust
let Ok(mut gpu) = engine_gpu::GpuBackend::new() else {
    eprintln!("skipping: no gpu adapter");
    return;
};
```

If the apply agent's machine has no GPU adapter, the test prints
"skipping: no gpu adapter" and exits 0. Treat this as **inconclusive**
— the patch can't be verified locally and the apply agent should
record that in the post-apply checklist for the next agent / human.

### What to do on failure

`gpu_pipeline_smoke` failures fall into three buckets:

**Bucket A — pipeline construction panics (BGL/WGSL/bind() mismatch).**
A specific kernel's `new()` panics during pipeline construction. The
panic message identifies the kernel. The CG-emitted file
(`crates/engine_gpu_rules/src/<kernel>.rs` or `.wgsl`) has drifted
from what `engine_gpu` expects. Document the kernel and panic message
in a follow-up note and proceed (don't roll back).

**Bucket B — `step_batch` runtime validation panic.** The dispatch
loop hit a wgpu validation error. Less common, more diagnostic depth
needed. Document the error and the kernel that was dispatching
when it failed.

**Bucket C — assertion failure (`state.tick != 1`).** Means the
schedule ran but state didn't advance. Document.

For all three: **DO NOT roll back automatically.** The user explicitly
sized 5.7's end-state to "what works works, what doesn't gets
documented for follow-up". Revert is a one-liner if needed:

```bash
# To revert the CG-canonical output to legacy:
cargo run --bin xtask -- compile-dsl
```

(without `--cg-canonical`, which re-runs the legacy emit path).

### Reporting on failure

If `gpu_pipeline_smoke` fails after the CG switchover, the apply agent
appends to the report:

```
gpu_pipeline_smoke status: FAIL
Failure bucket: A | B | C
Kernel implicated (if A or B): <name>
Panic message / assertion: <verbatim>
Diff stat: <output of git diff --stat>
Action taken: documented for follow-up; legacy emit reverted via
              `cargo run --bin xtask -- compile-dsl` | NOT reverted.
```

### Reporting on pass

```
gpu_pipeline_smoke status: PASS
Output: 1 test, 0 failures, 0 ignored
```

### Reporting on skip

```
gpu_pipeline_smoke status: SKIPPED (no gpu adapter on apply-agent host)
Action: switchover applied but unverified locally. Next agent / human
        with GPU access must re-run `cargo test -p engine_gpu
        --features gpu --test gpu_pipeline_smoke` to confirm.
```

---

## Apply order

Strict sequencing (the order is non-negotiable):

1. **Patch 1** (resident aliasing) — independent.
2. **Patch 2** (scoring view ordering) — independent of Patch 1
   structurally, but Patch 1's `resident_primary_field_for_view` helper
   is referenced by Patch 2's edit at §2.2. **Apply Patch 1 first**.
3. **Patch 3** (schedule wrappers) — independent of Patches 1+2.
4. **Patch 4** (xtask --cg-canonical) — consumes the post-Patch-1+2+3
   `cross_cutting.rs` (notably the `KernelKind`→`KernelId` rename). Apply
   AFTER Patches 1–3.
5. **Patch 5** (gate) — runs against the post-Patch-1–4 tree.

**Recommended order in practice:** 1 → 2 → 3 → 4 → 5.

**Commits.** Each patch SHOULD be a separate commit:

- `feat(dsl_compiler): CG resident-context view alias table (Task 5.7 P1)`
- `fix(dsl_compiler): CG scoring_view_buffers_slice respects binding order (Task 5.7 P2)`
- `feat(dsl_compiler): CG SCHEDULE emits FixedPoint/Indirect wrappers (Task 5.7 P3)`
- `feat(xtask): --cg-canonical switchover flag (Task 5.7 P4)`
- (Patch 5 is a verification step; if it produces a code change, that
  goes into a follow-up; if it just runs the gate, it's noted in the
  P4 commit's post-commit message or a separate "chore" commit.)

The user's CLAUDE.md notes "create NEW commits rather than amending" —
follow that.

---

## Apply notes

### Cross-patch interactions

**Patch 1 ↔ Patch 2.** Patch 2's `resident_primary_field_for_view` call
in §2.2 only resolves correctly when Patch 1 is applied first. If they
land in opposite order, Patch 2 has a forward-reference compile error.

**Patch 1+2 ↔ Patch 3.** The `synthesize_schedule` body Patch 3 modifies
shares no symbols with Patches 1+2 — applying in any order works once
all three are landed.

**Patches 1–3 ↔ Patch 4.** Patch 4's `KernelKind` → `KernelId` rename
crosses both `cross_cutting.rs::synthesize_schedule` (touched by Patch 3)
and `program.rs::synthesize_lib_rs`. If Patch 3 lands FIRST, its post-
edit content includes additional `KernelKind::*` references in the new
`FixedPoint` / `Indirect` arms — Patch 4's rename catches those by
search-and-replace on the post-Patch-3 file.

If Patch 3 hasn't landed yet, Patch 4 still works mechanically (it just
has fewer call sites to rename). The recommended order (1→2→3→4) makes
the rename comprehensive.

### Test interactions

The pre-existing tests in `cross_cutting.rs::mod tests`:

- `resident_context_emits_one_view_storage_field_per_materialised_view`
  uses `view_id=3, name="threat_level"` (non-aliased). Stays green
  through Patches 1+2.
- `resident_context_emits_fold_view_handles_accessor_per_view` same.
  Stays green.
- `resident_context_emits_fixed_fields_with_no_views` uses an empty
  program. Stays green (no view alias path triggers).
- `schedule_synthesises_dispatch_op_enum_and_const_from_stages` uses a
  ViewFold op (not PhysicsRule, not Indirect topology). Stays green
  through Patch 3; Patch 4's `KernelKind` → `KernelId` rename requires
  updating this test's assertion.
- `schedule_with_empty_stages_emits_empty_const` empty program. Stays
  green.

### Rollback strategy if Patch 5 fails irrecoverably

(Defined in Patch 5 §"What to do on failure". Single-line revert:
`cargo run --bin xtask -- compile-dsl` regenerates legacy artifacts.
The xtask `--cg-canonical` flag stays in code unused; no commit needs
revert.)

### What NOT to do

1. **Do not fix kernel-level WGSL or Rust emit bugs in 5.7.** If
   `gpu_pipeline_smoke` panics with a per-kernel BGL mismatch, the
   right outcome is a documented follow-up, not a 5.7-scope expansion.
2. **Do not touch `engine_gpu/src/lib.rs:644-918` (the dispatch loop)**.
   The `KernelKind` → `KernelId` decision (Patch 4 §"Symbol contract")
   keeps the consumer untouched.
3. **Do not touch `engine_gpu/src/physics.rs:89-90`** (direct field
   access on `standing_primary` / `memory_primary`). Patch 1's alias
   table preserves those field names.
4. **Do not delete `crates/dsl_compiler/src/emit_resident_context.rs`
   or `emit_schedule.rs`.** Those are used by the legacy emit branch
   which Patch 4 keeps live (just gated off with `if !args.cg_canonical
   { ... }`). Removing them would break the no-flag path.

---

## Post-apply checklist

- [ ] Patch 1 commit landed with title
      `feat(dsl_compiler): CG resident-context view alias table (Task 5.7 P1)`.
      4 unit tests added; all 4 pass.
- [ ] Patch 2 commit landed with title
      `fix(dsl_compiler): CG scoring_view_buffers_slice respects binding order (Task 5.7 P2)`.
      2 unit tests added; both pass.
- [ ] Patch 3 commit landed with title
      `feat(dsl_compiler): CG SCHEDULE emits FixedPoint/Indirect wrappers (Task 5.7 P3)`.
      Unit tests for Indirect topology + (optionally) FixedPoint pass.
- [ ] Patch 4 commit landed with title
      `feat(xtask): --cg-canonical switchover flag (Task 5.7 P4)`.
      `KernelKind` → `KernelId` rename complete across CG emit
      (`cross_cutting.rs` + `program.rs::synthesize_lib_rs` +
      docstrings + tests). Pre-existing assertions on `KernelKind`
      strings updated to `KernelId`. New `cg_canonical_flag_is_visible_in_help`
      test added; mutual-exclusion smoke test added.
- [ ] `cargo test -p dsl_compiler` — all green.
- [ ] `cargo test -p xtask` — all green.
- [ ] `cargo build` — clean.
- [ ] `cargo run --bin xtask -- compile-dsl --cg-canonical` — runs
      cleanly, prints a "wrote N wgsl + M rust file(s)" success line.
- [ ] `git diff --stat crates/engine_gpu_rules/src/` — captured in the
      report (shows the canonical-output deltas vs legacy).
- [ ] **Patch 5 status** — one of: PASS / FAIL (with bucket + diagnostic) /
      SKIPPED (no GPU). Recorded in the report.

### Deferred follow-ups (NOT 5.7 scope)

- IR-level alias hint plumbing (Patch 1 alternative (a)).
- engine_gpu refactor to expect distinct view-storage buffers per view
  (Patch 1 alternative (c)).
- CG `--check` mode (Patch 4 §4.4 deferred).
- `parity_with_cpu` greening — needs AST coverage for
  Match/Fold/Quantifier/PerUnit/For (still deferred per the reframed
  Phase 5 plan).
- Per-rule `@cascade(max_iter=N)` annotation surfacing through the
  CG IR (Patch 3 hardcodes 8).
- Runtime per-tick scratch allocator (replaces `transient_placeholders`).
