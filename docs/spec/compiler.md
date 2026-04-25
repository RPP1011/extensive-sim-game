# World Sim DSL Compiler Specification

Compiler contract. Companion to `language.md` (language reference) and `engine/spec.md` (runtime contract). This doc specifies HOW DSL source text lowers to engine calls, not WHAT the language means.

**The compiler has two consumers: the engine and external pytorch training scripts.** The engine ships two backends — `SerialBackend` (reference implementation) and `GpuBackend` — and the compiler emits Rust + SPIR-V that lands in the engine, ultimately running on whichever backend the engine has active. For external consumers the compiler also emits Python dataclasses + a pytorch `Dataset` over the trace format so training scripts see a typed API without reading engine internals. There is no "CPU vs GPU target" split at the compiler level; only **emission modes** that produce different artefacts (scalar Rust code, GPU dispatch code, SPIR-V shader source, Python trace-consumer module).

Extracted from `language.md` on 2026-04-19; reframed 2026-04-19 (same day) after the engine spec rewrite clarified that CPU + GPU are both first-class engine backends. Reframed again 2026-04-19 to pull ML concerns out of the DSL (policy architecture, training, reward shaping, observation packing all live in external pytorch code now).

---

## 1. Emission modes

The compiler emits **four artefact classes per DSL program**, all consumed by the unified engine or by external ML consumers:

| Emission | What gets emitted | Consumed by |
|---|---|---|
| **Scalar Rust** (§1.1) | Rust `fn` bodies + cascade closures for registering with `SerialBackend` | `SerialBackend` (engine) |
| **GPU dispatch code** (§1.2) | Rust code that calls into engine kernel-dispatch API with the right FieldHandles + SPIR-V kernel names | `GpuBackend` (engine) |
| **SPIR-V kernels** (§1.2) | Compiled shader bytecode for DSL-specific mask predicates / cascade handlers / view reductions | `GpuBackend` (loaded via `GpuHarness::load_kernel`) |
| **Python dataclasses + `Dataset`** (§3) | A Python module with one `@dataclass` per `entity`, `event`, and `@traced` view, plus a pytorch `Dataset` that wraps the trace ring-buffer format and yields typed rows | External pytorch training scripts |

For a complete DSL program, all four artefacts are produced from the same DSL source. The engine selects at init which engine backend to use (§25 of engine spec); the compiler doesn't choose. Every DSL mask predicate / cascade handler / view gets both a scalar Rust implementation (registered with `SerialBackend`) AND a SPIR-V kernel (registered with `GpuBackend`) — enabling cross-backend parity tests. The Python emission is strictly a trace-consumer API: it has no runtime role inside the engine.

### 1.1 Scalar Rust emission

- SoA buffers per entity kind, with `@hot` / `@cold` field partitioning (§1.3).
- Per-agent kernels as `fn` with `#[inline(never)]` in profiling builds for flamegraph attribution.
- `SimScratch` pools carry all per-tick scratch — zero steady-state allocation. Agent slot pool sized at init; ring buffers fixed-cap; event buffers use `SmallVec<[T; N]>` with CI-enforced worst-case bounds.
- **Spatial index is 2D-grid + per-column sorted z-list + movement-mode sidecar** (`language.md` §9 #25). Primary structure keys `(cx, cy) → SortedVec<(z, AgentId)>` with 16m cells matching voxel-chunk edges. Planar queries walk 9 columns (3×3) and take all. Volumetric queries walk 9 columns and binary-search the z-range. Agents with `movement_mode != Walk` (Fly / Swim / Climb / Fall) live in a separate `in_transit: Vec<AgentId>` that every spatial query scans linearly (expected |in_transit| ≪ N). Slope-walkers stay in the column index — the structure exploits floor-clustering, not flat-ground assumptions.
- RNG: a single `rng_state: u64` per world, consumed in a fixed order. Per-agent RNG streams seeded from `hash(world_seed, agent_id, tick, purpose)` for parallel sampling.

This is the reference implementation. The `SerialBackend` runs entirely on the host; its output is the ground truth for cross-backend determinism tests.

### 1.2 GPU dispatch + SPIR-V kernel emission

For `GpuBackend`, the compiler emits SPIR-V kernels (via `shaderc` at compile time) AND Rust dispatch code that invokes them through `voxel_engine::compute::GpuHarness`. Target is voxel-engine's Vulkan/ash + gpu-allocator stack, not wgpu and not raw CUDA.  Precedents: `terrain_compute.rs` (1024-slot LRU chunk pool), `ai/spatial.rs` (spatial indexing).

GPU emission covers the deterministic sim's rules layer — mask predicates, cascade handlers, event-folded views, spatial-hash queries. ML forward passes are NOT compiled here; ML is out of DSL scope (see `language.md` §10).

The compiler emits dispatch code that uses engine-shipped kernel handles:

```rust
pub struct RulesRuntime {
    harness:         voxel_engine::compute::GpuHarness,
    mask_field:      FieldHandle,    // [N × NUM_ACTIONS] u8 validity bits
    cascade_events:  FieldHandle,    // GPU-resident event buffer (replayable subset)
    view_outputs:    FieldHandle,    // per-view materialized-output buffers
    spatial_index:   FieldHandle,    // 2D column grid (voxel-chunk keyed)
    aggregate_pool:  FieldHandle,    // cross-entity index tables (standings, eligibility)
}

impl RulesRuntime {
    pub fn tick(&mut self, world: &WorldState) {
        upload_event_ring_delta(&mut self.harness, world);
        self.harness.dispatch("eval_mask_predicates",  &[...], [n_groups, 1, 1])?;
        cpu_patch_mask_for_cross_entity(&mut self.harness, world);
        self.harness.dispatch("dispatch_cascades",     &[...], [n_groups, 1, 1])?;
        self.harness.dispatch("update_materialized_views", &[...], [n_groups, 1, 1])?;
        self.harness.dispatch("rebuild_spatial_index", &[...], [n_groups, 1, 1])?;
    }
}
```

GPU-amenable kernels:

- Mask evaluation for intrinsic scalar predicates, including `distance` / `planar_distance` / `z_separation`.
- Cascade handlers that touch GPU-resident `AggregatePool<T>` — per-event match + emit loops with fixed-size iteration bounds.
- Event-fold materialization for commutative scalar views (sort events by target before reduction to preserve determinism).
- 3D spatial hash (voxel-chunk-keyed) for `query::nearby_agents` — reuses `voxel_engine::ai::spatial` infrastructure.

Always host-side (regardless of engine backend):

- Chronicle prose rendering (pure template expansion; no sim state mutation).
- Metric sink dispatch (§2.11 of language.md).
- Save/load serialization.
- Trace-format emission (consumed by the Python `Dataset`).

**Cascade rules with cross-entity walks** (`t in quest.eligible_acceptors`, `at_war(self, f)`): the **GpuBackend** handles these via GPU-resident `AggregatePool<T>` (with `T: Pod` discipline — see engine spec §16) and kernel-side iteration of fixed-size inline arrays. The **SerialBackend** uses `Box<dyn CascadeHandler>` closures with direct pool access. Per-tick determinism is verified by cross-backend parity. Cross-entity walks are no longer a CPU-only concern in the new framing — the compiler emits SPIR-V for them, targeting the `AggregatePool<T>` layout.

Quest-eligibility and auction-eligibility indices are cross-entity materialized views (engine spec §15) — GPU-dispatched on `GpuBackend` via sorted-key reductions; scalar on `SerialBackend`. Both backends expose the same view query API.

GPU determinism constraints:

- Reductions feeding scoring decisions use integer fixed-point or sorted-key accumulation to avoid float-associativity drift.
- Materialized views sort events by `target_id` before atomic accumulation.
- Reduction shader workgroup size is pinned via specialization constants.
- Utility-backend tiebreak RNG seeds from `hash(world_seed, agent_id, tick, "scoring")` so parallel evaluation is deterministic.

### 1.3 Hot/cold storage split

Mandatory at 200K scale. Authors annotate Agent fields with `@hot` or `@cold`; the compiler emits two SoA layouts and a per-tick sync schedule.

```
entity Agent {
  // Hot — resident, read every tick by masks, scoring, and traced views
  @hot pos:              vec3,
  @hot hp:               f32,
  @hot max_hp:           f32,
  @hot shield_hp:        f32,
  @hot needs:            [f32; 6],
  @hot emotions:         [f32; 6],
  @hot personality:      [f32; 5],
  @hot memberships:      SortedVec<Membership, 8>,

  // Cold — paged, loaded on scoring-tick for High fidelity only
  @cold memory_events:   RingBuffer<MemoryEvent, 64>,
  @cold behavior_profile: SortedVec<(TagId, f32), 16>,
  @cold class_definitions: [ClassSlot; 4],
  @cold creditor_ledger: [Creditor; 16],
  @cold mentor_lineage:  [AgentId; 8],
}
```

Fidelity gating: `@fidelity(>= Medium)` on a view or cascade skips evaluation for Background-fidelity agents. Background agents skip full scoring evaluation and cold-field access.

Target: hot ≤ 4 KB/agent, 200K × 4 KB = 800 MB; cold paged to SSD with LRU. Cold fields for non-High agents are swapped out.

---

## 2. Schema emission

The schema-hash rule (what the hash covers, when it must bump) lives in `language.md` §4. This section specifies the *emission mechanism* — how the compiler stamps those hashes into generated code and how CI enforces the rule.

The compiler emits four sub-hashes and one combined hash:

```
schema.state_hash    = sha256(canonicalize(entity_field_layouts))
schema.event_hash    = sha256(canonicalize(event_taxonomy))
schema.rules_hash    = sha256(canonicalize(physics_cascades + masks + verbs))
schema.scoring_hash  = sha256(canonicalize(scoring_tables))
schema.combined_hash = sha256(state_hash || event_hash || rules_hash || scoring_hash)
```

Loading a trace whose `combined_hash` differs from the current DSL is a hard error.  The error prints a diff of the four sub-hashes, a textual diff of which fields/variants changed, and a git-remediation hint:

```
error: trace format mismatch
  trace: traces/run_2026-04-10.bin
  trace schema_hash: sha256:a1b2c3...7890
  current DSL schema_hash: sha256:e4f5g6...2345
  diff:
    + appended entity field: Agent.war_exhaustion (offset 240, size 4, f32)
    + appended action variant: macro_kind::InviteToGroup (slot 4)
    + appended event: InvitePosted
  action: run the older engine version that matches the trace schema, or
          re-emit traces from the current DSL.
```

CI guard: a commit that modifies entity fields, events, cascades/masks/verbs, or scoring declarations computes the pre- and post-change hashes; non-append changes (remove, reorder, type change) block merge without an explicit schema bump.

---

## 3. Lowering passes

TBD — populate as compiler is implemented. Anticipated passes (all currently sketched only):

- **Verb desugaring.** `verb` declarations bundle mask + cascade + scoring into a named gameplay action. The compiler lowers `verb` decls into the underlying mask predicate, cascade rule, and scoring-table entries without extending the categorical action vocabulary.
- **Read → Ask lowering.** `Read(doc)` is language-surface sugar for `Ask(doc, QueryKind::AboutAll)`. The compiler rewrites every `Read(x)` expression and mask clause into the document-target branch of `Ask`; the runtime MicroKind enum carries `Ask` only. This keeps the runtime action vocabulary at 18 micros while preserving the readability of `Read` in source DSL. See `language.md` §3.3 and Appendix A *Information as an action class*.
- **View storage-hint selection.** `@materialized(on_event=[...], storage=<hint>)` authors pick `pair_map` / `per_entity_topk(K, keyed_on=<arg>)` / `lazy_cached`. Compiler rejects infeasible combinations (e.g. `pair_map` on `(AgentId, AgentId)` at N=200K). GPU/CPU routing follows from storage hint: intrinsic scalars + per-entity-slot materializations compile to GPU; lazy + unbounded-pair predicates stay CPU.
- **Cascade dispatch codegen.** `physics` rules lower to phase-tagged handlers with compile-time cycle detection, race detection, and schema-drift guards. Target generated code: Rust `match` on event kind, dispatched through an ordered handler table per phase.
- **Python emission.** Every `entity`, `event`, and `@traced` view gets a matching `@dataclass` in the emitted Python module. The Dataset class wraps the trace ring-buffer format and yields typed rows (one row per tick-agent pair, with per-event sub-rows streamed alongside). The Python module is checked-in generated output, not runtime-loaded by the engine.

---

## Decisions

Compiler-lowering decisions extracted from `language.md` §9. Numbers match the original `§9` entries for cross-reference.

16. **Mod event-handler conflict resolution** — **C (named lanes)**: handlers declare a lane `on_event(EventKind) in lane(Validation | Effect | Reaction | Audit)`. Lanes run in order; within a lane, handlers run in lexicographic mod-id (not install order). Multiple handlers per lane coexist (additive). Destructive overrides happen via forking the DSL source, not via a replace keyword.

24. **Utility backend is the production NPC backend.** Utility backend is permanent. ML training is external to the DSL; the compiler emits Python dataclasses + a pytorch `Dataset` for trace-format consumption (see §3 Lowering passes). `scoring` declarations drive utility-backend scoring AND are written to traces so external pytorch scripts can reshape them into rewards. Maintenance cost of the utility backend is bounded (~1 KLoC) and it remains the regression baseline + untrained-world bootstrap path.

---

## Non-goals

Compiler-layer non-goals extracted from `language.md` §10.

- Build-system integration beyond `shaderc` SPIR-V compilation — the DSL compiler is a cargo xtask, not a full build tool.

---

## References

- `language.md` — language reference (grammar, type system, runtime semantics, settled decisions, non-goals).
- `engine/spec.md` — runtime contract (pools, determinism, event ring, mask, utility-backend trait, tick pipeline, views, trace ring, save/load, invariants, probes, metric sinks, schema hash, debug & trace runtime).
-  — per-decision rationale log.
-  — per-batch user-story investigations that drove many of the compiler/runtime design choices.
