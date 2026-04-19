# World Sim DSL Compiler Specification

Compiler contract. Companion to `dsl/spec.md` (language reference) and `engine/spec.md` (runtime contract). This doc specifies HOW DSL source text lowers to engine calls, not WHAT the language means.

Extracted from `dsl/spec.md` on 2026-04-19. Language grammar, type system, and runtime semantics remain in `dsl/spec.md`; pools/determinism/tick-pipeline contract will migrate to `engine/spec.md`.

---

## 1. Compilation targets

### 1.1 Native Rust (CPU)

- SoA buffers per entity kind, with `@hot` / `@cold` field partitioning (§1.3).
- Rayon-parallel iteration on `[N, OBS_DIM]` observations, masks, and action application.
- Per-agent kernels as `fn` with `#[inline(never)]` in profiling builds for flamegraph attribution (`stories.md` §40).
- `SimScratch` pools carry all per-tick scratch — zero steady-state allocation. Agent slot pool sized at init; ring buffers fixed-cap; event buffers use `SmallVec<[T; N]>` with CI-enforced worst-case bounds (`stories.md` §30).
- **Spatial index is 2D-grid + per-column sorted z-list + movement-mode sidecar** (§9 #25). Primary structure keys `(cx, cy) → SortedVec<(z, AgentId)>` with 16m cells matching voxel-chunk edges. Planar queries walk 9 columns (3×3) and take all. Volumetric queries walk 9 columns and binary-search the z-range. Agents with `movement_mode != Walk` (Fly / Swim / Climb / Fall) live in a separate `in_transit: Vec<AgentId>` that every spatial query scans linearly (expected |in_transit| ≪ N). Slope-walkers stay in the column index — the structure exploits floor-clustering, not flat-ground assumptions.
- RNG: a single `rng_state: u64` per world, consumed in a fixed order (`stories.md` §29). Per-agent RNG streams seeded from `hash(world_seed, agent_id, tick, purpose)` for parallel sampling.

### 1.2 GPU (`voxel_engine::compute::GpuHarness`)

Target is voxel-engine's Vulkan/ash + gpu-allocator stack via `GpuHarness`, not wgpu and not raw CUDA. (`stories.md` §28.) Shader codegen emits SPIR-V via `shaderc` (already in voxel-engine's `build-dependencies`), loaded through `GpuHarness::load_kernel`. Precedents: `terrain_compute.rs` (1024-slot LRU chunk pool), `ai/spatial.rs` (spatial indexing).

The DSL compiler emits a `PolicyRuntime`:

```rust
pub struct PolicyRuntime {
    harness:       voxel_engine::compute::GpuHarness,
    obs_field:     FieldHandle,      // [N, OBS_DIM] f32 / f16
    mask_field:    FieldHandle,      // per-head boolean buffers
    logits_field:  FieldHandle,      // [N, NUM_LOGITS] f32
    action_field:  FieldHandle,      // [N] packed action rows
    weights_field: FieldHandle,      // safetensors-style
    event_ring:    FieldHandle,      // GPU-resident event buffer (replayable subset)
}

impl PolicyRuntime {
    pub fn tick(&mut self, world: &WorldState) -> &[Action] {
        upload_event_ring_delta(&mut self.harness, world);
        self.harness.dispatch("pack_observations", &[...], [n_groups, 1, 1])?;
        self.harness.dispatch("eval_mask_micro",    &[...], [n_groups, 1, 1])?;
        cpu_patch_mask_for_cross_entity(&mut self.harness, world);
        self.harness.dispatch("mlp_forward",        &[...], [n_groups, 1, 1])?;
        self.harness.dispatch("sample_with_mask",   &[...], [n_groups, 1, 1])?;
        self.harness.download(&ctx, &self.action_field)
    }
}
```

GPU-amenable kernels:

- Observation packing (structural gather over SoA agent fields). Per-slot `relative_pos: vec3` + `z_separation_log` pack contiguously — no layout change beyond width.
- Mask evaluation for intrinsic scalar predicates, including `distance` / `planar_distance` / `z_separation`.
- Neural forward (hand-emitted fused GEMM + activation shaders specialised per network topology, matching existing Grokking transformer pattern).
- Event-fold materialization for commutative scalar views (sort events by target before reduction to preserve determinism).
- 3D spatial hash (voxel-chunk-keyed) for `query::nearby_agents` — reuses `voxel_engine::ai::spatial` infrastructure.

CPU-only:

- Cascade rules with cross-entity walks (`t in quest.eligible_acceptors`, `at_war(self, f)`).
- LLM backend.
- Chronicle prose rendering.
- Quest-eligibility and auction-eligibility indices.
- Mixed CPU/GPU mask patching: GPU writes initial mask, CPU patches cross-entity bits, GPU sampler reads final mask (one fence per tick).

GPU determinism constraints (`stories.md` §29):

- Reductions feeding policy decisions use integer fixed-point or sorted-key accumulation to avoid float-associativity drift.
- Materialized views sort events by `target_id` before atomic accumulation.
- Reduction shader workgroup size is pinned via specialization constants.
- Policy sampling seeds from `hash(world_seed, agent_id, tick, "sample")` so parallel sampling is deterministic.

### 1.3 Hot/cold storage split

Mandatory at 200K scale (`stories.md` §31). Authors annotate Agent fields with `@hot` or `@cold`; the compiler emits two SoA layouts and a per-tick sync schedule.

```
entity Agent {
  // Hot — resident, packed into observation buffer
  @hot pos:              vec3,
  @hot hp:               f32,
  @hot max_hp:           f32,
  @hot shield_hp:        f32,
  @hot needs:            [f32; 6],
  @hot emotions:         [f32; 6],
  @hot personality:      [f32; 5],
  @hot memberships:      SortedVec<Membership, 8>,

  // Cold — paged, loaded on policy-tick for High fidelity only
  @cold memory_events:   RingBuffer<MemoryEvent, 64>,
  @cold behavior_profile: SortedVec<(TagId, f32), 16>,
  @cold class_definitions: [ClassSlot; 4],
  @cold creditor_ledger: [Creditor; 16],
  @cold mentor_lineage:  [AgentId; 8],
}
```

Fidelity gating: `@fidelity(>= Medium)` on a view or cascade skips evaluation for Background-fidelity agents. Background agents skip policy inference and cold-field access.

Target: hot ≤ 4 KB/agent, 200K × 4 KB = 800 MB; cold paged to SSD with LRU. Cold fields for non-High agents are swapped out.

---

## 2. Schema emission

The schema-hash rule (what the hash covers, when it must bump) lives in `dsl/spec.md` §4. This section specifies the *emission mechanism* — how the compiler stamps those hashes into generated code and how CI enforces the rule.

The compiler emits four sub-hashes and one combined hash:

```
schema.observation_hash = sha256(canonicalize(observation_schema))
schema.action_hash      = sha256(canonicalize(action_vocabulary))
schema.event_hash       = sha256(canonicalize(event_taxonomy))
schema.reward_hash      = sha256(canonicalize(reward_block))
schema.combined_hash    = sha256(observation_hash || action_hash || event_hash || reward_hash)
```

Loading a checkpoint whose `combined_hash` differs from the current DSL is a hard error. (`stories.md` §15, `stories.md` §23, `stories.md` §64.) The error prints a diff of the four sub-hashes, a textual diff of which fields/variants changed, and a git-remediation hint:

```
error: policy checkpoint schema mismatch
  checkpoint: generated/npc_v3.bin (trained 2026-04-10, step 1_400_000)
  checkpoint schema_hash: sha256:a1b2c3...7890
  current DSL schema_hash: sha256:e4f5g6...2345
  diff:
    + appended observation: self.war_exhaustion (offset 1655, size 1, norm identity)
    + appended action variant: macro_kind::InviteToGroup (slot 4)
    + appended event: InvitePosted
  action: retrain from current DSL, or git-checkout the commit whose
          schema_hash matches the checkpoint.
```

CI guard: a commit that modifies observation, action, event, or reward declarations computes the pre- and post-change hashes; non-append changes (remove, reorder, type change, norm change) block merge without an explicit checkpoint bump.

---

## 3. Lowering passes

TBD — populate as compiler is implemented. Anticipated passes (all currently sketched only):

- **Verb desugaring.** `verb` declarations bundle mask + cascade + reward into a named gameplay action. The compiler lowers `verb` decls into the underlying mask predicate, cascade rule, and reward-block entries without extending the categorical action vocabulary.
- **Read → Ask lowering.** `Read(doc)` is language-surface sugar for `Ask(doc, QueryKind::AboutAll)`. The compiler rewrites every `Read(x)` expression and mask clause into the document-target branch of `Ask`; the runtime MicroKind enum carries `Ask` only. This keeps the runtime action vocabulary at 18 micros while preserving the readability of `Read` in source DSL. See `dsl/spec.md` §3.3 and Appendix A *Information as an action class*.
- **View storage-hint selection.** `@materialized(on_event=[...], storage=<hint>)` authors pick `pair_map` / `per_entity_topk(K, keyed_on=<arg>)` / `lazy_cached`. Compiler rejects infeasible combinations (e.g. `pair_map` on `(AgentId, AgentId)` at N=200K). GPU/CPU routing follows from storage hint: intrinsic scalars + per-entity-slot materializations compile to GPU; lazy + unbounded-pair predicates stay CPU.
- **Cascade dispatch codegen.** `physics` rules lower to phase-tagged handlers with compile-time cycle detection, race detection, and schema-drift guards. Target generated code: Rust `match` on event kind, dispatched through an ordered handler table per phase.

---

## Decisions

Compiler-lowering decisions extracted from `dsl/spec.md` §9. Numbers match the original `§9` entries for cross-reference.

11. **LlmBackend distillation pipeline** — **B, part of the DSL runtime**. `backend "llm" { ... }` is a first-class DSL backend; trajectories are an opt-in export for Python training. No ML-algorithm details in DSL.

16. **Mod event-handler conflict resolution** — **C (named lanes)**: handlers declare a lane `on_event(EventKind) in lane(Validation | Effect | Reaction | Audit)`. Lanes run in order; within a lane, handlers run in lexicographic mod-id (not install order). Multiple handlers per lane coexist (additive). Destructive overrides happen via forking the DSL source, not via a replace keyword.

24. **Utility backend retirement milestone** — **A**: Utility backend never retires. Remains a regression-baseline + untrained-world bootstrap path. Maintenance cost is bounded (~1 KLoC); removal optimises a number that doesn't matter.

---

## Non-goals

Compiler-layer non-goals extracted from `dsl/spec.md` §10.

- Build-system integration beyond `shaderc` SPIR-V compilation — the DSL compiler is a cargo xtask, not a full build tool.

---

## References

- `dsl/spec.md` — language reference (grammar, type system, runtime semantics, settled decisions, non-goals).
- `engine/spec.md` — runtime contract (pools, determinism, event ring, mask, policy trait, tick pipeline, views, trajectory, save/load, invariants, probes, telemetry, schema hash, observation packer, debug & trace runtime).
- `dsl/decisions.md` — per-decision rationale log.
- `dsl/stories.md` — per-batch user-story investigations that drove many of the compiler/runtime design choices.
