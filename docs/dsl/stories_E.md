# Category E — Runtime / Sim Engineer Stories (25–33)

Analysis of the runtime-engineering user stories in `user_stories_proposed.md` against
the current DSL design (`proposal_policy_schema.md`, `proposal_universal_mechanics.md`,
`state_world.md`) and the voxel-engine GPU backend at `/home/ricky/Projects/voxel_engine/`.

Each story is scored SUPPORTED / PARTIAL / GAP against the schema as it exists today.
"Supported" means the proposal already describes a concrete mechanism; "partial" means
the mechanism exists but the story surfaces a concrete extension to spec; "gap" means
new design work is required.

---

### Story 25: One batched neural forward per tick
**Verdict:** SUPPORTED

**User's framing note:** "Good"

**How the DSL supports this:**
`proposal_policy_schema.md` §2.4 defines exactly one call point:
```
fn evaluate_batch(observations: &PackedObservationBatch,  // [N, OBS_DIM]
                  masks:        &PackedMaskBatch)
                  -> ActionBatch;                         // [N] of typed Action
```
One forward per tick for every alive agent is the committed shape. §2.1.8 sizes
the batch: `1655 × 4 bytes × 20K agents = 132 MB`. The proposal states this is "on-GPU
is free; CPU pack <5ms" (§2.1.8). §5 splits compilation: observation packing → GPU,
mask evaluation → GPU, neural forward → GPU (already what Burn does).

**Implementation walkthrough:**
1. `iter_alive_agents()` produces a stable agent index layout for the tick. The
   `GroupIndex` in `state_world.md` already clusters entities by
   `(settlement_id, party_id)`, which gives contiguous ranges suitable for
   `[N, OBS_DIM]` packing without per-agent scatter.
2. The DSL's `observation { ... }` block compiles to a packing kernel (§3). Each
   atom/block/slot array maps to an offset within the packed row. Compiler
   emits "packed struct with named field offsets" so debug tools can decode
   bytes back to named fields.
3. For the voxel-engine backend (story 28), packing compiles to a single
   compute shader bound to (a) an agent-ID buffer, (b) the source field buffers
   (hp, treasury, etc.), (c) the output packed observation buffer. One dispatch
   with workgroup-per-agent. `GpuHarness::dispatch` in
   `voxel_engine/src/compute/harness.rs` already supports exactly this shape —
   named kernels + storage-buffer fields + `[groups_x, 1, 1]` dispatch.
4. Mask evaluation is a second dispatch into a per-head boolean tensor
   `[N × NUM_KINDS]`, `[N × NUM_SLOTS]`, etc. (§2.3). Cross-entity
   predicates that are GPU-hostile (§5 — `t ∈ quest.eligible_acceptors`) fall
   back to CPU into the same boolean tensor.
5. Burn (or an equivalent tensor library over the harness-managed buffer) runs
   the forward pass. Sampling with mask is one fused kernel that consumes
   logits + mask and writes typed action rows.

Memory is pinned on GPU for the whole run. `GpuHarness::create_field` returns
`FieldHandle`s that stay allocated until explicit release — no per-tick Vulkan
allocations. The packed observation buffer, the mask buffer, the logits buffer,
and the action buffer are all created once at startup and reused per tick.

**Gaps / open questions:**
- **Batch layout for dead/asleep agents.** The proposal says "all alive NPCs go
  through one forward pass" but fidelity zones (`state_world.md`,
  `FidelityZone.fidelity ∈ {High, Medium, Low, Background}`) imply background
  agents may skip inference. Two options: (a) pack a compacted `[N_active, OBS_DIM]`
  and maintain an active-ID side list, or (b) pack `[N_max, OBS_DIM]` and use an
  active mask; compute cost is N_active either way but (b) avoids a compaction
  pass. Needs specification.
- **Macro-head cadence (open question #12 in schema).** If macro inference runs
  only every 10 ticks, the batch on those ticks is larger (more heads) than on
  pure-micro ticks. Either accept two batch shapes or always emit all heads
  and ignore the macro output 9/10 ticks.
- **Observation feature pipeline.** Some atoms depend on *derived views*
  (`view::mood`, `is_hostile(a, b)`, group-war standings). If those views are
  lazy (story 26 PARTIAL case), the pack kernel stalls on view computation.
  Story 26 is the fix.

**Related stories:** 26 (view materialization feeds the pack kernel), 28 (the
compilation target that *runs* the batched forward), 31 (determines batch size).

---

### Story 26: Materialize hot views eagerly
**Verdict:** SUPPORTED

**User's framing note:** "Essential"

**How the DSL supports this:**
`proposal_policy_schema.md` §5 calls out "Event-fold view materialization —
reductions over event rings (sum damage events for hp, etc.)" as GPU-amenable.
§1.1 commits to event sourcing: "current state is a fold over events + entity
baseline." This is exactly the materialization-vs-lazy-fold axis. The
`state_npc.md` derivation graph already distinguishes "Updated by" columns
(event-driven fields) from "Updater computes" (derived views). `behavior_profile`
(state_npc.md line 276) is the canonical "updated incrementally on each event"
example — not re-folded every tick.

**Implementation walkthrough:**
Proposed DSL annotation:
```
view mood(self) @materialized {
  initial: 0.0
  on event(DamageReceived{target=self}) { self.mood -= 0.1 * e.damage / self.max_hp }
  on event(FriendDied{observer=self})   { self.mood -= 0.3 }
  on event(NeedsSatisfied{agent=self})  { self.mood += 0.05 * e.delta }
  clamp: [-1, 1]
}
```
`@materialized` compiles to:
1. A field on the agent struct (append-only addition, §4 "schema versioning").
2. An event-handler dispatch table — for each matching event type, a CPU or GPU
   update call. Since the event stream is append-only per tick, this is a
   fused reduction: group events by target agent, accumulate the delta, write
   once.
3. The pack kernel (story 25) reads `agent.mood` directly with no fold.

Lazy / non-materialized views compile to inline folds inside the pack kernel.
The DSL distinguishes them by the `@materialized` annotation. Cost trade-off:
eager pays cost per event, lazy pays cost per observation pack. For
frequently-read views (hp, treasury, mood, morale, `is_hostile`, relationship
valence), eager wins decisively because packing runs every tick whereas some
events are rare.

**Event handler codegen shape:**
```
// compiled from the DSL
for evt in world.events_this_tick.iter() {
    match evt {
        Event::DamageReceived { target, damage } => {
            agent_mut(target).mood -= 0.1 * damage / agent(target).max_hp;
        }
        Event::FriendDied { observer, .. } => {
            agent_mut(observer).mood -= 0.3;
        }
        ...
    }
}
```
For GPU targets, the same handler is a compute kernel dispatched over the
event ring where each thread handles one event; it uses atomic fetch-add into
the target agent's materialized field when multiple events target the same
agent on the same tick.

**Gaps / open questions:**
- **Fold order vs replay determinism (story 33).** When handlers use atomics
  across events targeting the same agent, fold order is non-deterministic on
  GPU. Mitigation: sort events by (target_id, source_tick_index) before the
  reduction, or run the reduction on CPU. §5 classifies the event-fold
  materializer as GPU-amenable but only for scalar reductions with commutative
  operators; for non-commutative chains (e.g. "max of an ordered sequence"),
  CPU.
- **Cache invalidation on save/load (story 32).** If materialized views live in
  the serialized snapshot, the invariant is "fold-from-baseline ≡ materialized".
  A CI check should re-derive materialized views from the event log on a
  sampled snapshot and diff against stored values.
- **Which views need `@materialized`?** The proposal doesn't list them. Working
  list from the observation schema: `hp`, `mood`, `morale_pct`, `focus`,
  `gold`, `n_close_friends`, `fame_log`, `group.treasury`, `group.military_strength`,
  `relationship.valence`. All appear in §2.1 and are read every tick per alive
  agent.

**Related stories:** 25 (pack kernel reads from materialized views), 27
(new event types need corresponding event handler bindings), 30 (materialized
views live in fixed-cap agent struct fields, no per-tick alloc), 33 (fold
determinism bounds replay).

---

### Story 27: Add a physics rule without breaking others
**Verdict:** PARTIAL (core composition supported; compile-time validation underspecified)

**User's framing note:** "Essential, should have 'compile' time validation"

**How the DSL supports this:**
The DSL is event-sourced (§1.1). Rules are of the form
`on event(<Pattern>) { <cascade or mutation> }`. New rules don't edit existing
ones — they register additional handlers. The composition semantics are
"every handler matching the event fires."

`proposal_policy_schema.md` §5 already separates "cascade rules (CPU)" from
"GPU packers" and lists cascade rules as stay-CPU, keeping the dispatch surface
flexible for new rules.

**Implementation walkthrough:**

*Rule registration model.*
```
rule collapse_chain {
  on event(VoxelDamaged{pos=p, new_integrity=i}) where i <= 0.0 {
    emit StructuralEvent::FragmentCollapse { pos: p, cause: NpcHarvest }
    for q in voxel_neighbors_above(p) {
      damage_voxel(q, COLLAPSE_DAMAGE)
    }
  }
}

rule collapse_chronicle {  // independent, added later
  on event(StructuralEvent::FragmentCollapse{pos=p, cause}) {
    emit ChronicleEntry {
      category: ChronicleCategory::Crisis,
      text: format!("A section of ground collapsed at {:?}", p),
      entity_ids: entities_near(p, 20.0),
    }
  }
}
```
Both rules fire on the same underlying cascade chain. Adding
`collapse_chronicle` does not touch `collapse_chain`. The compiler collects all
rules with matching event patterns into a dispatch table.

*Ordering semantics.* The proposal needs to commit to one of:
  (a) **Unordered** — handlers must be commutative; compiler rejects rules
      that mutate fields other rules also write. Simple but restrictive.
  (b) **Priority-tagged** — `rule foo @priority(50)`; default 100. Compiler
      emits a topological warning if cycles would exist after priority resolution.
  (c) **Phase-based** — `on_pre_event`, `on_event`, `on_post_event` phases,
      strict within-phase commutativity.
  Recommendation: (c). Matches the existing determinism contract (from `CLAUDE.md`
  combat-sim: "unit processing order is shuffled per tick to prevent first-mover
  bias", but phase ordering is deterministic).

*Compile-time validation — what CAN be statically caught.* This is the user's
explicit extension:

1. **Event-type existence.** `on event(UnknownFoo{})` fails at compile time.
2. **Pattern-field mismatch.** `on event(EntityDied{weapon=w})` when the
   event struct has no `weapon` field.
3. **Field-write races in the same phase.** Two unordered rules both writing
   `agent.hp` → compiler error unless one of them uses an explicit
   commutative operator (`+=`, `min=`, `max=`).
4. **Cycle detection in cascades.** Rule A emits event type X, rule B handles
   X and emits Y, rule C handles Y and emits X, with no quiescence predicate
   → compiler warning. Detectable as a cycle in the
   event-type → emitter → event-type graph.
5. **Invariant violations (partial).** Rules annotated with pre/post-invariants
   (`@requires agent.hp >= 0`, `@ensures agent.hp >= 0`) can be checked with
   abstract interpretation over the mutation body. Full invariant checking is
   a research problem; first pass: syntactic checks (any write to a clamped
   field must go through the clamp helper).
6. **Mask-rule-observation coherence.** The proposal's mask predicates
   reference views; if a rule deletes a view's source event type, the mask
   still compiles but silently always-false. Compiler should warn.
7. **Schema-hash drift (§4).** Any change to event struct shape bumps the
   schema hash; CI blocks unless a migration is declared.

*What CANNOT be statically caught.*
- Semantic consistency (two rules that both decrement `hp` "correctly" but
  together overshoot). Requires runtime invariant checks.
- Rule interaction with RNG (`rng_state` is global serialized; two rules
  consuming randomness in different orders on different platforms are the
  determinism bug story 29 is about).
- Fixed-point convergence of cascade rules — a rule that re-emits on its
  own output may loop fewer/more times depending on agent count.

**Gaps / open questions:**
- Phase ordering not yet written into the proposal. Needs adding to §5 or a
  new §9 "rule composition".
- Compile-time checks 3, 5, 6 require a type-flow analysis pass in the
  compiler. Not specified.
- `emit` vs `apply` distinction — does the DSL allow direct mutation inside
  rules, or only event emission? Event-only is cleaner for replay (story 33)
  and for static analysis (all writes reduced to the event-dispatch handler),
  but is more verbose. The current proposal is ambiguous; §2.5 reward shows
  direct mutation (`delta(self.hp_frac)` reads), but §5 cascade rules say
  "event-fold". Pick one: **event-only**, with `apply` reserved for the
  compiler-generated fold itself.

**Related stories:** 25 (new rules add new mask predicates → observation packer
must be regenerated), 29 (rule ordering is a determinism knob), 33
(event-type additions require replay-log schema migration), 38 (new-rule
"dry-run" = self-contained tick reproduction).

---

### Story 28: Compile to either Rust or CUDA (use voxel-engine)
**Verdict:** PARTIAL (scoping is clear; voxel-engine integration shape needs specification)

**User's framing note:** "Essential, use voxel-engine though, that is what that
project is for"

**How the DSL supports this:**
`proposal_policy_schema.md` §5 explicitly scopes GPU compilation — not "compile
the whole DSL to CUDA". Concrete GPU kernels listed:
- Observation packing
- Per-head mask evaluation
- Neural forward pass
- Event-fold view materialization

GPU-hostile (stay CPU):
- Cross-entity mask predicates
- LLM backend
- Chronicle text generation
- Cascade rules with heterogeneous events
- World event emission

Hybrid:
- Spatial queries (nearby_actors, threats)

**Implementation walkthrough via voxel-engine:**

voxel-engine at `/home/ricky/Projects/voxel_engine/` already provides the
relevant Vulkan/ash GPU infrastructure:

- `src/compute/harness.rs` — `GpuHarness` with `create_field`,
  `load_kernel` (SPIR-V bytes + binding count), `dispatch`, `upload`,
  `download`. Host-visible, host-coherent memory with persistent mappings.
  Exactly the "taichi-style CPU↔GPU harness" shape the DSL compiler needs.
- `src/terrain_compute.rs` — reference for a real pipeline (materialization
  with LRU slot pool, descriptor sets, fences). Useful precedent for the
  DSL's chunked-field management model.
- `src/vulkan/instance.rs` + `src/vulkan/allocator.rs` — already-working
  Vulkan context + gpu-allocator.
- `shaders/` directory — 30+ precompiled compute shaders for
  physics/fluid/SDF work. Precedent for the shader build pipeline
  (`build.rs` + `shaderc`).

Integration shape the DSL compiler should produce:

```
// CPU-side Rust stub emitted by the DSL compiler
pub struct PolicyRuntime {
    harness: voxel_engine::compute::GpuHarness,
    obs_field: FieldHandle,     // [N, OBS_DIM] f32
    mask_field: FieldHandle,    // per-head boolean buffers
    logits_field: FieldHandle,  // [N, NUM_LOGITS] f32
    action_field: FieldHandle,  // [N] packed action rows
    weights_field: FieldHandle, // model weights
    ...
}

impl PolicyRuntime {
    pub fn tick(&mut self, world: &WorldState) -> &[Action] {
        // 1. Observation pack kernel
        upload_event_ring_delta(&mut self.harness, world);
        self.harness.dispatch("pack_observations", &[...], [N_groups, 1, 1])?;

        // 2. Mask evaluation (GPU-amenable predicates)
        self.harness.dispatch("eval_mask_micro", &[...], [N_groups, 1, 1])?;
        // CPU patch for cross-entity predicates
        cpu_patch_mask_for_quest_eligibility(&mut self.harness, world);

        // 3. Neural forward (Burn over the field buffers, or hand-written
        //    shader if Burn interop is a problem)
        self.harness.dispatch("mlp_forward", &[...], ...)?;

        // 4. Sample + writeback
        self.harness.dispatch("sample_with_mask", &[...], ...)?;
        self.harness.download(&ctx, &self.action_field)
    }
}
```

**Which DSL constructs map cleanly to voxel-engine kernels:**

| DSL construct | Maps cleanly? | Kernel |
|---|---|---|
| `observation { self.hp_pct = self.hp / self.max_hp }` | Yes | one line in pack shader |
| `slots nearby_actors[K=12] from query::nearby_agents(self, radius=50) sort_by distance` | Hybrid | Spatial query on CPU (or via `src/ai/spatial.rs`), gather on GPU |
| `block self.psychological { ... }` | Yes | struct-of-arrays gather in pack shader |
| `summary recent_chronicle[...] { group_by e.category output count_log }` | Partial | If the event ring is GPU-resident, yes; currently it contains `String` (`state_world.md` "chronicle, `Vec<ChronicleEntry>` with String") so it's CPU-only |
| `mask { Attack(t) when t.alive ∧ is_hostile(self, t) ∧ distance(t) < AGGRO }` | Yes for intrinsic predicates | `is_hostile` decomposes to relationship_valence < T ∨ groups_at_war bitset ∨ predator_prey table — all GPU-friendly scalars |
| `mask { AcceptQuest(qid) when self ∈ quest.eligible_acceptors }` | No | Stay CPU; write into shared mask buffer |
| `view mood @materialized { on event(X) ... }` | Yes | compute shader dispatched per event |
| `rule collapse_chain { on event(VoxelDamaged) ... }` | No | Stay CPU (heterogeneous events + emit chain) |
| `reward { +1.0 on event(EntityDied{killer=self ...}) }` | Yes | reduction over filtered event stream |

**Which constructs DON'T map cleanly:**
- **String-bearing events** (`ChronicleEntry.text`, `WorldEvent::Generic{text}`).
  These must stay CPU or be represented by event-ID + parameter-bag. Per story
  60 ("text gen should not be load bearing. Use numeric IDs for everything
  important"), chronicles carry a `category: u8 + params: [u32; 4]` tuple and
  text is rendered post-hoc on the CPU for display only.
- **Quest/auction eligibility** — quest membership is a `Vec<AgentId>` that
  grows dynamically; GPU-resident only if we commit to a fixed-cap
  per-quest member table. Story 30 pushes us toward this anyway.
- **Room-growth automaton** (construction.rs flood fill) — not a per-tick
  inference kernel; keep as-is, CPU.

**voxel-engine-specific integration notes:**
- voxel-engine is Vulkan/ash + gpu-allocator. **Not wgpu and not CUDA.**
  `proposal_policy_schema.md` §5 mentions "wgpu/CUDA"; this should be updated
  to **ash/Vulkan via voxel-engine's `GpuHarness`**. The DSL compiler emits
  SPIR-V (via `shaderc`, which is already in `voxel_engine/Cargo.toml`
  `build-dependencies`) and loads kernels via `harness.load_kernel`.
- Burn-on-Vulkan is technically supported (Burn's `wgpu` backend could bind
  to voxel-engine's Vulkan context via external memory, but that's a research
  integration). Simpler: hand-emit GEMM + activation shaders for the policy
  network, since observations and action-head shapes are declared in the DSL
  and the compiler can generate a fused forward shader specific to the
  network topology. This is the same technique Grokking transformer uses
  (our existing transformer is small: d=32, 4 layers — easily fits one
  compute shader).
- The CPU target ("native Rust, rayon-parallel") is the unchanged path —
  `evaluate_batch` with `par_iter` over the `[N, OBS_DIM]` rows. Used for
  debug builds and small-scale tests.

**Gaps / open questions:**
- **Weight upload cadence.** Neural weights change on each training
  checkpoint swap, not every tick. A separate `update_weights(checkpoint)` API
  on `PolicyRuntime` that calls `harness.upload` once, and a hot-swap fence
  to avoid tearing. Not currently specified.
- **Mixed CPU/GPU mask.** §5 says cross-entity predicates run on CPU and "that
  boolean tensor is what GPU consumes." Concretely, the flow is: GPU mask
  kernel writes initial mask → CPU patches the bits that require cross-entity
  walks → GPU sampler reads final mask. Needs two fences or a single
  read-back/write-back round trip per tick. Cost bound unclear.
- **Readback cost.** `harness.download` of `[N] × action_row` per tick. At N=200K
  and action_row = ~32 bytes, that's 6.4 MB/tick over PCIe. PCIe 4.0 x16
  handles this in ~70 µs; PCIe 3.0 ~140 µs. Fine, but worth tracking.
- **Shader codegen.** The compiler needs a SPIR-V emitter. Options:
  (a) emit GLSL and call `shaderc` at compile time; (b) emit SPIR-V directly
  via the `rspirv` crate; (c) emit Rust code that calls `shaderc` at runtime
  for specialized shaders. (a) is the simplest — matches voxel-engine's
  existing shader build pipeline.

**Related stories:** 25 (this is the deployment target), 29 (GPU determinism
constraints — sort event reductions, no float non-associativity for critical
reductions), 30 (GPU field buffers stay allocated for the run = zero-malloc),
31 (GPU is the only way to reach 200K).

---

### Story 29: Deterministic sim given seed
**Verdict:** SUPPORTED

**User's framing note:** "Essential"

**How the DSL supports this:**
Determinism is an existing first-class contract. From `state_world.md`:
```
rng_state | u64 | PCG-style RNG state; sole randomness source
next_rand_u32 / next_rand | all stochastic systems
```
and from project `CLAUDE.md`:
> All simulation randomness flows through `SimState.rng_state` via
> `next_rand_u32()`. Never use `thread_rng()` or any external RNG in simulation
> code. Unit processing order is shuffled per tick to prevent first-mover bias.
> Tests in `src/ai/core/tests/determinism.rs` verify reproducibility. CI runs
> determinism tests in both debug and release modes.

The DSL inherits this contract. Every `next_rand` call in generated code
advances the single `rng_state`.

**Implementation walkthrough:**

*RNG semantics.* Per-world, single state. Draws serialized by call order.
Inside a tick, the order is:
  1. Pre-tick events (fidelity zone membership rebuild — no RNG)
  2. Event handler dispatch (materialized-view updates — no RNG)
  3. Policy inference (deterministic given weights + obs + mask; sampling is
     temperature-softmax with RNG draws in a fixed agent order)
  4. Action application / cascade rules (RNG for dice rolls, damage variance)
  5. Construction / terrain updates (RNG for procedural gen if triggered)

*Agent ordering.* The existing rule ("unit processing order shuffled per tick
to prevent first-mover bias") translates to: the policy runs on all agents in
parallel (no order dependency because observations are packed from a frozen
pre-tick snapshot), but action APPLICATION is a serialized fold over agents
in a shuffled-but-seeded order. The shuffle itself consumes RNG draws from
`rng_state`, making the order reproducible.

*Per-agent RNG vs world RNG.* Proposal: stick with world RNG for now. Per-agent
RNG streams (seeded from world RNG at agent spawn) are a future optimization
for parallel policy sampling but they complicate save/load (story 32) because
each agent needs to carry its stream state. Deferred.

*Event ordering.* Events emitted within a single phase are collected into a
per-tick buffer (`world_events`, `structural_events`). The order of emission
follows agent processing order (shuffled with RNG). Handlers process the buffer
in append order. This gives deterministic cascade order.

*GPU determinism traps.* The GPU path (story 28) introduces non-determinism
risks:
- **Float associativity.** Sum reductions over large arrays on GPU use
  tree reductions with non-fixed ordering across warps → bit-different results.
  Mitigation: for reductions that feed into policy decisions, use integer
  fixed-point accumulation (scale floats to i32, sum, rescale), or sort by
  key before reduction.
- **Atomic fetch-add ordering.** If two events on the same tick both do
  `agent.mood -= X` via atomics, the final value is commutative-correct but
  the intermediate may not be observable. For materialized views this is fine
  because only the final value is read. For views that accumulate into a
  history ring, sort first.
- **Reduction shader warp counts.** Vendor-specific. Pin the reduction shader
  to a fixed workgroup size (specialization constants).

*What breaks determinism — and how the proposal handles it.*
- **Text generation** (per story 60): `ChronicleEntry.text` is allowed to
  diverge across replays. Story 60 marks text-gen events as non-replayable.
  The DSL uses numeric event IDs + parameter bags (agent IDs, quantities) as
  the replay-safe representation; text is a CPU-side render of those
  parameters and may use non-deterministic font/language models without
  breaking sim replay. Per user's framing: "text gen should not be load
  bearing. Use numeric IDs for everything important."
- **Wall-clock dependence.** Compiler forbids reading `SystemTime` or
  `Instant` in DSL code; only the tick counter is available.
- **HashMap iteration order.** `ahash::RandomState` is already seeded per
  process. If serializable, can seed deterministically from `rng_state`.
  The proposal should commit: hash_seed = (world_seed ^ tick) at state init,
  never re-seeded.
- **LLM backend** (§2.4) is explicitly non-deterministic and off the per-tick
  path.

**Gaps / open questions:**
- The GPU-determinism details above aren't in the current proposal. Needs a
  §6 "determinism under GPU compilation" addition.
- HashMap seed policy not specified.
- Policy-sampling determinism: temperature softmax + categorical sample needs
  a specified RNG stream. Current combat transformer uses a derived stream
  (hash of `(world_rng, agent_id, tick)`) per agent to parallelize safely.
  The DSL should document this pattern.

**Related stories:** 27 (rule ordering is a determinism knob), 28 (GPU
compilation introduces new determinism hazards), 32 (snapshot includes
`rng_state`), 33 (replay from seed + event log + initial state), 60
(text-gen bound on determinism).

---

### Story 30: Zero per-tick allocations in steady state
**Verdict:** PARTIAL (existing `SimScratch` pattern extends; three concrete malloc sites remain)

**User's framing note:** "Essential, ideally we can avoid memory allocation past
startup altogether."

This is a hard constraint. No malloc past startup. Let me inventory where the
current design still allocates.

**How the DSL supports this:**
`state_world.md` already documents `SimScratch` — pooled scratch buffers
reused across tick systems, cleared+refilled within each call, `Clone` returns
`Default` (doesn't duplicate scratch). Pre-pooling baseline: "~55 page faults
per tick, 220KB/tick allocator churn." This is the existing precedent. The DSL
extends it.

The per-agent struct is already fixed-size (atomic features, contextual blocks
are fixed-width from §2.1.1–2.1.7). Slot arrays are fixed K:
`nearby_actors[K=12]`, `known_actors[K=10]`, `known_groups[K=6]`,
`memberships[K=8]`, `behavior_profile` top-K=8, etc. Everything the observation
packer reads has a static upper bound.

**Implementation walkthrough — allocation sites and elimination strategies:**

**Site 1: `behavior_profile: Vec<(u32, f32)>` (state_npc.md line 276).**
Currently unbounded sorted tag_hash → weight pairs. Grows as agent accumulates
new action tags.
- *Fix:* cap at K=16 (matches the top-K=8 observation + headroom). When a 17th
  tag would be added, evict the lowest-weight tag. Store as
  `[(u32, f32); 16]` fixed array. Weight accumulation still uses binary search.

**Site 2: Relationship records.**
`Relationship` is per-pair; an agent may know N others. Currently implied `Vec`.
- *Fix:* `known_actors[K=10]` is already the observation-visible cap; extend
  this to the storage representation. The agent's "knows about" set is
  `[Relationship; K_MAX=32]` with LRU eviction by `last_known_age`.
  Full-resolution per-pair matrix is not stored; only the top-K known
  per agent. Distant rivals are not "forgotten" — they're evicted from the
  per-agent knowledge slot when a closer relationship takes precedence, just
  as real NPCs forget unimportant others. Relationships the agent should
  never forget (spouse, mentor) are pinned.

**Site 3: `memberships: Vec<Membership>` (state_npc.md, referenced in §2.1.3).**
- *Fix:* `K=8` cap, already matches the observation slot array. Agent can be
  in at most 8 groups simultaneously. Design constraint, enforced by the DSL.

**Site 4: Memory event ring.**
`memory_events` per agent — ring buffer of recent events agent observed.
- *Fix:* fixed-size ring buffer `[MemoryEvent; 64]` with write-index. Already
  a ring in the existing code per CLAUDE memory; just make it fixed-size at
  the type level.

**Site 5: Per-tick event buffers (`world_events`, `structural_events`).**
`state_world.md` documents `structural_events: Vec<StructuralEvent>` "Cleared
at tick start." That's pool-like but still a `Vec`.
- *Fix:* `SmallVec<[StructuralEvent; 256]>` with a documented overflow
  behavior (log warning + drop oldest, or pre-size with a worst-case bound).
  Since events are bounded-per-tick (per agent, each can emit at most
  K actions), worst case is `N_agents * K_actions`. Statically sizeable.

**Site 6: Chronicle.**
`chronicle: Vec<ChronicleEntry>` (bounded ring per `state_world.md`).
- Partly fixed — ring-bounded — but entries contain `String`. Per story 60's
  "use numeric IDs for everything important", convert to
  `ChronicleEntry { category: u8, tick: u64, entities: [u32; 4], param_ids: [u32; 4] }`.
  Text rendering is a separate CPU-side display layer with its own non-sim
  allocations (which don't count — they're out of the sim loop).

**Site 7: Quest/auction lists.**
`Quest.party_member_ids: Vec<AgentId>`. Conquest quests can have hundreds of
members.
- *Fix:* quests store an `ArcWindow` into a pre-allocated pool of agent-ID
  slots OR resolve `PartyScope::Group(g)` as a pointer to the group rather
  than materializing the member list. The latter is what
  `proposal_universal_mechanics.md` Open Question 2 already asks. Recommend:
  `PartyScope::Group(g) → members-view-at-tick(g)`, computed from the
  agent's `memberships` on demand; no stored list.

**Site 8: Spatial query results.**
`query::nearby_agents(self, radius=50)` returns a slice. Currently implied Vec.
- *Fix:* use `SimScratch.snaps` — already an existing pooled buffer. DSL
  codegen routes all `query::` calls through scratch.

**Site 9: GPU buffers (story 28).**
- *Non-issue.* `GpuHarness::create_field` allocates once at startup; the
  observation / mask / logits / action buffers stay resident for the run.

**Site 10: HashMap.**
`tiles: HashMap<TilePos, Tile>`, `chunks: HashMap<ChunkPos, Chunk>`,
`surface_cache`, `cell_census` — all allocate as they populate.
- Tiles and chunks legitimately grow as the world is explored (load-time
  allocations for new chunks). This is NOT per-tick steady-state allocation
  as long as chunk loading is gated to "when NPC enters new area". For a
  truly bounded world, pre-allocate all chunks at init (small-world mode
  already does this per existing commit `fix: skip CPU chunk pre-gen in
  --world small`).
- `surface_cache` and `cell_census` are lazy caches — per
  `state_world.md` they're "populated lazily on HashMap miss." Fix: switch
  to the flat-grid versions already mentioned in the doc ("Candidates for
  flat-grid conversion").

**Remaining malloc sites that are HARD to eliminate:**
- New agent spawn (e.g. birth event) — has to allocate an agent slot. Fix:
  pool of `MAX_AGENTS` slots allocated at init with a free list; spawning
  pops a slot. Despawn returns the slot. This is the "fixed-cap world"
  commitment — `MAX_AGENTS=200_000` (story 31). ~200KB/agent × 200K =
  40 GB, too big; need to tune per-agent size down or use tiered storage
  (hot/cold separation already exists per `state_world.md`
  `entity_index`/`hot`/`cold` pattern).
- New chunk load — real allocation. Either pre-allocate all chunks (small
  world, bounded) or accept that chunk-load triggers allocation (not
  "per-tick" in steady state, because an agent rarely crosses a chunk boundary
  every tick — mostly a bulk-load event at session start).
- Save-file I/O (story 32) — serialization uses a scratch buffer, can be
  pooled.

**Concrete zero-malloc DSL constraints:**
```
agent NpcAgent {
  // All fields fixed-size
  vitals: Vitals,
  needs: [f32; 6],
  emotions: [f32; 6],
  personality: [f32; 5],
  behavior_profile: [(u32, f32); 16],      // fixed-cap
  memberships: [Membership; 8],            // fixed-cap
  known_actors: [KnownActor; 32],          // fixed-cap, LRU-evicted
  memory_events: RingBuffer<MemEvent, 64>, // fixed-cap
  ...
}

// Compile error: unbounded collection in agent struct
agent Bad { history: Vec<Event> }  // ERROR: Vec fields forbidden
```

The DSL can statically forbid unbounded fields in agent / group / quest
structs. `Vec<T>` is only allowed in scratch pools (SimScratch) or in the
world-level event ring (itself bounded).

**Gaps / open questions:**
- Event buffer worst-case sizing is a hard capacity-planning problem. If
  drops are observable (determinism break), we need a static cap proof.
  Proposal: log a tick-level event-rate metric; CI test fails if any scenario
  exceeds 80% of the static cap.
- Chunk allocation on exploration genuinely is a malloc; either accept it
  (small-world mode eliminates) or use a fixed chunk slab pool with LRU
  eviction (mirrors voxel-engine's `terrain_compute.rs` 1024-slot pool).
- Save buffer — serialization of worldstate to a `Vec<u8>` — can be pooled
  (reuse buffer across saves) or streamed to disk with a fixed chunk size.
  Needs story 32 to specify.

**Related stories:** 25 (pack kernel writes into pre-allocated GPU buffer), 28
(GPU field buffers pre-allocated), 31 (200K agents means fixed-cap slot pool
is the only viable storage), 32 (save buffer pooling), 33 (event log must be
ring-bounded or streamed).

---

### Story 31: Scale 2K → 200K agents
**Verdict:** PARTIAL (architectural approach is right; specific bottlenecks need quantification)

**User's framing note:** "Essential, the purpose behind this entire thing"

This is the north-star constraint. Everything above rolls up into whether the
design scales linearly.

**How the DSL supports this:**
The design decisions that make 100× scale possible:
1. **Single neural backend** (§2.4) — no per-role dispatch overhead.
2. **Batched forward** (story 25) — inference cost is one shader dispatch
   regardless of N.
3. **Fixed-cap per-agent state** (story 30) — O(N) memory, no hidden O(N²).
4. **Top-K slot arrays** (§2.1.4/5) — observation is O(1) per agent, not
   O(N_nearby).
5. **Per-pair relationships bounded by K_known=32** (story 30) — no O(N²)
   relationship matrix.
6. **GroupIndex contiguous ranges** (`state_world.md`) — per-settlement
   iteration is a slice, not a scan.
7. **Fidelity zones** (`state_world.md`) — agents far from player run at
   Background fidelity, skipping policy inference.

**Per-agent and per-tick cost breakdown at 200K:**

*Observation packing.* 1655 f32 per agent × 200K agents = 330M f32 = 1.3 GB
observation tensor per tick. On a GPU with 8 GB VRAM: fits, but leaves 6.7 GB
for everything else (weights, mask, logits, voxel chunks). **First pressure
point.** Options:
- Pack as fp16 instead of fp32 (→ 660 MB). Most observation features are
  bounded in [-1, 1] after normalization; fp16 has enough precision.
- Use sparse observation for Background-fidelity agents (subset of features).
  Currently not specified.
- Per-fidelity observation dimension (High=1655, Medium=400, Low=80,
  Background=skip). The proposal hints at this with "per-role K" but doesn't
  commit.

*Mask evaluation.* `NUM_LOGITS` ≈ 16 macro × 16 quest_type × 6 party × 10 reward
× 6 payment = ~92K combinations per agent. Per-head mask is much smaller:
`[16] + [16] + [6] + [10] + [6] + [50 target slots]` ≈ 104 bytes per agent =
21 MB for 200K agents. Fine.

*Neural forward.* Assume a modest network: input 1655, hidden 256, hidden 256,
heads summing to ~256 outputs. Parameters: 1655×256 + 256×256 + 256×256 ≈
555K weights. FLOPs per agent: ≈ 1.1M. Total: 2.2e11 FLOPs/tick at N=200K.
An RTX 3080 does ~30 TFLOP/s fp32, so one forward is ~7 ms. Fine. fp16 cuts
to ~2 ms. **Not a bottleneck.**

*Event volume.* Per agent per tick: 1 action + cascade (~3 events) ≈ 4
events/agent/tick = 800K events/tick. At 32 bytes/event, 25 MB/tick event
buffer. At 60 ticks/sec, 1.5 GB/sec event traffic. **Second pressure point.**
Options:
- Only the deltas relevant to materialized views go through the event handler
  kernel (per story 26). Rare events (QuestPosted, MarriageFormed) are CPU.
- Event buffer is ring-bounded; historical events dropped after view updates
  fold them in.
- Chronicle ring: per story 60, already bounded.

*Cascade rule cost.* CPU rules running over 800K events/tick. If each rule
is O(1) per event and there are ~50 rules, that's 40M rule-invocations/tick.
At 10 ns per invocation (branch + mutation), 400 ms CPU time. **Third pressure
point.** Options:
- Rules grouped by event-type dispatch table; only rules matching the event
  type run. Typical event matches 2-3 rules, not 50.
- SIMD-friendly rule bodies for hot rules (damage application, hp clamp).
- Fidelity-gated rules: Background-fidelity agents skip cascade entirely
  except for death/spawn events.

*Memory footprint.* At 200KB/agent × 200K = 40 GB. Too much. Options:
- **Hot/cold split** already exists (`state_world.md`: "`entity_index`
  sentinel sizing", `hot_entity`/`cold_entity`). Hot = packed observation
  fields (~7KB at 1655×f32) kept resident. Cold = infrequent fields (full
  memberships list, behavior_profile, memory_events, RNG stream, class
  definitions) loaded on policy-tick for High fidelity, evicted otherwise.
- With hot alone at 7KB × 200K = 1.4 GB — fits. Cold paged to SSD with LRU.
  Only High-fidelity agents need cold resident.
- Per-agent size target: hot ≤ 4KB (pack booleans, use u8 enums, drop unused
  slots). 200K × 4KB = 800 MB.

*Cross-entity queries (spatial).* `query::nearby_agents(self, radius=50)` at
N=200K is O(N) naive, O(log N) with a spatial hash. Per `state_world.md`,
`chunk_census` / `surface_grid` already exist as spatial indices.
voxel-engine's `src/ai/spatial.rs` provides GPU-side spatial indexing (named
in the module listing). Use that for slot-array gather on GPU.

**Scaling table (targets to validate):**

| Component | 2K cost | 200K cost | bottleneck? |
|---|---|---|---|
| Observation pack | 0.05 ms | 5 ms | No (linear, GPU) |
| Observation tensor size | 13 MB | 1.3 GB | Yes — fp16 + tiered fidelity |
| Mask eval | 0.01 ms | 1 ms | No |
| Neural forward | 0.07 ms | 7 ms | No |
| Event volume/tick | 8K events | 800K events | Partial — ring-bound, fold on GPU |
| Cascade rules (CPU) | 4 ms | 400 ms | **Yes — fidelity-gate or SIMD** |
| Cold state paging | 400 MB RAM | 40 GB | **Yes — hot/cold split required** |
| Spatial queries | 0.2 ms | 20 ms | No (log N with voxel-engine `ai/spatial.rs`) |

**Bottlenecks ranked:** (1) cold state memory, (2) cascade rule CPU cost,
(3) observation tensor VRAM, (4) event volume.

**Gaps / open questions:**
- Hot/cold split not yet specified in the DSL surface. Should be an
  annotation: `@hot field treasury: f32`, `@cold field creditor_ledger: [Creditor; 32]`.
- Fidelity-gated rules not specified. Extend §5: rules may annotate
  `@fidelity(>=Medium)` to skip at lower fidelity.
- Scale validation: no concrete plan to benchmark 200K in the proposal. Needs a
  milestone in `prototype_plan.md` (referenced but unwritten).
- Network size (d_model, layers) is implicit. For 200K, a small network
  (~500K params) is ideal; training may want larger. Specify in DSL as
  `backend Neural { h_dim: 256 }` (already shown in §3). The compiler enforces
  static size compatibility with the packed obs shape.

**Related stories:** 25, 26, 28, 30, 33.

---

### Story 32: Save and reload mid-run
**Verdict:** PARTIAL (serialization strategy choice needs committing; one hard decision outstanding)

**User's framing note:** "Essential"

**How the DSL supports this:**
`state_world.md` already distinguishes primary vs derived state:
> **Primary state** (irreplaceable): `tick`, `rng_state`, `next_id`, `tiles`,
> `voxel_world.chunks`, `voxel_world.sea_level`, `region_plan` (regenerable
> from seed), `build_seeds`, `chronicle`, `world_events`, `fidelity_zones`
> (zone definitions), `structural_events` (per-tick buffer).
>
> **Derived state** (rebuildable from primary): `entity_index`, `group_index`,
> `settlement_index`, `surface_cache`, `surface_grid`, `cell_census`,
> `nav_grids`, `max_entity_id`, `fidelity_zones[].entity_ids`.

Everything derived has `#[serde(skip)]` annotations and a rebuild path.
The DSL inherits this distinction via field annotations: `@primary`,
`@derived(rebuild=rebuild_nav_grids)`, `@scratch`.

**Implementation walkthrough:**

*Snapshot contents.* A full snapshot must contain:
1. `tick`, `rng_state`, `next_id`, `max_entity_id` — scalars.
2. All primary per-agent fields (`@primary` annotated).
3. Group definitions (memberships, standings, treasuries).
4. Quests/auctions in flight.
5. `tiles`, `build_seeds`, `voxel_world.chunks`, `voxel_world.sea_level`.
6. `region_plan.seed` only — the plan itself regenerates from seed.
7. `chronicle` ring (bounded — capped memory).
8. `world_events` ring (bounded).
9. **Materialized views (decision point below).**

*Decision: serialize materialized views, or recompute from event log on
load?*
- **Option A — serialize views.** Pro: instant load. Con: invariant
  risk — if the view derivation changes between save and load, the snapshot
  is wrong. Mitigation: schema hash covers view definitions; hash mismatch
  forces option B.
- **Option B — recompute from event log on load.** Pro: view derivation
  changes are safe. Con: the event log must extend back to the point where
  the materialized state was initialized. That's the whole simulation
  history, which is not ring-bounded. Infeasible for long-running sims.
- **Option C — hybrid with event-log horizon.** Serialize both the
  materialized-view value AND the tick at which it was last fully
  reconstituted. On load, replay events from that tick forward. Horizon
  bounded by the event log ring size. Works if views stabilize over short
  horizons.
- **Recommendation: Option A with schema-hash guard.** Rebuild from
  baseline (not event log) when hash mismatches, treating it as a migration.

*Event log in the snapshot.* If the event log is ring-bounded (say 10K
events), snapshot stores the ring + head-index. On load, derived views fold
from that ring forward — all earlier events are already baked into the
saved materialized view values. This bounds snapshot size.

*Snapshot size at 200K agents.* Assuming 4KB hot + 8KB cold per agent, 12KB
per agent × 200K = 2.4 GB per snapshot. Plus chunks (300KB each × ~5000 loaded
chunks = 1.5 GB). Plus event ring (10K × 32B = 320KB). Plus chronicle
(bounded, ~1 MB). Total: ~4 GB. Disk I/O at 1 GB/s SSD = 4 s save, 4 s load.
Acceptable for "save every N ticks".

*Incremental snapshots.* Save only changed chunks + agent deltas since last
snapshot. Reduces to ~100 MB for a typical save. More complex; defer to v2.

*Reload correctness.* The contract: `sim(seed).step(N).save()` + `load()` +
`step(M)` ≡ `sim(seed).step(N+M)`. Determinism tests
(`src/ai/core/tests/determinism.rs` per CLAUDE.md) already check this for the
combat sim; extend to world sim.

**Zero-malloc save/load (story 30 crossover).** Save writes into a pre-allocated
scratch buffer (reuse across saves). Load reads into the same
pre-allocated agent pool slots — no new allocations, just refill. Snapshot
format must be length-prefixed + deserialize-in-place. Bincode 2 supports
this with `decode_from_std_read_borrowed`.

**Gaps / open questions:**
- Commit on Option A vs C for view restoration.
- Event log size bound (story 33 dependency).
- Snapshot format — custom binary or bincode/postcard. Performance
  implications at 4 GB/save.
- GPU-resident buffers (story 28). On save, they must be downloaded.
  `GpuHarness::download` gives us this. On load, upload. Cost: ~1 GB GPU
  buffer traffic at PCIe bandwidth ≈ 100 ms. Add to save/load timing.

**Related stories:** 26 (materialized view persistence), 29 (save must
include full RNG state), 30 (snapshot format must be reallocation-free
on load), 33 (replay-from-snapshot is the same machinery as replay-from-tick-0).

---

### Story 33: Replay any range of ticks
**Verdict:** SUPPORTED

**User's framing note:** "Essential"

**How the DSL supports this:**
Event sourcing (§1.1) is the foundation: "current state is a fold over events
+ entity baseline." Replay is literally running the fold.

Story 60's ruling — "text gen should not be load bearing. Use numeric IDs for
everything important" — explicitly bounds replay determinism: text-bearing
events are allowed to diverge on replay.

**Implementation walkthrough:**

*Replay model: event log + RNG + baseline.*
Recorded artifacts per run:
1. Initial snapshot (story 32 save format) at tick T0.
2. Event log from T0 onward. Each event entry:
   `{ tick: u64, kind: u16, params: [u32; 4], source_agent: u32 }`.
   Numeric IDs only. No strings.
3. Tick-boundary RNG checkpoints (every 100 ticks). Bounds drift on replay
   divergence: any mismatch is caught within 100 ticks.

To replay from tick A to tick B:
1. Load snapshot at tick ≤ A (nearest).
2. Replay events from snapshot-tick to A, advancing sim step-by-step.
3. Optionally halt and emit a full snapshot at A.
4. Continue step-by-step from A to B, allowing inspection hooks.

*Event types that MUST be preserved.*
- All events that drive materialized view updates (story 26).
- All events that drive policy-observable state (positions, HP, relationships).
- All action events that trigger cascades.
- RNG state at each tick boundary checkpoint.

*Event types that can be skipped on replay.*
- `ChronicleEntry` emissions (story 60 — text is post-hoc). The tick at
  which a chronicle was emitted matters for age-based observations
  (`recent_chronicle_event_counts_by_category` in §2.1.7). So record the
  category + tick but not the text. Text regenerates on display.
- `WorldEvent::Generic{text}` — same treatment, carry category ID.

*Deterministic re-execution.* Story 29's contract makes this work: same seed +
same DSL + same input → same output. Input to a replay segment is:
  - snapshot state
  - events (numeric ID form) from the recorded log
  - RNG state at segment start
If the replay diverges, the tick-checkpoint RNG mismatches and we fail fast
with "replay divergence at tick T".

*Text-gen interaction.* User-facing display of a replay re-runs the chronicle
renderer on the numeric events; LLM-generated names for new NPCs born during
replay may differ from the original run. This is explicitly OK per story 60.
The cast of agent IDs is identical — only display strings differ.

*Storage cost.* 200K agents × 4 events/tick × 32 bytes/event = 25 MB/tick.
At 60 tps, 1.5 GB/s. Per hour of sim: 5.4 TB. **Too much to store raw.**
Compression options:
- Batched delta encoding (many events per tick are ambient decay with same
  shape) → 10-50× reduction.
- Log only the "policy-input-affecting" events, fold derivation locally.
  Any event used solely to update a materialized view can be reconstructed
  from its inputs (the agent state pre-event + the rule). Hard to formalize.
- Snapshot every 10 minutes + event log per 10-minute segment. 10-min segment
  = 900 GB uncompressed, 50 GB compressed. Still large.
- **Practical replay scope:** not "replay an entire hours-long run", but
  "replay a bug-report window of 1000 ticks (≈16 sec)". 1000 ticks × 25 MB
  = 25 GB uncompressed, 2-5 GB compressed. Feasible.

*Replay of GPU-sim.* Replay uses the same GPU path; nothing about replay
requires CPU. As long as GPU determinism (story 29 GPU-determinism traps) is
enforced, replay produces bit-identical output.

*Policy weight versioning.* Replaying a run requires the exact policy
checkpoint used originally. Snapshot metadata records `policy_weights_hash`
from §4. Loading a snapshot with mismatched weight hash → hard error.

**Gaps / open questions:**
- Storage compression strategy not specified. Needs design.
- Whether replay is guaranteed only for a bug-report window or for whole
  runs. Practical scope recommendation is bug-report window (1000 ticks).
- Partial-replay for debugging — can we replay only a *subset* of agents? No:
  agent interactions cross-cut; a partial replay gives a partial sim, which
  is the "A/B compare" story 36, not 33.
- Text-gen events that *feed back* into sim (agent speech influencing
  listener NPC). Per user: "Use numeric IDs for everything important" — so
  the text itself is not load-bearing; only the act of speech (event
  `SpeechEmitted{topic_id, speaker, audience}`) matters. The actual text is
  display-only. Compiler should forbid policy code from reading
  `event.text` fields.

**Related stories:** 26 (materialized views depend on deterministic
re-execution), 28 (GPU determinism is a replay precondition), 29 (replay is
determinism validated over time), 32 (snapshot = replay starting point),
38 (self-contained tick reproduction is replay-of-range N=1), 60 (text-gen
boundary rules).

---

## Cross-cutting themes

1. **Hot/cold split is a recurring requirement.** Stories 30, 31, 32 all
   depend on it. The DSL should specify `@hot` vs `@cold` annotations on
   agent fields as a first-class concern, not bolt-on.

2. **Event-type IDs instead of strings.** Story 60's "numeric IDs for
   everything important" rules out strings from the event log (story 33),
   materialized-view handlers (story 26), and replay artifacts. Needs to be
   elevated to a schema-level rule: **no `String` fields in `@primary`
   event or state structs.** Strings only allowed in display-time render
   code.

3. **voxel-engine is the GPU backend, not wgpu.** Update
   `proposal_policy_schema.md` §5 from "wgpu/CUDA" to "ash/Vulkan via
   `voxel_engine::compute::GpuHarness`". Shader compilation via `shaderc`
   (already in voxel-engine's build-dependencies). SPIR-V bytes loaded via
   `GpuHarness::load_kernel`.

4. **Compile-time validation is the DSL's differentiation.** Story 27 asks
   for it explicitly. The compiler needs a static-analysis pass covering:
   event-type existence, field-write race detection, cascade cycle detection,
   schema-hash drift, and optionally invariant checking. This is a
   multi-week piece of work but it's the payoff justifying the DSL over
   hand-written Rust.

5. **Fixed-cap everything.** Zero-malloc (story 30) and 200K scale (story 31)
   both require this. The DSL surface should statically reject unbounded
   collections in agent/group/quest/event structs. Only `SimScratch` and
   world-level ring buffers may contain `Vec`. This constraint propagates
   through observation design (top-K slots already do this) and quest
   semantics (`party_member_ids` resolved from groups lazily, not stored).

6. **Determinism under GPU compilation is under-specified.** Story 29's
   core contract is supported, but story 28's GPU-compilation target
   introduces new hazards (float associativity, atomic ordering, warp-count
   dependence). Needs a dedicated §6 in the policy schema proposal.

## Suggested proposal-schema deltas

- **§5 GPU compilation** — rename "wgpu/CUDA" to
  "Vulkan via voxel-engine `GpuHarness` + SPIR-V via shaderc".
  Add "GPU determinism constraints" subsection covering float
  associativity, atomic ordering, workgroup-size pinning.

- **New §6 Rule composition** — phase-based ordering
  (pre_event / event / post_event), commutativity within phase, compile-time
  race detection, cycle detection in cascade-rule graph.

- **New §7 Static analysis / compile-time checks** — enumerate the
  compile-time validations (story 27): event-type existence, field-write
  races, schema-hash drift, unbounded-collection rejection in agent structs.

- **New §8 Hot/cold storage model** — `@hot`/`@cold` field annotations,
  cold-eviction policy, fidelity-zone integration with cold paging.

- **Extension to §4 versioning** — schema hash scope includes: observation
  shape, action vocabulary, event-type registry, rule set identity (hash
  of sorted rule bodies).

- **Extension to §2.4 backend** — explicit commitment that the backend
  trait is called with GPU-resident `PackedObservationBatch` handles
  (not owned buffers); weight loading is a separate lifecycle op, not
  per-tick.
