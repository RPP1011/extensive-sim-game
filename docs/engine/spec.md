# World Sim Engine Specification

Runtime contract for implementers. Companion to `../dsl/spec.md` (language reference) and `../compiler/spec.md` (text-to-engine lowering). An implementation in any language is correct if and only if it satisfies §§2–23.

The engine owns determinism-load-bearing, cross-cutting infrastructure: entity pools, event ring, spatial index, mask tensor, tick scheduler, policy trait, trajectory emit, save/load, invariant runtime, probe harness, telemetry sink, schema hash, observation packer, and the debug & trace runtime. It does **not** parse DSL, generate code, hard-code domain-specific types, or render chronicle prose.

---

## 1. Scope

The engine is a Rust library crate (`crates/engine/`) that provides runtime primitives the DSL compiler targets. Primitives are generic over agent/event/action kind so that different DSL programs — a medieval town sim, a wolf-pack ecology, a sci-fi colony — reuse the same engine without code changes.

**Owned by the engine:**

- Generic `Pool<T>` with `NonZeroU32` IDs, freelist, slot reuse.
- Event ring with byte-stable SHA-256 over the replayable subset.
- 3D spatial index (2D columns + per-column z-sort + movement-mode sidecar).
- Per-world PCG RNG + per-agent keyed sub-streams.
- Universal `MicroKind` enum (18 variants) and `MacroKind` enum (4 variants) with built-in step semantics for primitives that don't require domain types.
- `MaskBuffer` with per-head validity layout.
- `PolicyBackend` trait + zero-alloc tick loop contract.
- `MaterializedView` trait with three storage modes.
- `TrajectoryWriter` + `TrajectoryReader` over safetensors.
- State snapshot + load with schema-hash versioning.
- `Invariant`, `Probe`, `TelemetrySink` traits.
- Debug & trace runtime (trace_mask, causal_tree, tick_stepper, tick_profile, agent_history, snapshot).

**Not owned by the engine** (deferred to compiler / domain):

- DSL parser and codegen.
- Verb desugaring, `Read → Ask(doc, AboutAll)` lowering.
- Domain types (which items exist, which abilities exist, which group kinds exist).
- Cascade *rules* (registered at init; engine provides the dispatch runtime).
- Chronicle prose templates.
- Curriculum pipelines (engine provides the per-scenario runner; scheduling is external).
- GPU compute kernels (voxel-engine's `GpuHarness` is a downstream consumer).

---

## 2. Determinism contract

The engine promises: same seed + same compile-time `schema_hash` + same agent spawn order + same action sequence ⇒ bit-exact SHA-256 hash over the replayable-subset event log, on every supported platform.

Implementation obligations:

- All randomness derives from `WorldRng` (PCG-XSH-RR). Calling `rand::thread_rng` or any external RNG from inside the engine is a determinism bug.
- Iteration order over `HashMap` / `HashSet` is forbidden in hot paths. Use `BTreeMap` / `BTreeSet` or index Vecs.
- Float reductions that feed decisions must sort by a stable key (agent id, event id) before accumulation. Associativity violations break determinism across thread counts.
- Per-tick agent action ordering is shuffled via Fisher-Yates keyed on `per_agent_u32(seed, AgentId(1), tick << 16 + i, b"shuffle")`. No first-mover bias.
- Event byte-packing uses `f32::to_bits().to_le_bytes()` plus explicit variant tags — `Debug` formatting is NOT stable and is forbidden in hash input.
- The `replayable_sha256()` result is the canonical determinism witness; CI runs a 100-agent × 1000-tick acceptance test and pins its hash.

Two same-seed runs that disagree on `replayable_sha256()` are a blocker. The first step is always `git bisect` over the engine crate — domain code cannot violate determinism if the engine's contract holds.

See `../dsl/spec.md` §7.2 for the language-level framing; this section pins the byte-level mechanics. Implementation: `crates/engine/src/event/ring.rs`, `crates/engine/src/rng.rs`, `crates/engine/tests/determinism.rs`.

---

## 3. State model

Agents live in `Pool<Agent>` where `AgentId` is `NonZeroU32` (giving `Option<AgentId>` a free niche). The pool has three layers:

1. **Slot index.** `AgentSlotPool` tracks which `u32` slots are alive via a boolean Vec and a freelist; `alloc()` pops the freelist, `kill()` pushes. Slot numbers can be reused after death; generational counters are out of scope for MVP — callers hold live references only through the tick they were alive.
2. **Hot fields.** One `Vec` per hot field: `hot_pos: Vec<Vec3>`, `hot_hp: Vec<f32>`, `hot_max_hp: Vec<f32>`, `hot_alive: Vec<bool>`, `hot_movement_mode: Vec<MovementMode>`. Layout is true Structure-of-Arrays. A field is hot if it's read or written every tick.
3. **Cold fields.** One `Vec<Option<T>>` per cold field: creature type, channel set, spawn tick. Read rarely (spawn, chronicle, debug).

Bulk-slice accessors (`hot_pos()`, `hot_hp()`, …) are the kernel-friendly read path. Per-id accessors (`agent_pos(id)`, `set_agent_hp(id, v)`) are the scalar path for domain code that walks one agent at a time.

```rust
pub struct SimState {
    pub tick: u32,
    pub seed: u64,
    pool: AgentSlotPool,
    hot_pos:           Vec<Vec3>,
    hot_hp:            Vec<f32>,
    hot_max_hp:        Vec<f32>,
    hot_alive:         Vec<bool>,
    hot_movement_mode: Vec<MovementMode>,
    cold_creature_type: Vec<Option<CreatureType>>,
    cold_channels:      Vec<Option<ChannelSet>>,
    cold_spawn_tick:    Vec<Option<u32>>,
}
```

Aggregate (non-spatial) entities — parties, charters, treaties, quests — live in a parallel `AggregatePool<T>` (§14).

See `../compiler/spec.md` §1.3 for the hot/cold split heuristic; the engine implements the mechanism and the compiler decides which DSL fields land in which bucket. Implementation: `crates/engine/src/state/`.

---

## 4. Event log

`EventRing` is a VecDeque with a fixed capacity, seeded by `with_cap(n)`. `push(event)` appends; if full, the oldest event is dropped. `iter()` yields events in push order.

Events partition into two classes:

- **Replayable** — contribute to the deterministic hash. Engine-owned variants: `AgentMoved`, `AgentAttacked`, `AgentDied`. Compiler-emitted variants are registered via the `Event` enum's `#[non_exhaustive]` extension point (open sum type in the compiler's codegen).
- **Chronicle** — prose side-channel. `Event::ChronicleEntry { text: String, tick: u32 }` holds template-rendered text for dev-facing UIs. Excluded from the replayable hash. Templates can reference state nondeterministically (locale, prose variations) without breaking determinism.

The byte-packing format for hash input:

| Offset | Bytes | Content |
|--------|-------|---------|
| 0      | 1     | Variant tag (0 = AgentMoved, 1 = AgentAttacked, 2 = AgentDied, 128+ = compiler-extended) |
| 1..    | var   | Packed fields: u32 → LE bytes, f32 → `to_bits().to_le_bytes()` |

`replayable_sha256()` hashes the concatenation of all packed replayable events in push order. The tick number is included in every event struct so the hash is sensitive to ordering.

Cascade events (events emitted by physics handlers in response to other events, §9) carry `cause: Option<EventId>` where `EventId = (tick: u32, seq: u32)`. This enables `causal_tree()` queries (§22) without affecting the hash — the `cause` field is NOT included in the byte-packed hash input; only the event's own fields are.

Adding a new replayable variant bumps the schema hash (§20). Adding a new chronicle variant does not.

See `../dsl/spec.md` §2.2 (language grammar) and §7.3 (replay scope). Implementation: `crates/engine/src/event/`.

---

## 5. Spatial index

The spatial index is a 2D-column BTreeMap + per-column z-sort + movement-mode sidecar.

```rust
pub struct SpatialIndex {
    columns:  BTreeMap<(i32, i32), SortedVec<(f32, AgentId)>>,
    sidecar:  Vec<AgentId>,   // fliers / swimmers / climbers
    cell_size: f32,           // 16.0 m, aligned to voxel-chunk edges
}
```

Agents with `MovementMode::Walk` live in the column map, keyed on `(floor(pos.x / cell_size), floor(pos.y / cell_size))` with their `z` position as the sort key within the column. Agents with any other movement mode (Fly / Swim / Climb / Fall) live in the sidecar and are scanned linearly on every query; the expected population of the sidecar is much smaller than the total alive count.

Query API (all return owned `Vec<AgentId>` in the MVP; §17 zero-alloc variants via scratch):

- `within_radius(center: Vec3, r: f32) -> Vec<AgentId>` — 3D Euclidean distance.
- `nearest_k(center: Vec3, k: usize) -> Vec<(f32, AgentId)>` — k nearest in 3D.
- `in_column_z_range(col: (i32, i32), z_range: Range<f32>) -> Vec<AgentId>`.

All distance checks use 3D Euclidean (the XY-only "planar" variant is a separate API clearly named — confusing the two has been a historical regression, see spec §9 D25).

Updates on `spawn_agent`, `kill_agent`, and `set_agent_pos` (which also updates the column key when the agent crosses a cell boundary).

The BTreeMap ordered iteration guarantees determinism. The `SortedVec` insertion preserves the z-order.

See `../dsl/spec.md` §9 D25 for the decision rationale; this section pins the API. Implementation: `crates/engine/src/spatial.rs`.

---

## 6. RNG streams

`WorldRng` is PCG-XSH-RR with a single 64-bit state + 64-bit stream id, constructed by `from_seed_with_stream(seed: u64, stream_id: u64)`. The standalone `next_u32()` and `next_u64()` methods mutate state.

For operations that need independent streams (per-agent sampling, per-event noise, shuffle keys), the engine exposes keyed hash functions that do NOT mutate the global state:

```rust
pub fn per_agent_u32(
    world_seed: u64,
    agent_id: AgentId,
    tick_or_nonce: u32,
    purpose: &[u8],
) -> u32;

pub fn per_agent_u64(
    world_seed: u64,
    agent_id: AgentId,
    tick_or_nonce: u32,
    purpose: &[u8],
) -> u64;
```

Implementation: a fixed-keyed `ahash::RandomState::with_seeds(K1, K2, K3, K4)` where K1..K4 are hard-coded constants baked into the engine (`0xA5A5_A5A5_A5A5_A5A5`, `0x5A5A_5A5A_5A5A_5A5A`, `0xDEAD_BEEF_CAFE_F00D`, `0x0123_4567_89AB_CDEF`). The `runtime-rng` feature of `ahash` is explicitly disabled — that feature would reseed the hasher from system entropy per process and break determinism. Pinning these constants is how the engine guarantees `per_agent_u32(...)` returns the same value across runs, compilers, and architectures.

Golden tests (`crates/engine/tests/rng.rs`) pin specific input→output pairs for both functions. Any change to K1..K4 or to the ahash algorithm bumps the schema hash (§20).

See `../dsl/spec.md` §9 D12. Implementation: `crates/engine/src/rng.rs`.

---

## 7. Action space — MicroKind

The engine carries a closed set of 18 `MicroKind` variants covering the universal primitives from `../dsl/spec.md` Appendix A. These are the *runtime* primitives — after the compiler has lowered `Read(doc)` into `Ask(doc, QueryKind::AboutAll)`.

```rust
#[repr(u8)]
pub enum MicroKind {
    // Movement (3)
    Hold, MoveToward, Flee,
    // Combat (3)
    Attack, Cast, UseItem,
    // Resource (4)
    Harvest, Eat, Drink, Rest,
    // Construction (3)
    PlaceTile, PlaceVoxel, HarvestVoxel,
    // Social (2)
    Converse, ShareStory,
    // Info — push + pull (2)
    Communicate, Ask,
    // Memory (1)
    Remember,
}

impl MicroKind {
    pub const ALL: &'static [MicroKind] = &[/* 18 variants */];
}
```

Step semantics break into three tiers:

1. **Fully implemented in the engine** — `Hold`, `MoveToward`, `Flee`, `Attack`, `Eat`, `Drink`, `Rest`. These need only `SimState` fields (pos, hp, hunger, thirst, rest_timer) and can apply without a domain-specific cascade.
2. **Needs compiler-registered cascade** — `Cast`, `UseItem`, `Harvest`, `PlaceTile`, `PlaceVoxel`, `HarvestVoxel`, `Converse`, `ShareStory`, `Communicate`, `Ask`, `Remember`. The engine emits the corresponding event (`AgentCast`, `AgentHarvested`, `InformationRequested`, etc.) and delegates the effect to a handler registered via `CascadeRegistry::register` (§9).
3. **Stubbed** — none in MVP; every variant has either built-in step semantics or a registered cascade.

Every `Action` that flows through the tick loop carries a `MicroKind` plus a typed target slot:

```rust
pub struct Action {
    pub agent: AgentId,
    pub micro_kind: MicroKind,
    pub target: ActionTarget,
}

pub enum ActionTarget {
    None,
    Agent(AgentId),
    Position(Vec3),
    ItemSlot(u8),
    AbilityIdx(u8),
    Resource(ResourceRef),
    Voxel(IVec3),
    FactRef(EventId),
    Document(ItemId),
    Query(QueryKind),
}
```

Extending the action vocabulary to add a new universal primitive (e.g. if we discover the spec is missing one) means: add a variant to `MicroKind`, add step semantics OR a cascade hook, bump the schema hash baseline. Domain-specific verbs are handled by compiler desugaring on top of existing primitives, not by extending `MicroKind`.

See `../dsl/spec.md` §3.3 and Appendix A. Implementation: `crates/engine/src/policy/action.rs` (MVP has a subset; full 18 lands in Phase 2 of engine build-out).

---

## 8. Macro mechanisms

The engine carries 4 `MacroKind` variants — the universal action mechanisms that exist in every world:

```rust
#[repr(u8)]
pub enum MacroKind {
    PostQuest,            // auction / contract posting
    AcceptQuest,          // aliased with JoinParty for group-joining semantics
    Bid,                  // place a bid on an open auction
    Announce,             // broadcast information to group / area / anyone
}
```

Each macro has a complex parameter head set (quest_type, party_scope, quest_target, reward_kind, payment_kind, group_kind, announce_audience, standing_kind, resolution) enumerated in `../dsl/spec.md` §3.2. The engine holds those enums as data types; semantics (which quest kinds exist in a given world, how a specific `Resolution` breaks ties) come from compiler-registered handlers.

The 4 macros + 18 micros = 22 runtime action kinds in the `MicroKind` ∪ `MacroKind` space. When a policy emits an action with `macro_kind != NoOp`, the micro parameter heads are ignored and vice versa (spec §3.2 line 642).

Built-in cascade on emission:

- **`PostQuest`** → emit `QuestPosted { quest_id, poster, … }`; compiler-registered quest bookkeeping handler.
- **`AcceptQuest`** → emit `QuestAccepted { quest_id, acceptor, role }`; compiler handler resolves role-in-party.
- **`Bid`** → emit `BidPlaced { auction_id, bidder, payment }`; compiler handler implements resolution policy.
- **`Announce`** → cascade enumerates recipients (group members, area radius, or `MAX_ANNOUNCE_RADIUS` sphere), emits `RecordMemory` per recipient; overhear scan adds bystanders with reduced confidence. Bounded by `MAX_ANNOUNCE_RECIPIENTS`. This cascade IS implemented in the engine, not delegated, because it's fully specified by the spatial index + the recipient enumeration — no domain-specific logic.

Implementation: `crates/engine/src/policy/macro.rs` (MVP has NoOp + AnnounceAction; full 4 lands in Phase 2).

---

## 9. Physics cascade runtime

After an action applies and emits its primary event, registered cascade handlers run. A cascade handler is:

```rust
pub trait CascadeHandler: Send + Sync {
    fn trigger(&self) -> EventKindId;
    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing);
}

pub struct CascadeRegistry {
    handlers: FxHashMap<EventKindId, SmallVec<[Box<dyn CascadeHandler>; 4]>>,
}

impl CascadeRegistry {
    pub fn register<H: CascadeHandler + 'static>(&mut self, h: H);
    pub fn dispatch(&self, event: &Event, state: &mut SimState, events: &mut EventRing) -> usize;
}
```

Dispatch runs every handler whose `trigger()` matches the event kind. Handlers MAY emit new events into `events`; those events are themselves dispatched in the same tick. This creates a fixed-point loop.

**Bounded iteration.** The loop terminates after `MAX_CASCADE_ITERATIONS = 8` passes. In dev builds (debug assertions on), exceeding the bound panics with the cascade trail. In release, the engine logs a warning and truncates — the overflowing events are emitted but their downstream cascades are not dispatched this tick. They may still fire next tick if the triggering event re-occurs.

**Handler ordering.** Within a single event dispatch, handlers run in registration order. The DSL-compiler resolves mod-level conflicts via lane discipline (`../compiler/spec.md` §Decisions D16 — Validation / Effect / Reaction / Audit lanes with lexicographic sort); the engine preserves lane order via a per-lane registry structure. Non-modded engine tests see only one lane.

**Handler purity.** Handlers must not hold external references or block. They see `&mut SimState` and `&mut EventRing` exclusively. Sleeping, IO, or RNG-outside-the-keyed-streams is forbidden — violations break determinism.

See `../dsl/spec.md` §2.4 (language grammar) for how DSL `physics` rules become registered handlers. Implementation: `crates/engine/src/cascade/` (not yet built; lives in the engine build-out plan).

---

## 10. Mask buffer

The `MaskBuffer` is the per-head validity tensor consumed by the policy backend. Two heads are built-in:

- `micro_kind`: `Vec<bool>` of size `n_agents × MicroKind::ALL.len()`.
- `target`: `Vec<bool>` of size `n_agents × TARGET_SLOTS` where `TARGET_SLOTS = 12`.

The layout is `bit[slot * n_kinds + kind_idx]` for `micro_kind`, `bit[slot * TARGET_SLOTS + target_idx]` for `target`.

```rust
pub struct MaskBuffer {
    pub micro_kind: Vec<bool>,
    pub target:     Vec<bool>,
    pub n_agents:   usize,
}

impl MaskBuffer {
    pub fn new(n_agents: usize) -> Self;
    pub fn reset(&mut self);
    pub fn mark_hold_allowed(&mut self, state: &SimState);
    pub fn mark_move_allowed_if_others_exist(&mut self, state: &SimState);
    // Domain predicates registered via MaskBuilder below.
}
```

Additional heads (macro_kind, channel, quest_type, announce_audience, …) are added via:

```rust
pub trait MaskBuilder {
    fn add_head(&mut self, name: &str, n_bits_per_agent: usize) -> HeadId;
    fn set(&mut self, head: HeadId, agent_slot: usize, bit: usize, allowed: bool);
}
```

Universal predicates (target-in-range, cooldown-ready, is-alive, has-free-inventory-slot) are built into the engine. Domain predicates (quest-eligibility, standing-at-war, member-of-group) are registered by compiler-generated code.

**Mask validity invariant:** every `Action` returned by a policy must correspond to a `true` bit in the mask that was passed to its `evaluate` call. The engine enforces this in a regression test (§21 §7); a violating backend is a bug.

See `../dsl/spec.md` §2.5 (grammar) and Appendix B.2 (concrete predicates). Implementation: `crates/engine/src/mask.rs`.

---

## 11. Policy backend

The policy backend is a trait invoked once per tick per agent-batch. Zero-alloc is a hard requirement on the hot path.

```rust
pub trait PolicyBackend: Send + Sync {
    fn evaluate(
        &self,
        state: &SimState,
        mask:  &MaskBuffer,
        out:   &mut Vec<Action>,
    );
}
```

The engine ships two implementations in-tree:

- **`UtilityBackend`** — hand-scored argmax over masked candidates. Score table is a `&'static [(MicroKind, f32)]` + HP-aware bonuses (low HP → Eat > Attack; high HP → Attack > Rest). Tie-break rule: lowest `MicroKind::ALL` index wins. Used for bootstrapping and as the regression baseline. Scoring rules live in the compiler-generated `utility_rules.rs` when the DSL has `backend "utility"`; the in-tree default is for smoke tests.
- **`NeuralBackend`** — STUB in MVP. Trait-only: loads safetensors weights and runs inference. Full implementation deferred; see Phase 2 engine plan.

Additional backends (`LlmBackend`, `GoapBackend`) are implemented by the compiler or by downstream crates. The engine doesn't take a dependency on any ML framework; `NeuralBackend` impls live outside the engine crate.

**Argmax rule.** When two actions tie on score, the one with the lower `MicroKind` ordinal index wins. This is deterministic and reproducible across runs. Stochastic sampling (softmax + temperature) is a compiler-generated feature; the engine exposes hooks but doesn't implement it in `UtilityBackend`.

See `../dsl/spec.md` §2.7 (grammar) and §3.5 (backend semantics). Implementation: `crates/engine/src/policy/`.

---

## 12. Tick pipeline

One tick is six ordered phases:

1. **Mask build.** Reset `MaskBuffer`, mark universal predicates, dispatch registered domain predicates. Reads `SimState`, writes `MaskBuffer`.
2. **Policy evaluate.** `backend.evaluate(state, mask, &mut actions)`. Reads both, writes `actions`.
3. **Action shuffle.** Per-tick Fisher-Yates over `0..actions.len()` using `per_agent_u32(seed, AgentId(1), tick << 16 + i, b"shuffle")`. Determinism-load-bearing — prevents first-mover bias.
4. **Apply actions + emit events.** For each action in shuffled order: mutate state (pos, hp, cooldown), push primary event, run cascade (§9).
5. **View fold.** Registered materialized views see all events emitted this tick. Views are incremental: `view.fold(events_since_last_tick)`.
6. **Invariant + telemetry.** Registered invariants (§17) check post-state; violations dispatch per their failure mode. Telemetry sink (§19) emits built-in metrics (tick_ms, event_count, agent_count) and any registered domain counters.

After phase 6, `state.tick += 1`. `SimScratch` holds all per-tick buffers (`mask`, `actions`, `shuffle_idx`) to achieve zero steady-state allocation.

```rust
pub fn step<B: PolicyBackend>(
    state:   &mut SimState,
    scratch: &mut SimScratch,
    events:  &mut EventRing,
    backend: &B,
    cascade: &CascadeRegistry,
    views:   &mut [&mut dyn MaterializedView],
    invariants: &[&dyn Invariant],
    telemetry:  &dyn TelemetrySink,
);
```

The MVP `step()` signature is simpler (no cascade / views / invariants / telemetry); they plug in as the build-out progresses. See the engine-build-out plan in `../superpowers/plans/`.

See `../dsl/spec.md` §7.1. Implementation: `crates/engine/src/step.rs`.

---

## 13. Views

Views are derived state computed from events. The engine supports three storage modes:

- **`materialized`** — a full per-entity Vec, folded every tick. `DamageTaken: Vec<f32>` indexed by agent slot is the canonical example.
- **`lazy_cached`** — computed on demand, cached with a staleness marker. Invalidated when any triggering event is pushed. Good for expensive queries that most ticks don't need.
- **`per_entity_topk(K, keyed_on)`** — fixed-size top-K per entity, e.g. "the 8 agents with highest reputation toward me". Bounded memory regardless of N.

```rust
pub trait MaterializedView: Send + Sync {
    fn fold(&mut self, events: &EventRing);
}

pub trait LazyView: Send + Sync {
    fn invalidated_by(&self) -> &[EventKindId];
    fn compute(&mut self, state: &SimState);
    fn is_stale(&self) -> bool;
}

pub trait TopKView: Send + Sync {
    const K: usize;
    fn update(&mut self, event: &Event);
    fn topk(&self, agent: AgentId) -> &[(AgentId, f32); Self::K];
}
```

Storage mode is chosen by the compiler from DSL `@materialized(storage=<hint>)` annotations; the engine exposes the three traits without making routing decisions. GPU eligibility follows from storage hint and is a compiler-side concern (`../compiler/spec.md` §1.2).

**Determinism.** For `materialized` views over commutative scalar operations (sum, max), events must be sorted by target id before reduction to avoid float-associativity drift (§2). The engine's reference `DamageTaken::fold` uses in-order iteration which is deterministic by construction; GPU implementations need the explicit sort.

See `../dsl/spec.md` §2.3. Implementation: `crates/engine/src/view/`.

---

## 14. Aggregates

Aggregates are entity-shaped things that don't have a spatial position: parties, charters, treaties, quests, groups. They live in a parallel pool.

```rust
pub struct AggregatePool<T> {
    slots:     Vec<Option<T>>,
    freelist:  Vec<u32>,
    alive:     Vec<bool>,
}

impl<T> AggregatePool<T> {
    pub fn alloc(&mut self, t: T) -> AggregateId;
    pub fn kill(&mut self, id: AggregateId);
    pub fn get(&self, id: AggregateId) -> Option<&T>;
    pub fn get_mut(&mut self, id: AggregateId) -> Option<&mut T>;
}
```

`AggregateId` is `NonZeroU32` like `AgentId` — same niche optimization, same slot-reuse semantics. The engine instantiates `AggregatePool<Quest>`, `AggregatePool<Group>`, etc. based on compiler-generated type definitions.

Cross-references (aggregate → member agents, agent → memberships) are materialized views (§13) maintained by cascade handlers on the relevant events (`MemberJoined`, `MemberLeft`, `GroupDissolved`).

**Dissolution semantics.** When a `Group` is killed, cascade handlers cascade-mark all `Membership` records invalid; the `memberships` materialized view rebuilds from the event log within one tick. Dangling `GroupId` references in agent state become `Option::None` on the next `memberships::fold` pass.

See `../dsl/spec.md` §5 type system. Implementation: `crates/engine/src/pool.rs` (generic over T; agent and aggregate pools both use it).

---

## 15. Trajectory emission

Trajectories are per-tick state snapshots serialized to safetensors for downstream ML training. The writer registers named tensors; each `record_tick` call appends one row.

Built-in tensors (produced on `TrajectoryWriter::new(n_agents, n_ticks)`):

| Name        | Shape       | Dtype | Semantics                              |
|-------------|-------------|-------|----------------------------------------|
| `positions` | `[t, n, 3]` | f32   | Per-agent 3D position at each tick     |
| `hp`        | `[t, n]`    | f32   | Per-agent HP at each tick              |
| `tick`      | `[t]`       | u32   | Absolute tick number                   |

Extension API:

```rust
impl TrajectoryWriter {
    pub fn register_tensor<F>(&mut self, name: &'static str, shape: &[usize],
                              dtype: Dtype, getter: F)
    where F: Fn(&SimState, &mut [u8]) + Send + Sync + 'static;
}
```

Custom tensors (observations, actions, rewards, attention maps) register via `register_tensor` with a getter that writes the per-tick row. The writer's `write(path)` emits the combined safetensors file.

**File format.** `safetensors` — canonical because it has Python round-trip support, a stable header format, and no pickle security surface. The writer guarantees that two `cargo test` runs with the same seed produce byte-identical tensor values (but NOT byte-identical files — safetensors metadata ordering is not guaranteed; `TrajectoryReader::load` is the value-equality oracle).

See `../dsl/spec.md` §6. Implementation: `crates/engine/src/trajectory.rs`; Python round-trip: `scripts/engine_roundtrip.py`.

---

## 16. Save / load

State snapshots capture: SoA field Vecs, freelist, tick, seed, event-ring tail (last N events for replay-from-snapshot), schema hash.

```rust
pub fn save_snapshot(state: &SimState, events: &EventRing, path: &Path) -> Result<(), Error>;
pub fn load_snapshot(path: &Path) -> Result<(SimState, EventRing), Error>;
```

File layout:

| Block                | Size              | Content                                   |
|----------------------|-------------------|-------------------------------------------|
| Header               | 64 B              | Magic (`WSIMSV01`), schema_hash (32 B), tick, seed |
| SoA hot field Vecs   | `n_fields × n × 4`| Little-endian field bytes                 |
| SoA cold field Vecs  | varies            | Option<T> encoded with present bit + body |
| Slot pool state      | `n + freelist_len × 4` | alive Vec<bool> + freelist Vec<u32>   |
| Event ring tail      | varies            | Serialized events for replay continuity   |

Loading rejects a snapshot whose header `schema_hash` differs from the current binary's `schema_hash` with a hard error. To migrate, the caller registers a migration function:

```rust
pub fn register_migration(from_hash: [u8; 32], to_hash: [u8; 32],
                          migrate: impl Fn(&[u8]) -> Result<Vec<u8>, Error>);
```

Migrations are chain-composable — a save at hash `A` can be loaded at hash `C` if migrations `A→B` and `B→C` are both registered.

**Non-goals:** partial snapshots (one region, one faction) are a Phase 3 optimization. MVP is whole-world only.

See `../dsl/spec.md` §7.4. Implementation: `crates/engine/src/snapshot.rs` (not yet built).

---

## 17. Invariant runtime

Invariants are assertion-shaped functions that run in phase 6 of the tick loop. Failure modes are compile-time configurable.

```rust
pub trait Invariant: Send + Sync {
    fn name(&self) -> &'static str;
    fn check(&self, state: &SimState, events: &EventRing) -> Option<Violation>;
    fn failure_mode(&self) -> FailureMode;
}

pub enum FailureMode {
    Panic,         // dev builds — abort the process with the violation message
    Log,           // prod builds — emit telemetry event, continue
    Rollback(u32), // replay from tick - N using event log; useful for soft invariants
}

pub struct Violation {
    pub invariant: &'static str,
    pub tick:      u32,
    pub message:   String,
    pub payload:   Option<serde_json::Value>,
}
```

Built-in invariants:

- `mask_validity` — every action in last tick had its mask bit set.
- `pool_non_overlap` — no agent slot is both alive and in the freelist.
- `event_hash_stable` (dev only) — re-hashing the replayable subset gives the same result.

Domain invariants (e.g. "no agent has negative hp", "every active quest has at least one party member") are registered by compiler-generated code.

See `../dsl/spec.md` §2.8. Implementation: `crates/engine/src/invariant.rs` (not yet built).

---

## 18. Probe harness

Probes are scripted smoke tests — spawn a fixed set of agents, step N ticks, assert expected event sequence or view value. The harness runs them as ordinary `cargo test` cases.

```rust
pub struct Probe {
    pub name:    &'static str,
    pub seed:    u64,
    pub spawn:   fn(&mut SimState),
    pub ticks:   u32,
    pub assert:  fn(&SimState, &EventRing) -> Result<(), String>,
}

pub fn run_probe(p: &Probe) -> Result<(), String>;
```

The DSL `probe` declaration is compiled to a `Probe` struct + a `#[probe]` proc-macro (the proc-macro lives in the compiler crate; the engine only defines the runner). When a DSL file adds a probe, the compiler emits a `#[test] fn probe_<name>()` that calls `run_probe`.

Probes are deterministic by construction: same seed + same spawn + same tick count → same outcome. A flaky probe is a determinism bug.

See `../dsl/spec.md` §2.9. Implementation: `crates/engine/src/probe.rs` (not yet built).

---

## 19. Telemetry sink

Telemetry is a single `TelemetrySink` trait implementation plugged into the tick loop:

```rust
pub trait TelemetrySink: Send + Sync {
    fn emit(&self, metric: &'static str, value: f64, tags: &[(&'static str, &'static str)]);
    fn emit_histogram(&self, metric: &'static str, value: f64);
    fn emit_counter(&self, metric: &'static str, delta: i64);
}
```

Built-in metrics (emitted every tick by the engine):

| Metric                | Type      | Meaning                                  |
|-----------------------|-----------|------------------------------------------|
| `engine.tick_ms`      | histogram | Wall-clock time for the tick             |
| `engine.event_count`  | counter   | Events pushed this tick                  |
| `engine.agent_alive`  | gauge     | Alive agents                             |
| `engine.cascade_iterations` | histogram | Cascade-loop iterations until fixed point |
| `engine.mask_true_frac` | gauge  | Fraction of mask bits set                |

Domain metrics (emitted by compiler-generated code) flow through the same sink; naming convention is `domain.<declaration>` (e.g. `domain.view_mood_us`).

In-tree implementations: `NullSink` (discards), `VecSink` (collects in memory for tests), `FileSink` (appends JSON lines to a file). Production sinks (Prometheus, StatsD) live in downstream crates.

See `../dsl/spec.md` §2.11. Implementation: `crates/engine/src/telemetry.rs` (not yet built).

---

## 20. Schema hash

The engine computes a SHA-256 hash over a canonical string listing layout-relevant types and variants. The hash is baselined in `crates/engine/.schema_hash` and checked by a CI guard.

Coverage:

- SoA hot/cold field layout (name, type, order)
- `Event` variants and field signatures
- `MicroKind` variants (ordered)
- `MacroKind` variants (ordered)
- `MovementMode` variants
- `CreatureType` variants (engine's universal subset; compiler extends via stable ordinal)
- `CommunicationChannel` variants
- `TARGET_SLOTS` constant
- Trajectory tensor keys and shapes
- Invariant names and failure modes (whose presence affects determinism via rollback)

Bump triggers: adding, removing, or reordering any variant above; changing any SoA field type; changing `TARGET_SLOTS`; changing `MAX_CASCADE_ITERATIONS`; changing RNG hard-coded keys. Adding a new chronicle event variant is NOT a bump (chronicle doesn't affect the replayable hash).

The CI test compares the computed hash against the baseline. Mismatch is a hard failure with a message listing the schema string so the author can re-baseline deliberately.

See the existing implementation at `crates/engine/src/schema_hash.rs`; baseline at `crates/engine/.schema_hash`.

---

## 21. Observation packing

The observation packer builds a feature tensor from `SimState` for policy input. Layout is `[n_agents × feature_dim]` f32.

```rust
pub struct ObsPacker {
    feature_sources: Vec<Box<dyn FeatureSource>>,
    feature_dim: usize,
}

pub trait FeatureSource: Send + Sync {
    fn dim(&self) -> usize;
    fn pack(&self, state: &SimState, agent: AgentId, out: &mut [f32]);
}

impl ObsPacker {
    pub fn pack_batch(&self, state: &SimState, out: &mut [f32]);  // [n × feature_dim] row-major
}
```

Built-in feature sources:

- `VitalsSource` — hp_frac, hunger_frac, rest_frac (dim 3)
- `PositionSource` — `pos.x`, `pos.y`, `pos.z`, `movement_mode_one_hot` (dim 7)
- `NeighborSource<K>` — top-K nearest agents, each contributing relative position + hp + group_rel (dim `K × 8`)
- `CooldownSource<N>` — per-ability cooldown remaining (dim N)

Domain features (emotion, needs, personality, cumulative stats) register via compiler-generated `FeatureSource` impls.

**Determinism.** `pack_batch` writes in agent-id order; parallel implementations chunk by agent slot and join in order. No sorting-by-value is allowed (that would couple output to float associativity).

See `../dsl/spec.md` §3.1 (grammar) and Appendix B.1 (budget). Implementation: `crates/engine/src/obs.rs` (not yet built).

---

## 22. Debug & trace runtime

Six components make the simulation introspectable. All are on-demand — none run in the hot path of a production tick.

### 22.1 `trace_mask(agent, action_idx, tick)`

Explains why a mask bit is set or unset. The compiler emits two artefacts per mask predicate:

1. A fast boolean kernel (used in production to fill `MaskBuffer`).
2. An explanation kernel — same AST, but each sub-clause captures its inputs and result.

At debug time, `trace_mask` re-runs the explanation kernel against a captured observation snapshot for `(agent, tick)` and returns:

```rust
pub struct MaskTrace {
    pub agent: AgentId,
    pub tick: u32,
    pub action_idx: usize,
    pub ast: Vec<AstNode>,
}

pub struct AstNode {
    pub node_id: u32,
    pub expr: &'static str,
    pub inputs: Vec<(&'static str, serde_json::Value)>,
    pub result: bool,
}
```

The first `result=false` node in a conjunction is the failing clause; its inputs show why (e.g. `distance(self, t) = 84 > AGGRO_RANGE = 50`).

Observation snapshots are captured when the caller enables `ObsSnapshotRing` on `SimScratch` — an N-tick rolling buffer of the packed observation tensor, per agent. At 200K agents × ~1.6KB × 1000 ticks = 320 GB, this is debug-build only; prod builds do not capture unless explicitly enabled.

### 22.2 `causal_tree(root_event_id, tick)`

Every event carries `cause: Option<EventId>` with `EventId = (tick: u32, seq: u32)`. Root events (raw agent actions, scheduled physics) have `cause = None`. `causal_tree` walks the DAG edge-set from a root and returns its transitive closure: an event tree showing the cascade fan-out.

```rust
pub fn causal_tree(events: &EventRing, root: EventId) -> CausalTree;

pub struct CausalTree {
    pub root: EventId,
    pub children: FxHashMap<EventId, Vec<EventId>>,
}
```

Chronicle entries (`Event::ChronicleEntry`) sit at leaves in the tree — they're emitted by cascade rules but don't trigger further cascades.

**Retention caveat.** If the `cause` event has been evicted from the ring buffer, the chain is truncated. Tools render "truncated — root was at tick T, outside retention" and the partial tree.

### 22.3 `tick_stepper`

A debug driver that halts the tick between phases (§12) and exposes each phase's input/output via a `TickDebugHandle`:

```rust
pub fn tick_stepper<B: PolicyBackend>(
    state:   &mut SimState,
    scratch: &mut SimScratch,
    events:  &mut EventRing,
    backend: &B,
    until:   StepStage,
) -> TickDebugHandle;

pub enum StepStage {
    AfterMask, AfterPolicy, AfterShuffle, AfterApply, AfterViews, AfterInvariants
}

pub struct TickDebugHandle<'a> {
    pub obs:        &'a [f32],       // [n × feat_dim]
    pub mask:       &'a MaskBuffer,
    pub actions:    &'a [Action],
    pub emitted:    &'a [Event],
    pub view_deltas: Vec<(&'a str, serde_json::Value)>,
}
```

A phase is pure-functionally re-runnable — its output is a deterministic function of its input. Re-running the backend with a different temperature, for example, is a supported debug primitive.

**RNG stream alignment.** The sampler RNG is seeded from `per_agent_u32(seed, agent, tick, b"sample")` — stage-boundary-addressable. Re-running a phase does NOT advance any global RNG state; re-sampling with the same seed gives the same action.

### 22.4 `tick_profile`

Flamegraph-style scope tracing. The engine wraps each named kernel with `scope_begin(sym) / scope_end(sym)`; each call appends a `(sym, start_ns, dur_ns, tick)` tuple to a flat trace buffer.

```rust
pub struct TickProfile {
    pub samples: Vec<ScopeSample>,
}

pub struct ScopeSample {
    pub sym:   &'static str,
    pub start_ns: u64,
    pub dur_ns: u64,
    pub tick: u32,
}

pub fn with_profile<T>(f: impl FnOnce() -> T) -> (T, TickProfile);
```

Built-in scopes: `engine::mask_build`, `engine::policy_eval`, `engine::shuffle`, `engine::apply`, `engine::cascade::<event_kind>`, `engine::view_fold::<view_name>`, `engine::invariant::<name>`. Compiler-emitted code adds `domain::<declaration>::eval`.

Profiling builds compile declarations with `#[inline(never)]` to preserve scope boundaries. Release builds strip scope instrumentation via `#[cfg(feature = "profile")]`.

### 22.5 `agent_history(id, t_from..t_to)`

Rolling per-agent decision ring. Each agent's last N decisions are buffered: `(tick, action, chosen_score, margin_over_second)`. Query primitives:

```rust
pub fn agent_history(id: AgentId, range: Range<u32>) -> Vec<DecisionRecord>;
pub fn actions_by_kind(id: AgentId, kind: MicroKind) -> Vec<DecisionRecord>;
pub fn never_chose(id: AgentId, kind: MicroKind) -> bool;
pub fn pattern_search(id: AgentId, pat: &[MicroKind]) -> Vec<u32>;  // tick offsets
```

Buffer size per agent is compile-time configurable (`AGENT_HISTORY_TICKS = 500` by default). At 200K agents × 500 × 64 B = 6.4 GB — prod builds opt in via feature flag; default is a small "focus list" (e.g. 50 agents).

### 22.6 `snapshot(tick) → ReproBundle`

Captures everything needed to reproduce a single tick offline:

```rust
pub struct ReproBundle {
    pub schema_hash:  [u8; 32],
    pub state_before: Vec<u8>,        // SimState serialized (§16)
    pub events:       Vec<Event>,     // events emitted during this tick
    pub decisions:    Vec<Action>,    // actions chosen this tick
    pub seed:         u64,
    pub tick:         u32,
}

pub fn snapshot(state: &SimState, events: &EventRing, tick: u32) -> ReproBundle;
```

A repro bundle is a self-contained test case. `reproduce(bundle)` loads it, runs one tick, and checks that the re-emitted events match `bundle.events` byte-for-byte. Schema-hash mismatch is a hard error — the bundle is tied to the DSL version that produced it.

---

Implementation status: the trace runtime is fully designed in `../dsl/stories.md` §§34–40 but not yet implemented. It lands after the engine MVP primitives; see the engine build-out plan for sequencing.

See `../dsl/stories.md` §34 (trace_mask), §35 (causal_tree), §37 (tick_stepper), §40 (tick_profile), §39 (agent_history), §38 (snapshot).

---

## 23. Non-goals

The engine does not:

- **Parse DSL source text.** That's the compiler (`../compiler/spec.md` §1).
- **Generate Rust / SPIR-V code.** That's the compiler.
- **Desugar verbs.** `verb` declarations lower to mask + cascade + reward at compile time.
- **Know which items exist in a world, which abilities exist, which group kinds exist.** Compiler-generated code registers these on engine primitives.
- **Render chronicle prose.** `Event::ChronicleEntry.text` is template-rendered by compiler-generated code before being pushed; the engine treats the text as an opaque string.
- **Schedule curricula.** A curriculum is an external pipeline that decides which scenarios run in what order; the engine provides the per-scenario runner.
- **Implement GPU compute kernels.** `voxel_engine::compute::GpuHarness` lives downstream; the engine exposes traits that downstream crates implement for GPU execution.
- **Implement the neural policy forward pass.** `NeuralBackend` is trait-only in the engine. The weights loader + forward pass + autograd belong in a separate crate (`crates/nn`) that the compiler may target.
- **Handle save-game migrations across domain-type evolution.** Migration function composition is supported (§16); writing the migration logic is a domain concern.
- **Enforce real-time budgets.** The engine aims for ≤ 2 s / 100-agent × 1000-tick in release, but the compiler and domain code can violate this with hot-path allocations. The `dhat-heap` test enforces steady-state zero-alloc at the engine layer only.

---

## Implementation map

The engine spec above names all the primitives. Implementation status as of 2026-04-19:

| Section                         | Module                               | MVP? | Notes |
|---------------------------------|--------------------------------------|------|-------|
| §3 State model                  | `state/`                             | ✅   | Agents only; aggregates Phase 2 |
| §4 Event log                    | `event/`                             | ✅   | 4 variants; compiler extends |
| §5 Spatial index                | `spatial.rs`                         | ✅   | |
| §6 RNG streams                  | `rng.rs`                             | ✅   | Keyed constants baked |
| §7 MicroKind                    | `mask.rs`, `policy/`                 | ⚠️   | 4 of 18 variants; full set Phase 2 |
| §8 MacroKind                    | —                                    | ❌   | Phase 2 |
| §9 Cascade runtime              | —                                    | ❌   | Phase 2 |
| §10 Mask buffer                 | `mask.rs`                            | ✅   | 2 heads built in; extensible API Phase 2 |
| §11 Policy backend              | `policy/`                            | ✅   | `UtilityBackend` in-tree; Neural is trait-stub |
| §12 Tick pipeline               | `step.rs`                            | ✅   | 3 of 6 phases (mask / policy / apply); full Phase 2 |
| §13 Views                       | `view/`                              | ⚠️   | MaterializedView only; Lazy + TopK Phase 2 |
| §14 Aggregates                  | —                                    | ❌   | Phase 2 |
| §15 Trajectory                  | `trajectory.rs`                      | ✅   | Built-in tensors; register_tensor Phase 2 |
| §16 Save/load                   | —                                    | ❌   | Phase 2 |
| §17 Invariants                  | —                                    | ❌   | Phase 2 |
| §18 Probes                      | —                                    | ❌   | Phase 2 |
| §19 Telemetry                   | —                                    | ❌   | Phase 2 |
| §20 Schema hash                 | `schema_hash.rs`                     | ✅   | |
| §21 Observation packing         | —                                    | ❌   | Phase 2 |
| §22 Debug & trace               | —                                    | ❌   | Phase 3 (depends on invariant + probe) |

The engine build-out plan (`../superpowers/plans/`) sequences the ❌ and ⚠️ entries into a fresh TDD-driven plan that brings every section to ✅.

---

## References

- `../dsl/spec.md` — language reference (grammar, type system, worked example, settled decisions)
- `../compiler/spec.md` — compiler contract (compilation targets, schema emission, lowering passes, decisions)
- `../dsl/stories.md` — per-batch user-story investigations (the trace runtime design lives here)
- `../dsl/decisions.md` — per-decision rationale log
- `crates/engine/src/` — Rust implementation
- `crates/engine/tests/` — acceptance + regression + determinism tests
- `crates/engine/benches/` — throughput baselines
