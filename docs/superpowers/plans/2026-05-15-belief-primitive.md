# Belief Primitive — generalized partial-knowledge surface (Plan I)

> Goal: introduce a first-class `belief <name>(observer: Agent[, key]) -> T { ... }` DSL declaration that subsumes the four bespoke "per-agent partial knowledge" instances already shipping (threats view, ToM `beliefs_flags`, dungeon_horde's hand-rolled `hero_known_rooms`, the threat_stresstest grudge-decay design). Picks storage shape from the signature, reuses the existing `@materialized` + `@dispatch(per_agent_event_scan)` + `@decay` machinery, adds one new social-propagation op kind so beliefs can copy from agent to agent (the AllyDied / Conversed gossip pattern). Three existing belief instances become 10-15 line declarations; new belief systems (resource awareness, market intel, reputation, room knowledge) become ~10 lines instead of bespoke .sim + host plumbing + per-fixture runtime tests.

## Goal

The threats system shipped this session (commits `70a6634d` → `6da7fe7e`) is one instance of a recurring pattern across the codebase:

| Instance | Storage | Key | Propagation | Decay |
|---|---|---|---|---|
| `view threats(observer: Agent) -> f32` | `@per_entity_ring(K=4)` struct cells | none / pos | `EffectCastBeginApplied` | `expires_at_tick` per cell |
| `view beliefs_flags(observer: Agent, subject: Agent) -> u32` | `pair_map` | subject | `BeliefUpdate` events | none (sticky bits) |
| `viewer_runtime::ViewerApp::hero_known_rooms: [u64; 5]` | host-side bitmap | room_idx | doorway peek + ally-death broadcast | none |
| Plan H stresstest `grudges` (designed, not built) | per_entity_ring | rival_idx | `BargainBroken` / `OutbidSimulated` | slow decay |

Each instance reimplements the storage hint, the propagation rule, the decay pass, and the query helper from scratch. The threats-side query Builtin (`threats.intensity_at`, `threats.nearest`, `threats.dir_away_from_nearest`) duplicates ~3 prelude helpers that all walk the same per-observer ring. The `beliefs_flags` ToM view has the same dispatch shape as threats but a different storage hint and a different query path. The hero_known_rooms bitmap is hand-rolled in viewer_runtime/src because no .sim primitive yet exposes "per-agent bool per slot."

**This plan adds one DSL declaration form** — `belief <name>(observer: Agent[, key]) -> T { initial / on / merge from / decay / clamp }` — that lowers to whichever existing storage shape fits the signature. Storage hint is **inferred from the signature**, not author-picked. Propagation reuses the existing fold-handler shape. Decay reuses `@decay`. The one new piece is a **social-merge op kind** so beliefs can be copied from one agent to another on event (e.g. `merge from d: bit_or` on `AllyDied { dead: d }` — receiver bitwise-ORs the dying ally's bitmap into their own).

Once `belief` is a recognized declaration, the four bespoke instances above migrate to ~10-line declarations. Adding new belief systems (resource awareness, market intel, per-enemy patrol knowledge, reputation/grudges) becomes ~10 lines each instead of ~4-6 hours of bespoke .sim + host + runtime test work per instance.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/dsl_ast/src/ir.rs:924` — `pub enum ViewKind { Lazy, Materialized(StorageHint) }`. Add `Belief(BeliefSpec)` variant.
  - `crates/dsl_ast/src/ir.rs:863-870` — `pub struct ViewIR { name, params, return_ty, body, annotations, kind, decay, ... }`. The belief declaration reuses 100% of this struct; `kind` discriminates.
  - `crates/dsl_ast/src/parser.rs:194` — `Some("view") => view_decl(c, annotations, start).map(Decl::View)`. Add a parallel `Some("belief") => belief_decl(...)` arm; the body parser at line 955 (`parse_view_body`) is reused verbatim.
  - `crates/dsl_compiler/src/cg/lower/view.rs:229` — `pub fn lower_view`. Add a `Belief` arm parallel to `Materialized`. Storage-hint inference replaces the author-picked hint.
  - `crates/dsl_compiler/src/cg/lower/view.rs:490` — `lower_one_handler`. Each `on <Event>` handler in a belief lowers via this existing path unchanged.
  - `crates/dsl_compiler/src/cg/program.rs:404` — `pub storage_hint: Option<CgStorageHint>`. Belief signatures auto-populate this from the (param-types, result-type) → storage-shape inference table.
  - `crates/dsl_compiler/src/cg/program.rs:507-526` — `ViewLayout` for struct cells. Beliefs whose return type is a struct (with an `expires_at_tick` field) reuse this exact layout machinery — same as the threats fixture (`assets/sim/threats_struct_probe.sim:97-167`).
  - `crates/dsl_compiler/src/cg/op.rs::ComputeOpKind` — add `BeliefSocialMerge { belief, from_field, op }` variant.
  - `crates/dsl_compiler/src/cg/lower/expr.rs:2253-2306` — existing `Builtin::Threats*` lowerings. Belief queries lower via the same `BuiltinId::ViewCall { view }` + helper-emit pattern (`compose_view_storage_prelude` in `cg/emit/program.rs:1075`); a new sibling `compose_belief_query_prelude` emits per-belief helpers.
  - `crates/dsl_compiler/src/cg/emit/kernel.rs:855-885` — body-scan that adds per-view `view_storage_<view>_primary` BGL bindings. Reused for belief storage (different binding name prefix `belief_<name>_storage`).
  - `assets/sim/threats_struct_probe.sim` — concrete reference for storage-hint inference + struct-cell + propagation. The post-Plan-H form (`6da7fe7e`) shows every cell field reading real per-caster data via `agents.pos(source_candidate)` / `agents.busy_until_tick(source_candidate)` / `abilities.telegraph_*` — that pattern carries over to the belief-primitive emit unchanged.
  - `crates/viewer_runtime/src/lib.rs:279-282` — `pub hero_known_rooms: [u64; dungeon::N_HEROES as usize]`. Hand-rolled bitmap that the new `belief room_known(observer: Agent, room: u32) -> bool` declaration replaces.

  Search method: `rg`, `grep -nE`, direct `Read`.

- **Decision:** **extend existing**. The view declaration shape, the materialized-fold lowering, the dispatch shapes (`per_agent_event_scan`), the decay primitive, and the storage hints (`pair_map`, `per_entity_ring`) all already exist. The belief primitive is a thin syntactic + lowering wrapper that picks the right combination based on the signature. ONE new piece is genuinely additive: `BeliefSocialMerge` op kind + a per-shape merge body emit (bitmap word-OR, scalar max/replace) for the `merge from <agent>: <op>` clause.

- **Rule-compiler touchpoints:**
  - DSL inputs edited (in-tree migrations): `assets/sim/dungeon_horde.sim` migrates `hero_known_rooms` from host bitmap to `belief room_known(observer: Agent, room: u32) -> bool`. `assets/sim/dungeon_stealth.sim` + `assets/sim/dungeon_horde.sim` simplify `view beliefs_flags` to `belief detected_subject(observer: Agent, subject: Agent) -> bool`. `assets/sim/threats_struct_probe.sim` swaps `view threats` for `belief active_threat(observer: Agent, source: Agent) -> ThreatCell`.
  - Generated outputs re-emitted: every per-runtime `OUT_DIR/<fixture>/generated.rs` regens. Per-runtime build.rs auto-regenerates. Schema hash bump (see P2).
  - New parser keyword: `belief`. Reserved in `crates/dsl_ast/src/parser.rs`.

- **Hand-written downstream code:** NONE for the engine-side primitive — all lowerings use the existing emit infrastructure plus the new `BeliefSocialMerge` op kind which lives in `crates/dsl_compiler/src/cg/emit/`. Migration of `crates/viewer_runtime/src/lib.rs::hero_known_rooms` to read from the new belief storage requires deleting the host-side bitmap + replacing the propagation in `advance_hero_exploration` with a chronicle event the belief consumes — that's host-side host-readable code, not a sim handler, so it's a legitimate viewer-runtime change. Plan E hook bypass via `RUNTIME_EDIT_JUSTIFIED=1` for the migration step (justified as compiler-driven cleanup of code the new primitive subsumes).

- **Constitution check:**
  - **P1 (Compiler-First Engine Extension):** PASS. The belief primitive lowers entirely through existing emit paths plus the new `BeliefSocialMerge` op kind. No `impl Rule` in `crates/engine/src/handlers/`. Evidence: `crates/dsl_compiler/src/cg/lower/view.rs:229-700` is the existing path that absorbs the belief variant; the new op kind lives in `crates/dsl_compiler/src/cg/op.rs` (compiler crate, not engine).
  - **P2 (Schema-Hash on Layout):** REQUIRES BUMP. Adding the `BeliefSocialMerge` op kind + the `Belief` variant of `ViewKind` are both shape-relevant for snapshot/replay (a belief view's storage layout is part of the persistent fixture state). One regen of `crates/engine/.schema_hash` per the documented procedure.
  - **P3 (Cross-Backend Parity):** PASS. Bitmap bit-OR via `atomicOr` is commutative + associative + bit-exact (P11-trivial). Scalar belief writes use the existing CAS+add path the threats view already exercises. Social-merge op kind dispatches via `PerAgentEventScan` over `(receiver, giver)` pairs — same shape `view beliefs_flags` already uses, with cross-backend parity covered by the existing `tests/parity_*.rs` suite extended with `tests/parity_belief_social_merge.rs`.
  - **P4 (`EffectOp` Size Budget):** N/A. No new `EffectOp` variants — beliefs are storage views, not actions.
  - **P5 (Determinism via Keyed PCG):** PASS. No new RNG draws. Propagation is event-driven (already deterministic). Social-merge iterates over a deterministic event ring.
  - **P6 (Events Are the Mutation Channel):** PASS. Belief storage mutations come exclusively from `on <Event>` handlers (existing fold path) and the social-merge op (new but event-triggered). No direct field writes outside the documented kernel API.
  - **P7 (Replayability Flagged):** PASS. Beliefs inherit the `replayable` flag from the events they consume. Decay is deterministic per-tick — replayable by construction.
  - **P8 (AIS Required):** PASS — this section satisfies it.
  - **P9 (Tasks Close With Verified Commit):** PASS — each slice below is sized for one commit with verifiable tests; `closes_commit` UDA per the project-DAG skill.
  - **P10 (No Runtime Panic):** PASS. Belief-read helpers return typed defaults on out-of-bounds (false / 0 / sentinel) — same pattern as the existing `Builtin::Threats*` sentinel-fallback at `crates/dsl_compiler/src/cg/lower/expr.rs:2289-2306`.
  - **P11 (Reduction Determinism):** PASS. Bitmap OR is commutative + associative; scalar max/min are commutative + associative; `replace` is idempotent under per-event ordering (the chronicle's `seq` field disambiguates). Social-merge op kind generates `atomicOr` / `atomicMax` calls that compile to bit-exact GPU ops on both backends.

- **Runtime gate:**
  - `crates/dsl_compiler/tests/belief_primitive_lower.rs` — author a `belief room_known(observer: Agent, room: u32) -> bool` declaration in a synthetic fixture, lower it, assert: (a) storage hint inferred as bit-packed pair map, (b) propagation handler lowered to a `ComputeOpKind::ViewFold`, (c) `BuiltinId::ViewCall` registered for `belief.room_known`, (d) WGSL helper `belief_room_known_read(observer, room) -> bool` emits with the bitmap-extract body.
  - `crates/sims/tests/belief_room_known_pin.rs` — drive `dungeon_horde` (post-migration) one tick, seed an observer at slot 0's centroid, step, assert `belief.room_known(observer, 0)` returns true and `belief.room_known(observer, 5)` returns false. Then trigger an `AllyDied` event near the observer with the dying ally having visited slot 5; assert observer's belief now returns true for slot 5 (social-merge).
  - `crates/sims/tests/parity_belief_social_merge.rs` — run a small fixture that exercises the social-merge op kind on both `SerialBackend` and `GpuBackend`, assert byte-equal storage post-tick.
  - `crates/sims/tests/dungeon_horde_pin.rs` — existing pin must remain green after the `hero_known_rooms` migration. Heroes' fog-of-war floor tinting in the viewer must still respond to per-hero knowledge (visual, not asserted by the pin but checked manually + in `crates/viewer_runtime/tests/diag_hero_movement.rs`).

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

## Implementation slices

Sized for one commit per slice. Each closes with a verified commit (P9).

### I.1 — Parser + AST recognition for `belief` keyword

- `crates/dsl_ast/src/parser.rs:194` — add `Some("belief") => belief_decl(c, annotations, start).map(Decl::View)`. The body parser `parse_view_body` is reused verbatim. The `belief_decl` helper validates the signature shape (first param must be `Agent`, return type must be `bool` / `u8` / `u32` / `f32` / a registered struct) and rejects unsupported forms with a clear error.
- `crates/dsl_ast/src/ir.rs:924` — extend `ViewKind`: `Belief(BeliefSpec)` variant. `BeliefSpec` carries the propagation handlers + social-merges + decay + clamp + the (later-inferred) storage hint slot.
- New AST node `SocialMergeHandler { pattern, source_agent_local: LocalRef, op: MergeOp }`. Parser recognizes `merge from <ident>: <op_name>` clause.
- Tests: `crates/dsl_ast/tests/belief_parse.rs` — round-trip the four migration target declarations through parse + display, assert AST shape.

**Closes commit:** `feat(dsl_ast): belief declaration parser + AST shape`

### I.2 — Resolver bindings + signature validation

- `crates/dsl_ast/src/resolve.rs:1443` — extend the `Decl::View` resolver arm to handle `ViewKind::Belief`. Same scope binding as materialized views (observer in scope; key params in scope; event-pattern binders in fold handlers). Social-merge handlers get an extra binding: `from_agent` is the AgentId from the named event field.
- Validation: signature must match one of the supported shapes (table in storage-hint inference below). Unsupported signatures (e.g. `(observer: Agent) -> Vec3`, multi-key non-Agent first param) surface as typed `ResolveError::UnsupportedBeliefSignature`.
- Tests: `crates/dsl_ast/tests/belief_resolve.rs` — assert signature-shape validation rejects unsupported forms with the documented error variant.

**Closes commit:** `feat(dsl_ast): belief resolver + signature validation`

### I.3 — CG lowering: storage-hint inference

- `crates/dsl_compiler/src/cg/lower/view.rs:229` — add `Belief` arm. Inference table:

  | Signature | `CgStorageHint` | Cell encoding |
  |---|---|---|
  | `(Agent) -> bool` | `SingleKey` | bit-packed: `agent_cap / 32` u32s |
  | `(Agent) -> <scalar f32/u32>` | `SingleKey` | dense `agent_cap` of scalar |
  | `(Agent, u32) -> bool` w/ explicit `key_cap` annotation | `PairMap` | bit-packed: `agent_cap × key_cap / 32` u32s |
  | `(Agent, u32) -> <scalar>` w/ explicit `key_cap` | `PairMap` | `agent_cap × key_cap` of scalar |
  | `(Agent, Agent) -> u32` | `PairMap` | existing `beliefs_flags` shape |
  | `(Agent) -> <Struct{expires_at_tick: u32, ...}>` | `PerEntityRing { k }` | existing threats shape |

  The `key_cap` annotation is `@key_cap(<n>)` on the belief decl; defaults to 32 for u32 keys. Struct beliefs inherit the existing `register_view_layout` path.

- `crates/dsl_compiler/src/cg/program.rs:404` — auto-populate `storage_hint` from inference instead of from author-picked annotation.
- Per-handler propagation lowering: each `on <Event>` lowers to `ComputeOpKind::ViewFold` via the existing `lower_one_handler` at line 490. Belief's social-merge handlers lower to a NEW `ComputeOpKind::BeliefSocialMerge { belief, from_field, op }` whose dispatch shape is `PerAgentEventScan { source: AllAgents }` and whose body iterates over the storage word-by-word doing atomic merges.
- Tests: `crates/dsl_compiler/tests/belief_lowering_shape_matrix.rs` — six declarations covering each signature shape; assert the inferred storage hint, op kinds emitted, and binding declarations match expected.

**Closes commit:** `feat(dsl_compiler): belief storage-hint inference + lowering`

### I.4 — WGSL emit: per-shape helper + social-merge op body

- `crates/dsl_compiler/src/cg/emit/program.rs:1075` — add `compose_belief_query_prelude(body, prog)` sibling of `compose_view_storage_prelude`. For each `belief_<name>_read(` substring detected in the kernel body, emit the per-shape helper. Bitmap shape:

  ```wgsl
  fn belief_room_known_read(observer: u32, room: u32) -> bool {
      let bit_idx = observer * <key_cap>u + room;
      let word = belief_room_known_storage[bit_idx >> 5u];
      return (word & (1u << (bit_idx & 31u))) != 0u;
  }
  ```

  Scalar pair_map shape mirrors the existing `view_<id>_get` helper. Struct-ring shape reuses the threats `view_<id>_get(observer)` ring-walk verbatim.

- `crates/dsl_compiler/src/cg/emit/kernel.rs:855-885` — extend body-scan to detect `belief_<name>_storage` references and append the matching BGL binding (per-belief storage buffer). Mirrors the existing `view_storage_<view>_primary` body-scan.
- `crates/dsl_compiler/src/cg/emit/wgsl_body.rs::ability_registry_column_token` shape — add `belief_<name>_storage` token convention. New `BuiltinId::BeliefRead { belief: BeliefId }` mirrors `BuiltinId::ViewCall`.
- `BeliefSocialMerge` op kind WGSL emit: dispatch over `(receiver, giver)` pairs filtered by the where-clause; body is a `for (var w: u32 = 0u; w < <words_per_observer>u; w = w + 1u) { atomicOr(&belief_X_storage[receiver * words + w], belief_X_storage[giver * words + w]); }` for bitmap shape, or scalar atomic max/replace for non-bitmap shapes.
- Tests: `crates/dsl_compiler/tests/belief_emit_helpers.rs` — assert each shape's helper substring + body-scan-induced binding emission. Negative arms: unreferenced helpers don't emit (mirrors the threats prelude tests).

**Closes commit:** `feat(dsl_compiler): belief WGSL emit + social-merge op body`

### I.5 — Runtime gate: smoke fixture + behavioural pin

- New fixture `assets/sim/belief_smoke_probe.sim` — minimal belief declaration of each shape (5 declarations, one per row of the inference table), each with one propagation handler and one social-merge handler. No combat — just belief reads and writes.
- `crates/sims/tests/belief_smoke_pin.rs` — drives the fixture for 10 ticks, asserts: (a) initial state matches `initial:` clause, (b) propagation handler updates the cell on event, (c) social-merge OR'd a peer's bitmap into the receiver's, (d) decay reduced a scalar belief by the expected amount over 10 ticks.
- `crates/sims/tests/parity_belief_social_merge.rs` — run the social-merge fixture on both `SerialBackend` and `GpuBackend`, assert byte-equal storage post-tick.
- Schema-hash regen committed in this slice if tests pass.

**Closes commit:** `feat(sims): belief primitive smoke fixture + parity pin`

### I.6 — Migrate dungeon_horde's `hero_known_rooms` to `belief room_known`

- `assets/sim/dungeon_horde.sim` — add `belief room_known(observer: Agent, room: u32) -> bool { ... }` declaration. Replaces the host-side bitmap. Social-merge handler on `AllyDied` does the existing party-wide gossip.
- `crates/viewer_runtime/src/lib.rs` — delete `hero_known_rooms` field + propagation logic in `advance_hero_exploration`. Read from the GPU-side belief storage when computing fog-of-war floor tints. (Plan E hook bypass via `RUNTIME_EDIT_JUSTIFIED=1`; justification: compiler-driven cleanup of code the new primitive subsumes.)
- Existing `crates/sims/tests/dungeon_horde_pin.rs` must remain green.
- Existing `crates/viewer_runtime/tests/diag_hero_movement.rs` must remain green.

**Closes commit:** `refactor(dungeon_horde): migrate hero_known_rooms to belief primitive`

### I.7 — Migrate `view beliefs_flags` to `belief detected_subject`

- `assets/sim/dungeon_horde.sim` + `assets/sim/dungeon_stealth.sim` + `assets/sim/dungeon_layout.sim` + `assets/sim/tom_probe.sim` (every fixture using `view beliefs_flags`).
- The bit-OR fold becomes a `belief detected_subject(observer: Agent, subject: Agent) -> bool`. The existing `pair_map` storage shape is preserved by inference. The propagation handler (`on BeliefUpdate ...`) carries over verbatim.
- All fixtures' tests must remain green.

**Closes commit:** `refactor(dsl): migrate beliefs_flags to belief primitive across fixtures`

### I.8 — Migrate `view threats` to `belief active_threat`

- `assets/sim/threats_struct_probe.sim` — replace `view threats` with `belief active_threat(observer: Agent, source: Agent) -> ThreatCell { ... }`. Reuses the existing 8-field struct cell layout.
- `assets/sim/dodger_probe.sim` + `assets/sim/threats_view_probe.sim` + `assets/sim/threats_with_decay_probe.sim` + `assets/sim/threat_stresstest.sim`.
- Existing threats Builtins (`threats.intensity_at` / `nearest` / `dir_away_from_nearest`) become aliases for `belief.active_threat.intensity_at` / etc., or get auto-rewritten by the lowering. Decision in slice — if alias is feasible, ship it; otherwise update the dodger fixture's score expression.
- Existing threats pins (`threat_stresstest_pin.rs`, `threats_struct_probe_pos_keyed_pin.rs`, `dodger_probe`'s behavioural pin) must remain green.

**Closes commit:** `refactor(threats): migrate to belief primitive`

## Verification end-to-end

After all 8 slices land:

1. `cargo build --release` — clean.
2. `cargo test -p dsl_ast --release` — parse + resolve tests for the new keyword; existing tests unaffected.
3. `cargo test -p dsl_compiler --release` — lowering + emit shape tests for the new primitive; existing tests unaffected.
4. `cargo test -p engine --test schema_hash --release` — confirms the bumped baseline matches the regen.
5. `RUST_MIN_STACK=33554432 cargo test -p sims --release` — every existing pin remains green; new belief_smoke_pin + parity_belief_social_merge pin pass.
6. `cargo test -p viewer_runtime --release --test diag_hero_movement` — heroes still move (path > 30 budget).
7. Workspace-wide `cargo test --workspace --release` — no fixture-runtime regression.
8. Manual: launch viewer, verify fog-of-war tinting still responds to per-hero knowledge (visual confirmation of slice I.6).

## Estimated work

- I.1 (parser): 1-2h
- I.2 (resolver): 1h
- I.3 (lowering): 2-3h
- I.4 (emit): 2-3h
- I.5 (smoke fixture + parity): 1-2h
- I.6-I.8 (3 migrations): 1h each = 3h

Total: ~10-14 hours across 8 commits. Slice-able across multiple sessions; each slice ships independently.

## What this enables next (out of scope for this plan)

- **Resource awareness** belief: per-NPC knowledge of where gold/food/items dropped. ~10 lines.
- **Market intel** belief: per-trader knowledge of prices at known markets. Decays as world prices drift. ~10 lines.
- **Per-enemy patrol knowledge**: enemies route only through rooms they've patrolled. ~10 lines + add enemy-side `EnemyMoved` event consumer.
- **Reputation/grudges**: per-agent belief about another agent's standing. The `threat_stresstest` design's "long-term grudge" half. ~10 lines.
- **Fog-of-war confidence decay**: heroes' room knowledge could decay (forget unseen rooms over time). One-line addition to existing belief decl.

Each of these would be a 1-2 hour slice on top of the belief primitive, vs the current ~4-6 hour bespoke implementation per system.
