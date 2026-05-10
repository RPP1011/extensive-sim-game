# Plan G G3 — Threats Materialised View (Design)

> Status: design-only. Implementation deferred — the analysis below
> identifies a new view-shape variant + a new dispatch shape as the
> minimum-viable surface, both of which are too large to slip into G3
> without staging.
>
> Worktree: `agent-af301e455b901c8a8`, branch
> `plan-g-g3-threats-view-design` (cut from main `68682ecd`).

## Goal

Per Plan G (`docs/superpowers/plans/2026-05-09-cast-state-and-threat-zones.md`,
"Threats materialised view" + "AI consumption pattern"), give per-agent
scoring rows a primitive `threats.in_zone(self) -> bool` (and friends) so
the dodger fixture can avoid an in-flight Firebolt cast before it
resolves.

The threats view is the missing piece between Plan G's existing
machinery (`busy_*` SoA columns populated by `EffectCastBeginApplied`
consumers — already landed) and Plan G's behavioural pins (`dodger_avoids_*`
in firebolt_probe_runtime — not yet authored).

## What the view stores per agent

A bounded ring of active threat zones, populated each tick by the fold,
queried per-agent at scoring time:

```text
struct ThreatZoneCell {
    // Discriminator. Today: 0 = inactive (sentinel), 1 = circle,
    // 2 = line. Encodes the projected telegraph shape — pinned to a
    // closed-set u8 enum so emit can branch with a `switch`.
    zone_kind:        u8,

    // Centre of the projection (origin for circle, midpoint for line),
    // in q8 fixed-point world coordinates. i16 each ⇒ ±128 world
    // units (matches the q8 packing in EffectCastBeginApplied's
    // target_x_q8 / target_y_q8 pair).
    center_x_q8:      i16,
    center_y_q8:      i16,

    // Radius (circle) or half-length (line), q8.
    radius_q8:        u16,

    // Optional secondary axis — line direction unit vector, q8. Zero
    // for circle zones.
    dir_x_q8:         i16,
    dir_y_q8:         i16,

    // Tick at which this threat resolves (caster's busy_until_tick).
    // Reads in scoring filter rows that already passed: zone is live
    // iff `world.tick < expires_at_tick`.
    expires_at_tick:  u32,

    // The casting agent. Lets a future "intensity" calc weight by
    // who's casting (fights involve teams; threats from enemies of
    // self should rank higher than threats from allies).
    source:           u32,
}
```

Total: 1 + 2 + 2 + 2 + 2 + 2 + 4 + 4 = 19 bytes ⇒ pad to 20 (or 24 for
align(4)). Per-agent ring of K=4 cells = 80 bytes/agent ⇒ 16 MB at
N=200K — feasible for the same VRAM budget that already accommodates
the existing per-agent SoA family.

A view of struct payload is itself new (every existing view stores a
single scalar — `f32` standing, `u32` belief bitset, etc.). See
[Gap analysis](#gap-analysis) below.

## How the fold dispatch works

Three logical phases, each a separate compute op:

1. **Per-agent ring reset.** `DispatchShape::PerAgent` op zeroes every
   threat-zone cell at the start of the threats fold (same shape as
   `ComputeOpKind::ViewDecay`'s per-agent kernel — single thread per
   slot, atomic store). This is necessary because the source data
   (`busy_with_ability_id`) is overwritten in-place each tick; we want
   "snapshot of currently-active threats", not "history".

2. **Project busy ⇒ threat zones.** For each (observer_agent,
   busy_caster) pair, test:
   - caster has `busy_with_ability_id != 0`
   - caster's ability metadata declares a `telegraph` shape (a future
     ability-registry column)
   - observer is within the caster's projected zone

   If all true, append a `ThreatZoneCell` to the observer's ring (mod K
   on the cursor — same primitive `PerEntityRing` already documents).

   This is an O(N×B) op (N = total agents, B = busy agents). Today's
   `DispatchShape::PerPair { source: AllAgents }` is structurally
   correct (one thread per (a, b) pair; each thread tests + appends),
   but **PerPair is currently a mask/scoring-only dispatch shape** —
   ViewFold's lowering hard-codes `DispatchShape::PerEvent { source_ring }`
   (see `crates/dsl_compiler/src/cg/lower/view.rs:462-463`).

3. **Per-event slot decay (optional).** As busy timers tick down, ring
   cells expire automatically — scoring filters check
   `world.tick < expires_at_tick`. No GC needed; the next reset (phase
   1) drops them.

The fold body's `self += <expr>` form is wrong shape — we need
`self.ring[self.cursor].kind = circle; self.ring[self.cursor].center =
...; cursor = (cursor + 1) % K`. That's a multi-statement
ring-append primitive, not a scalar accumulate.

## Scoring primitives the AI calls

Minimum viable surface (implements `dodger_avoids_*`):

```text
threats.in_zone(self) -> bool          # any live cell where self.pos in zone
threats.intensity_at(self.pos) -> f32  # sum of (radius - distance) over live cells
```

Stretch (implements perpendicular-move score):

```text
threats.nearest(self) -> AgentId       # source of closest live zone
threats.dir_away_from_nearest(self) -> Vec3
```

All four lower to a per-agent walk over the agent's threat-zone ring
(K=4 cells, unrolled). The walk produces an aggregate (any / sum /
argmin / vec) — the same shape `view::damage_dealt(self)` lowers to,
extended with a per-cell unrolled body.

## Gap analysis

The existing view-fold infrastructure does NOT cleanly cover this:

| Concern | Existing infra | Gap |
|---|---|---|
| Storage shape | `PairMap`, `PerEntityTopK`, `SymmetricPairTopK`, `PerEntityRing { k }`, `LazyCached` | `PerEntityRing` exists in IR but is **unused by any `.sim` file today** and **the WGSL emit (`emit/kernel.rs:2186`) explicitly TODOs the ring-append primitive**. No GPU-side test exercises the path. |
| Cell payload | Scalar `f32` / `u32` | Need struct payload (8 fields, 20 bytes). View signatures (`view_signatures: HashMap<ViewId, (params, result_ty)>`) carry a single `CgTy` for the result; multi-field cell payloads need a `CgTy::Struct` (or a per-view named layout shipped through the storage emit). |
| Fold body | `self += <expr>` and `self \|= <expr>` (single op) | Need ring-append: `self.ring[cursor].field = X` × N + `cursor = (cursor+1) % K`. The view-fold lowerer rejects `IrStmt::Let`, `IrStmt::If`, and any non-`SelfUpdate` shape (`view.rs:604-655`). |
| Dispatch shape | `PerEvent { source_ring }` (hard-coded for ViewFold) | Need `PerPair { source: AllAgents }` or a new `PerAgentSpatialFilter` shape. The PerPair shape exists for masks/scoring but not for views; lighting it up means changing `lower_one_handler` and the WGSL fold-kernel emit (`emit/kernel.rs::build_view_fold_wgsl_body`). |
| Source-of-fold | An event ring (the ViewFold dispatches per event entry) | The threats fold dispatches per (observer, busy-source-agent) pair — the source is per-agent SoA, not an event ring. The ViewFold `on_event: EventKindId` field is a category mismatch. |

In short: every dimension of the existing view-fold infra (storage,
payload, body, dispatch, source) needs an extension to express the
threats fold cleanly. This is a new shape, not a new view.

## Estimated work breakdown

A future iteration that picks this up should land it in slices of
roughly equal size:

1. **G3a — `PerEntityRing` end-to-end smoke test.** Author a fixture
   (`assets/sim/per_entity_ring_probe.sim`) that uses `@per_entity_ring(K=4)`
   on a scalar-payload view (e.g. ring of recent damage amounts).
   Force the WGSL emit to grow the ring-append primitive (`atomicAdd`
   on `Cursors` slot + indexed write to `Primary`). Closes the
   "PerEntityRing exists in IR but no fixture exercises it" gap.

2. **G3b — Multi-field view-storage payload.** Extend `ViewSignature`
   (or add a sibling `ViewLayout` table) to carry a struct layout for
   the cell payload. Update `synthesize_resident_context` so
   `view_storage_<name>: wgpu::Buffer` is sized for the struct (not
   just `4 * agent_cap * k`). New `CgStmt::AssignField { target:
   ViewStorage{..}, field_idx, value }` lowering for per-field
   assignment.

3. **G3c — Multi-statement fold body.** Lift the
   `view::lower_stmt`'s "only `SelfUpdate +=` allowed" gate to admit
   `Let` (for cursor/index locals) + a sequence of field-assigns. Keep
   `If` deferred — the threats projection can express its predicate at
   the dispatch level (skip-emit cells where the spatial test fails).

4. **G3d — `DispatchShape::PerAgentEventScan` (new variant).** A new
   shape: dispatch per (observer_agent, source_agent) pair where
   the source candidate set is "all agents with `busy_with_ability_id
   != 0`". Could in principle reuse `DispatchShape::PerPair { source:
   AllAgents }`, but PerPair today is mask/scoring-only and routes
   through verb expansion — surfacing it for views would entail
   threading a `(observer, candidate)` pair binding scope into
   `lower_view::lower_stmt` (which currently only knows about
   event-pattern bindings). A dedicated shape variant keeps the
   semantics narrow and the verb-mask shape unchanged.

5. **G3e — Telegraph metadata column.** Add an
   `ability_registry.telegraph_kind: u8` (and shape params) column.
   The threats fold reads `(busy_with_ability_id ⇒ telegraph_kind)` to
   project the right zone. Could begin life as a hard-coded
   per-fixture lookup (e.g. firebolt is always a circle of radius 4)
   and generalize once a second telegraph-bearing ability lands.

6. **G3f — `threats.*` scoring primitives (Builtin variants).** Add
   four new `Builtin` enum variants in `dsl_ast::ir`:
   `ThreatsInZone`, `ThreatsIntensityAt`, `ThreatsNearest`,
   `ThreatsDirAwayFromNearest`. Lower each to a CG-level read +
   per-cell unrolled walk over the per-agent threat-zone ring. (The
   `view::<name>` namespace path won't reach naturally because these
   accept a non-view base.)

7. **G3g — Threats view declaration.** Author the `view threats(self:
   Agent)` declaration in a new `assets/sim/threats_view_probe.sim`
   fixture. Exercises the full chain: PerEntityRing storage + struct
   payload + multi-statement fold body + PerAgentEventScan dispatch +
   the four scoring primitives.

8. **G3h — Smoke test.** `dsl_compiler/tests/threats_view_lower.rs`
   asserting the view's kernels emit and the scoring primitives lower.
   No GPU runtime crate yet — that's G4's runtime drive.

## Why design-only

The four-dimensional gap (storage shape, payload shape, body shape,
dispatch shape, source shape) makes G3 too large to implement in a
single slice without speculative architecture decisions on every axis.
Picking a wrong shape on any one axis cascades — e.g. authoring a
struct-payload `PerEntityRing` and discovering at G3d that the
PerAgentEventScan dispatch wants a different layout means a redo.

The smaller, lower-risk path is the slice list above (G3a → G3h),
where each closes a gap independently and tests it on a runnable
fixture before the next slice depends on it. G3a alone (PerEntityRing
end-to-end smoke) is a clean follow-up that lights up an existing-but-
unused IR variant.

## Pointers (for the next iteration)

- `crates/dsl_compiler/src/cg/lower/view.rs:411-481` — `lower_one_handler`,
  the lowering entry point that needs to grow `DispatchShape::PerAgentEventScan`.
- `crates/dsl_compiler/src/cg/lower/view.rs:530-655` — `lower_stmt`,
  which currently rejects everything except `SelfUpdate +=`/`|=`.
- `crates/dsl_compiler/src/cg/data_handle.rs:655-689` — `ViewStorageSlot`
  enum, where multi-field-cell payloads need new slots (or a struct
  representation).
- `crates/dsl_compiler/src/cg/dispatch.rs:111-168` — `DispatchShape`,
  where the new `PerAgentEventScan` variant lands.
- `crates/dsl_compiler/src/cg/emit/kernel.rs:2186` — the explicit
  TODO marker confirming PerEntityRing's WGSL emit is unimplemented.
- `assets/sim/firebolt_probe.sim` — the existing G2.5 fixture with
  `EffectCastBeginApplied` consumers writing the busy SoA family.
  G3's threats view reads from the SAME columns this fixture
  populates.
