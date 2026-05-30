# Edgeworld — Survival World Sim Design Spec

> Name: **Edgeworld**.
> Status: design, approved in principle 2026-05-29; ready for planning.
> Date: 2026-05-29.

## One-line

A **zero-player** survival world sim built entirely on the `.sim` DSL: a small band
of person-like survivor agents living inside a creature ecosystem. They forage,
eat, rest, and flee on their own; the player watches whether they make it. The
content is emergent — there is no player intervention and no win condition.

## Why this fits the engine

The DSL is, at its core, a deterministic per-agent decision loop:
each tick every agent picks its highest-scoring eligible `verb` (argmax), the
verb fires an `apply_ability` program, the program emits events, and `physics`
folds consume those events back into agent state. A survival sandbox is almost a
1:1 mapping onto that loop, and the closest existing fixtures
(`foraging_colony`, `predator_prey`, `trophic_3tier`, `ecosystem_cascade`,
`village_day_cycle`) already establish the patterns.

| Game concept | DSL mechanism |
|---|---|
| A need that grows over time | `physics` fold: `hunger += rate` per tick on a hunger SoA column |
| Deciding what to do | verb/argmax: `SeekFood` / `Eat` / `Rest` / `Flee` / `Wander` compete by `score` |
| Eating / depleting a food source | verb → `apply_ability` → event → fold that lowers hunger and depletes the node |
| Starvation / death | `hunger > max` → emit death event → `alive = false` |
| Predation | predator verb `Hunt`; prey verb `Flee`; spatial neighbor queries for perception |
| The world pushing back | food regrows slowly; a day/cold cycle adds a second pressure |

Determinism is a feature, not a constraint: same world-seed → same saga, every
run, on both backends (P3/P5/P11). The "saga" is replayable and inspectable from
the event log **and from rendered frames**.

## Design pillars (what makes it DF/RimWorld-deep, not a toy)

The cast stays small (8–20 agents) until these pillars are deep enough to
generate stories worth following. **Depth before scale** is the explicit
directive. The four axes of depth, in build order:

1. **Needs** — survival is multi-variable (hunger, then warmth/rest/safety), so
   agents make real trade-offs rather than optimizing one number.
2. **Imperfect perception** — agents act on a *private, decaying belief* about
   the world (e.g. last-known food location), not ground truth. This is the
   engine's genuine differentiator over RimWorld (whose pawns are omniscient
   about base state); it is introduced only once the survival need makes it
   matter, never bolted on for its own sake. Substrate already exists:
   `belief <name>(observer[, subject]) -> T` with `@decay` and social `merge`.
3. **Society** — food sharing, group foraging, and **gossip** (belief
   propagation between agents who meet) make cooperation-vs-collapse emergent.
4. **World** — terrain, seasons, and resource depletion drive **migration**: the
   band must move when a region is exhausted.

## Architecture

- A new DSL fixture: `assets/sim/edgeworld.sim` (entities, events, config, verbs,
  physics folds, beliefs/views).
- A new runtime crate `crates/edgeworld_runtime` with its own `build.rs` that
  compiles `edgeworld.sim` into `OUT_DIR` and an `edgeworld_app` binary that seeds
  the world, steps the sim, and **dumps render frames**. Follows the existing
  per-runtime crate pattern (`boids_runtime`, etc.).
- **No new engine SoA columns** for the prototype. Per-agent semantic state rides
  on existing repurposed columns (the `spy_network` / `village_economy`
  precedent), e.g. hunger on `hunger`, energy/rest on `mana`, a need or memory
  counter on `shield_hp` / `disguise_expires_at_tick`. If/when depth genuinely
  requires first-class columns, that is a deliberate schema-change task
  (P2 schema-hash bump), not a silent add.
- World space: continuous 2D positions (`pos: vec3`, z unused early) with
  `@spatial` neighbor queries for perception, matching `detective_investigation`
  / `foraging_colony`. Food sources are entities. Terrain/voxel grid is deferred
  to Pillar 4 / Phase 5.

## Rendering & verification (first-class from Phase 0)

The prototype must **show** that it lives, not just log it. Two render tiers:

- **Tier 1 — headless PNG dump (Phase 0).** The `edgeworld_app` binary writes a
  top-down PNG of the world every N ticks (and/or a final contact sheet):
  survivors as dots, food green, predators red, dead greyed. Deterministic,
  no GPU window required, cheap. Crucially, these PNGs are **inspectable by the
  agent directly via the Read tool**, so visual verification happens every
  iteration without human-in-the-loop. A small population-over-time plot
  (ASCII or PNG) accompanies it to make boom/bust legible at a glance.
- **Tier 2 — live voxel viewer (Phase 6).** Wire `viewer_runtime` for a
  watchable real-time render; if it serves on localhost, the `screenshot-localhost`
  skill captures stills. This is the "nice version," not the verification path.

**Verification loop:** the agent renders → Reads the frames → confirms the
dynamics look right (population curve, agents clustering on food, predators
culling) → iterates without waiting. The **human sign-off** is reserved for
milestone "this is a saga worth watching" judgments, not every tick of work.

## Entity & behavior model — Phase 0 ("It lives, and you can see it")

The smallest slice that already produces visible drama (boom/bust):

- **Entities**: `Survivor : Agent` (the band) and `FoodNode : Agent` (a plant
  food source with a depletable quantity riding on a repurposed column).
- **Need**: hunger rises every tick (`physics` fold on `Tick`).
- **Verbs** (argmax per survivor):
  - `Eat` — gated on (standing on/adjacent to a non-empty FoodNode AND hungry);
    high score. Fires `apply_ability` → event → folds lower hunger + deplete node.
  - `SeekFood` — gated on (hungry AND a FoodNode in perception range); moves
    toward nearest known food; medium score scaled by hunger.
  - `Wander` — default idle drift; low constant score.
- **Death**: a physics fold emits a `Died` event when `hunger > max`; the agent's
  `alive` flag clears.
- **Regrowth**: FoodNode quantity regenerates slowly via a per-tick fold.
- **Render**: PNG frames + population trace per the rendering section.
- **Observable**: population over time, per-agent death tick, total food
  remaining. Even this bare version yields the classic forage/deplete/crash/
  recover cycle.

This phase reuses only mechanisms already proven in shipped fixtures, so it
de-risks the engine path before any new DSL surface is exercised.

### Phase 0 success criterion

A seed that **reliably renders a legible boom/bust**: population grows or holds,
strips local food, suffers a visible crash (several deaths in a window), and
leaves a surviving remnant rather than flat-lining or instant total extinction.
Evidenced by rendered frames + the population curve. The agent verifies this
from the PNGs; the human signs off that it reads as a saga worth watching.

## Roadmap

Each phase is a watchable increment. Phase 0 is the prototype; 1+ adds depth.

- **Phase 0 — It lives, and you can see it.** Hunger + food + forage/eat/starve/
  regrow + Tier-1 PNG rendering + population trace. (slice above)
- **Phase 1 — Predators (ecosystem layer).** Add predator creatures that hunt the
  band (and, as a sub-step, huntable prey the band can eat — the "both layered"
  flavor). Survival becomes a food-vs-safety trade-off; `Flee` enters the argmax.
  Borrows `predator_prey` / `trophic_3tier`. (Day/cold cycle deferred to a later
  abiotic-pressure sub-step.)
- **Phase 2 — Imperfect memory (Pillar 2).** Survivors hold a decaying `belief`
  about last-known food (and threat) locations. Stale beliefs send them to
  depleted spots; survival now depends on whether their mental map tracks
  reality. Introduces belief-read-from-scoring (the `detective_investigation`
  gap surface) and `@decay`.
- **Phase 3 — Renewal.** Reproduction when well-fed → long-run population cycles,
  lineages, recovery-or-extinction outcomes.
- **Phase 4 — Society (Pillar 3).** Food sharing, group foraging, and gossip
  (belief `merge from … : max`) — knowledge of food/danger spreads between
  agents who meet. Cooperation vs. collapse emerges.
- **Phase 5 — A real world (Pillar 4).** Terrain, seasons, resource depletion →
  migration when a region is exhausted. Brings in the voxel/terrain surfaces.
- **Phase 6 — Watch it live (Tier 2).** Wire the voxel `viewer_runtime` for a
  real-time render of the saga.
- **Phase 7+ — Scale.** Only once depth is DF/RimWorld-grade, raise the cast from
  ~20 toward the engine's 1k–10k ceiling and shift from individual stories to
  aggregate emergence.

## Risks & known DSL gaps (to confirm during planning)

- **Belief-read inside a `score` expression** (Phase 2) is a known unproven
  surface — `detective_investigation.sim` documents the pair-keyed view-call gap
  and used a surrogate. Phase 2 may have to start with a scalar-view or
  SoA-column surrogate and treat the true belief-in-scoring path as a DSL task.
- **Agent spawning/despawning at runtime** (reproduction, Phase 3) — need to
  confirm the runtime supports population growth, not just a death/`alive=false`
  bitmap. May constrain Phase 3 to a fixed-capacity pool with respawn semantics.
- **Config is build-time baked** — per-run parameterization (food density, band
  size) currently bakes to WGSL literals; runtime override is limited (see the
  config-driven-sims gap). Acceptable for a fixture; note for tuning ergonomics.
- **SoA repurposing ceiling** — only a handful of spare columns exist; deep
  multi-need agents (Pillar 1) may force a real schema change sooner than later.
- **Render path** — confirm the cheapest deterministic way to emit a PNG from the
  runtime (a tiny image-write dep, or raw PPM/PNG encode) without pulling the GPU
  viewer into Phase 0.

## Constitution touchpoints (for the eventual plan's AIS)

- P1 — all behavior originates in the DSL; no hand-written engine rules.
- P2 — any first-class SoA need column is a schema-hash task, not a silent add.
- P3/P5/P11 — determinism & parity hold; RNG via `per_agent_u32`; reductions
  (food depletion fan-in, population folds) sort-then-fold.
- P7 — death/birth/chronicle events flagged `replayable` correctly.

## Resolved decisions

- **Name:** Edgeworld.
- **Phase 0 done:** renders a legible boom/bust the agent verifies from frames
  and the human signs off on (criterion above).
- **Phase 1 pressure:** predators first.
- **Verification posture:** agent self-verifies via rendered-frame inspection
  each iteration; does not block on human feedback except at milestone sign-offs.

## Open questions

- Tier-1 render: tiny PNG crate vs. hand-rolled PPM/PNG encode — settle in
  planning based on what keeps `edgeworld_runtime`'s deps minimal and
  deterministic.
