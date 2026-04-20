# World Sim ECS DSL — Design Docs

Working folder for the ECS DSL design effort. Six canonical docs after the 2026-04-19 consolidation pass. Status: settled on surface; implementation pending.

Companion trees:

- `docs/compiler/` — compiler contract (codegen, lowering, schema emission). Split out of `docs/dsl/` on 2026-04-19.
- `docs/engine/` — runtime contract (pools, determinism, event ring, policy trait, tick pipeline, debug & trace runtime). 23-section spec at `docs/engine/spec.md`, settled 2026-04-19. (Engine spec retains ML-era terminology; pass 3 will reconcile.)

## Docs (reading order)

1. **`spec.md`** (~1450 lines) — Canonical language specification. Authoritative. §§1–10 cover the core spec (language overview, top-level declarations, action vocabulary, schema versioning, type system, compilation targets, runtime semantics, worked example, settled decisions, non-goals). Appendix A reprints the universal-mechanisms treatise (PostQuest / AcceptQuest / Bid / Announce). ML concerns (policy architecture, training, reward shaping, observation packing) are out of scope — see `spec.md` §10 and `docs/compiler/spec.md` §3 for the Python emission path.
2. **`state.md`** (~1650 lines) — Unified field-level state catalog across three domains (Agent state, Aggregate state, World state). Implementation reference: which struct holds which field, who updates it, who reads it.
3. **`systems.md`** (~3310 lines) — Essential-vs-emergent classification of all ~160 simulation systems across five batches (Economic, Social/Inner/Needs, Combat/Quests/Threat, Politics/Faction/Meta, World/Terrain/Low-level). Per-system schema: Classification / Reframe / Required NPC actions / Required derived views / Required event types / One-line summary.
4. **`stories.md`** (~4640 lines) — Per-batch user-story investigations (59 stories, prefixed by batch letter: AB / C / D / E / FGIJ / H). Each story traces observation → policy → action → cascade → outcome; gaps in the spec surface here first.
5. **`decisions.md`** — Standing decision log. Per-decision rationale for the 29 settled items referenced by `spec.md` §9. (To be populated — see iteration agenda.)
6. **`README.md`** — this file.

## Cross-reference convention

- `spec.md §N` — section N of the specification (e.g., `spec.md §9` = Settled Decisions).
- `stories.md §X.N` — story N in batch X (e.g., `stories.md §E.30` = Story 30 in Batch E: Runtime / Sim Engineer).
- `state.md#agent-state` / `state.md#aggregate-state` / `state.md#world-state` — domain-scoped catalog lookups.
- `systems.md#batch-N-...` — batch-scoped system inventory.

## Where we are

### Settled (major commitments)

- **Strict event-sourcing rubric.** State mutations are events; current state is a fold over events + entity baseline. ~19 of ~160 systems are truly essential physics; the rest are emergent NPC actions, derived views, or dead.
- **Universal action vocabulary.** Four macro mechanisms (PostQuest / AcceptQuest / Bid / Announce) + 18 micro primitives (including Communicate / Read / Remember). Spec `spec.md` §3.2–§3.3.
- **Utility backend is the permanent production NPC backend.** `scoring` declarations drive per-action utility scoring; masks gate candidates. ML is out of DSL scope — compiler emits Python dataclasses + pytorch `Dataset` over the trace format for external training.
- **Role power = mask + cascade**, not a smarter scorer. Leaders and commoners use the same scoring machinery; masks + downstream cascades differentiate impact.
- **Three entity types: Agent + Item + Group** (+ optional Projectile). Buildings and resources are derived views over world tiles + voxels + harvest-event history.
- **3D positions (`vec3`)** with movement-mode sidecar (`Walk | Climb | Fly | Swim | Fall`). Slopes are Walk; volumetric and transitioning agents route through the sidecar.
- **Source-tagged information** with theory-of-mind. `MemoryEvent` carries `source: Source` and `confidence: f32`. `Communicate` / `Announce` / `Ask` / `Read` are first-class primitives. Mask predicates gate information-sensitive actions. `believed_knowledge` carries per-bit volatility (Short / Medium / Long half-lives).
- **Multi-group memberships + emergent standings.** Per-pair `Relationship` + `Group.standings` drive hostility, alliance, vassalage — all from the universal `SetStanding(target, kind)` macro.
- **No-escrow quest rewards.** Rewards compute at completion from current state; poster reliability is emergent reputation (`QuestDefault` event for underpayment). Accepters price risk via `believed_intrinsic_value`.

### Settled in the 2026-04-19 interview pass (29 items, referenced from `spec.md` §9)

Action / quest mechanics, runtime / infrastructure, schema / memory, modding — full detail in `spec.md` §9 (and forthcoming `decisions.md`). Headline items:

- `Resolution` enum adds `Coalition{min_parties}` + `Majority`; macro head runs every tick.
- Hybrid push (Announce cascade) + pull (GatherInformation) quest discovery.
- K=12 spatial-view caps; role-scaled non-spatial view caps; eager cross-entity indices (standing, quest eligibility, same-building).
- Multi-quest membership (K=4) with scoring-level exclusion.
- `spouse_ids: SortedVec<AgentId, 4>` (polygamy); multi-parent `ChildBorn`; `AuctionKind::Service` inverts payment direction.
- Per-agent RNG via `hash(world_seed, agent_id, tick, purpose)` — no stored per-agent state.
- Materialized-view restoration via schema-hash guard + rebuild fallback; N=500 snapshot cadence + zstd event-log compression.
- Chronicle prose: eager templates + async LLM rewrite for flagged categories; saved prose canonical across template changes.
- Training is out of DSL scope — compiler emits Python dataclasses + pytorch `Dataset` over the trace format; training code lives in external pytorch scripts.
- Utility backend is the permanent production NPC backend.
- Spatial index: 2D columns + per-column z-sort + `movement_mode ≠ Walk` sidecar.
- Overhear confidence: category base (SameFloor / DiffFloor / Outdoor) × exp(−distance / OVERHEAR_RANGE).
- `believed_knowledge` 3-tier volatility with per-bit refresh + negative-evidence clearing.
- `Document` drops stored `trust_score`; gains `author_id` + `seal`; readers derive confidence from relationship + seal.
- `FactPayload` materialized inline on Quest/Document creation.
- Mod handlers: `on_event(Kind) in lane(Validation | Effect | Reaction | Audit)`; lexicographic mod-id ordering within lane.

### Open (next-iteration deliverables)

1. **`decisions.md`** — extract per-decision rationale from `spec.md` §9 into a standing decision log. §9 stays as an at-a-glance summary.
2. **Auction implementation.** Concrete state machine matching the `Resolution` enum; `AuctionItem`, `BidPlaced`, `AuctionResolved` events; per-world-config cadence per `AuctionKind`.
3. **`Reward` / `Payment` / `PartyScope` / `StandingKind` enum finalization.** Concrete serialization + mask-predicate implications.
4. **Cross-entity mask index implementation** — event-triggered rebuild paths for standing / quest eligibility / same-building.
5. **Materialized-view serialization format** — safetensors layout + schema-hash guard + rebuild path.
6. **GPU kernel emission details** — SPIR-V codegen for mask predicates, cascade handlers, view event-folds, and the spatial-hash kernel. Currently sketched in `../compiler/spec.md` §1.2.
7. **Grammar formalization.** Tokenizer + parser + error messages. Defer until prototype validates surface shape.

## Conventions

- `[OPEN]` tags formerly marked unresolved decisions. After the 2026-04-19 pass, none remain in `spec.md`; any future open items should reopen via new `[OPEN]` markers.
- Per-batch system entries use a uniform schema: `{ File, Classification, Reframe, Required NPC actions, Required derived views, Required event types, One-line summary }`.
- State catalog entries use `{ Type, Meaning, Updated by, Read by, Emergent? }`.
