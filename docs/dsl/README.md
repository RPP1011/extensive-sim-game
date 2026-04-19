# World Sim ECS DSL — Design Docs

Working folder for the ECS DSL design effort. Active docs are listed in reading order below. Status: iterative — proposals will be revised; nothing here is final.

## Reading order

### Foundations (read first)

1. **`state_npc.md`** (475 lines) — every per-entity field with updater/reader, derivation graph
2. **`state_aggregate.md`** (569 lines) — settlement/faction/region/guild/quest fields
3. **`state_world.md`** (473 lines) — tiles, voxels, indices, caches, scratch buffers

### System inventory

4. **`systems_economic.md`**, **`systems_social.md`**, **`systems_combat.md`**, **`systems_politics.md`**, **`systems_world.md`** — strict-rubric reclassification of all 158 systems. Each system labeled ESSENTIAL / EMERGENT / DERIVABLE-VIEW / DEAD with required NPC actions, event types, and derived views.

### Active design

5. **`proposal_universal_mechanics.md`** — PostQuest / AcceptQuest / Bid as the universal action vocabulary. Faction-as-party. Auction state machine spec. Extended QuestType enum.
6. **`proposal_policy_schema.md`** — current policy/observation/action schema. Single neural backend, role differentiation via mask + observation features, ~1600-float observation, hierarchical macro/micro action heads.
7. **`debate_policy_schema_attacker.md`** + **`debate_policy_schema_defender.md`** — structured debate findings that informed the current proposal. Read for opposing views and the rationale behind specific design choices.

## Where we are (as of this checkpoint)

### Settled

- **Strict event-sourcing rubric.** State mutations are events; current state is a fold over events + entity baseline. ~19 of 158 systems are truly essential; the rest are emergent NPC actions, derived views, or dead.
- **Universal action vocabulary** of ~15 verbs. Three macro mechanisms (PostQuest, AcceptQuest, Bid) collapse the 110-verb sketch into structured parameter heads. Documented in `proposal_universal_mechanics.md`.
- **Single neural backend** for production. Utility scoring kept as bootstrap/baseline only. LLM is a research tool, not a per-tick path.
- **Role power = mask + cascade**, not a smarter policy. Leaders and commoners use the same model; leaders' masks pass more verbs and their actions cascade to higher-impact events.
- **Rich observation** (~1600 floats per agent) covering self atomic + contextual + spatial slots + non-spatial named-reference slots + context blocks. Combat-style ~30-feature observations would throw away the social/economic/political signal.
- **Slot-by-distance + named-reference hybrid.** Spatial slots (nearby_actors, nearby_structures, nearby_resources) handle physical encounters; `known_actors[K=10]` and `known_groups[K=6]` handle non-spatial references (group leaders, distant rivals, absent spouses).
- **Hierarchical action heads.** Macro categorical (NoOp/PostQuest/AcceptQuest/Bid/Announce/InviteToGroup/AcceptInvite) + micro categorical (Hold/Move/Attack/Communicate/Read/...) + parameter heads (target pointer, fact_ref pointer, pos_delta, magnitude, quest_type, party_scope, reward_type, payment_type, announce_audience). Mask is per-head.
- **Schema versioning is first-class.** Schema hash, `@since` annotations, append-only growth, CI guard.
- **GPU compilation is scoped.** Observation packing, mask evaluation, neural forward, event-fold views are GPU kernels. Cascade rules and quest-eligibility predicates stay CPU.
- **Three entity types: Agent + Item + Group** (+ optional Projectile).
  - `Agent` — any agentic actor: humans, wolves, dragons, goblins. Same struct, distinguished by `creature_type` tag + personality/needs config + group memberships.
  - `Item` — path-dependent owner / durability / history.
  - `Group` — universal social-collective primitive (faction, family, guild, religion, party, hunting pack, settlement, court, cabal). Holds leadership, treasury, standing relations to other groups, history.
- **Buildings and resources are derived views** over the world's spatial data. `query::structures_at(pos, radius)` walks tiles + voxels to enumerate buildings; `query::resources_in(region)` walks voxel materials + harvest event log to enumerate resource nodes. Construction progress, occupants, building_type, remaining-quantity are all derivations.
- **Inter-agent disposition is per-pair relationships + group standings.** Each agent has `memberships: Vec<Membership>` (with role per group) and per-pair `Relationship` records. `is_hostile(a, b)` is a derived view over relationship valence + group war standings. Multi-group membership produces emergent loyalty conflicts.
- **Agent positions are 3D (`vec3`).** Ground-locked `creature_type`s (Human, Elf, Dwarf, Wolf, Goblin) carry a post-movement snap constraint pinning `pos.z` to `surface_height(pos.xy)` outdoors or `floor_height(pos, building)` indoors. Volumetric `creature_type`s (Dragon, Fish, Bat) move freely in z. Distance comes in three flavors: `distance` (3D Euclidean, combat/threat), `planar_distance` (xy-only, social/audibility), `z_separation` (|Δz|, altitude gates).
- **Information is source-tagged.** `MemoryEvent` carries a `source: Source` (`Witnessed` / `TalkedWith(a)` / `Overheard(a)` / `Rumor{hops}` / `Announced(g)` / `Testimony(doc)`) and a `confidence: f32 ∈ [0, 1]` derived from source. Agents share information explicitly: `Communicate(recipient, fact_ref)` is a micro primitive; `Announce(audience, fact_ref)` joins PostQuest/AcceptQuest/Bid as a fourth macro mechanism. `Read(doc)` ingests the facts written in an `ItemKind::Document`. Mask predicates `knows_event`, `knows_agent`, `confident_about`, `recent` gate information-sensitive actions. The `chronicle` stream is dev-only — agents never read it.

### Settled in this round (2026-04-19 interview pass)

All 29 `[OPEN]` markers in `proposal_dsl_surface.md` §9 resolved. Summary — authoritative detail lives in §9 of that doc.

- **Action / quest mechanics**: `Resolution` enum adds `Coalition{min_parties}` + `Majority`; macro head runs every tick (mostly NoOp); GAE(γ=0.99, λ=0.95) with per-head γ override; hybrid push (Announce cascade) + pull (GatherInformation) discovery; K=12 spatial slots + role-scaled non-spatial slots; eager indices for standing / quest eligibility / same-building; multi-quest membership (K=4 active) with policy-level exclusion; **no-escrow completion rewards** → reliability is emergent reputation; taker-only `WithdrawQuest`, no amendment; `Payment::Combination{material, immaterial}` with per-agent `intrinsic_value` + theory-of-mind valuation; nested quest cancellation via `child_quest_ids`; `spouse_ids` polygamy + multi-parent `ChildBorn`; `AuctionKind::Service` inverts payment direction; alliance/war emergent from `SetStanding(target, kind)` standings (no first-class alliance); agent-level invites only.
- **Runtime / infrastructure**: LLM is a first-class DSL backend (not distilled into Neural); per-agent RNG derived via `hash(world_seed, agent_id, tick, purpose)`; materialized views serialize with schema-hash guard + rebuild fallback; event-log compression = subset filter + N=500 snapshot cadence + zstd; chronicle prose = eager templates + async LLM rewrite for flagged categories, saved prose canonical; probe episode counts config-definable; **training algorithms live in Python**, DSL emits pytorch-compatible trajectories; utility backend is permanent (no retirement); spatial index = 2D columns + z-sort + `movement_mode ≠ Walk` sidecar; overhear confidence = category base (`SameFloor/DiffFloor/Outdoor`) × exponential distance decay; believed_knowledge bits carry 3-tier volatility (Short=500 / Medium=20k / Long=1M ticks half-life) with per-bit refresh timestamps + negative-evidence clearing.
- **Schema / memory**: `Document` drops stored `trust_score`, gains `author_id` + `seal`; reader derives confidence from relationship + seal; `FactPayload` materialized inline on Quest/Document creation (ring eviction independent).
- **Modding**: `on_event(Kind) in lane(Validation|Effect|Reaction|Audit)`; within-lane ordering is lexicographic mod-id; destructive overrides happen via forking the DSL source.

### Open (next-iteration deliverables)

1. **`prototype_plan.md`** — MVP validation scope: observation packing + utility backend for small N, neural bootstrap on utility trajectories, measured decision quality vs current `action_eval` path. Go/no-go checkpoints before full DSL implementation.
2. **Auction implementation.** Real state machine matching the `Resolution` enum; `AuctionItem`, `BidPlaced`/`AuctionResolved` events; per-world-config cadence per `AuctionKind`.
3. **`Reward` / `Payment` / `PartyScope` / `StandingKind` enum finalization.** §3.3 lists variants; concrete serialization + mask-predicate implications still to spell out.
4. **Cross-entity mask index implementation** (standing, quest eligibility, same_building) — event-triggered rebuild paths.
5. **Materialized-view serialization format** (safetensors layout, schema-hash guard check, rebuild path).
6. **GPU kernel emission details** — SPIR-V codegen from DSL for observation packing, mask evaluation, neural forward, mask-patched sampling. Currently sketched in §6.2; needs concrete lowering rules.
7. **Grammar formalization.** Pseudocode in §3–§7 isn't a grammar; tokenizer + parser + error-message design still needed. Defer until the prototype validates the surface.
8. **Schema migration tooling.** Matters once we have a model checkpoint to carry forward.

## Iteration agenda

1. **`prototype_plan.md`** — validation blueprint for the observation+utility path (the first real check against the DSL surface).
2. **Decisions log (`docs/dsl/decisions.md`)** — move the per-question rationale out of §9 into a standing decision log; §9 stays as a reference summary.
3. **Synthesis doc** — cross-batch consolidation of all system docs into master vocabularies + ESSENTIAL-kernel inventory.
4. **Grammar sketch** — only after prototype validates surface shape.

## How to use this folder

- **For design discussion**: read this README + `proposal_policy_schema.md` + `proposal_universal_mechanics.md`. Skim the debate briefs for opposing views.
- **For implementation reference**: the `state_*.md` and `systems_*.md` docs are the field-level inventory. They tell you what data exists and which systems own writes/reads.
- **For DSL surface design**: defer until the policy schema stabilizes. Pseudocode in the current proposal is suggestive only.
- **For prototyping**: see `prototype_plan.md` once written.

## Conventions

- `[OPEN]` tags in proposal docs mark unresolved decisions. (None currently — §9 of `proposal_dsl_surface.md` is fully settled.)
- Per-batch system docs use a uniform schema: per-system { File, Classification, Reframe, Required NPC actions, Required derived views, Required event types, One-line summary }.
- State catalogs use a uniform schema: per-field { Type, Meaning, Updated by, Read by, Emergent? }.
