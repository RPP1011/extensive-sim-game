# Engine Plan — Agent-State Catalogue Port (2026-04-19)

> **Status:** ✅ executed. This file documents what was done rather than
> prescribing what to do — the 2026-04-19 audit
> (`docs/audit_2026-04-19.md`) found that `SimState` covered ~20% of what
> `docs/dsl/state.md` committed to. Rather than grow the schema one field
> per future plan (and bump the schema hash each time), we ported the full
> agent catalogue in one branch, behind storage-only accessors. Subsequent
> plans (masks, cascade handlers, save/load, ability effects) read these
> fields without re-bumping.

## Goal

Make the engine's agent-state SoA a superset of `docs/dsl/state.md`'s
top-level agent catalogue — every field the spec names has a hot or cold
Vec-per-slot slot with a default, an accessor, a mutator, and a bulk slice.
No behaviour; subsequent plans wire it in.

## Rationale

- **Stabilise the schema hash.** Every plan that added SoA fields
  (Plan 3.5 engagement, Ability Plan 1 combat) bumped the baseline.
  Landing the full catalogue once amortises the bump.
- **Unblock plan authors.** Future plans can assume `state.agent_safety(id)`
  exists and simply attach semantics, rather than arguing field shape in
  review.
- **Resolve the 6-dim needs drift.** Per audit §3a-b the engine had 3
  needs (hunger/thirst/rest_timer) and state.md committed to 6
  (Maslow-inspired). Rather than pick one, we **add 5 psychological + keep
  3 physiological = 8 needs total**. Plan 1 Eat/Drink/Rest keeps working;
  future Maslow-driven aspiration reads the psychological bank.

## Deltas

### Hot fields added (28 new)

| Group | Field | Type | Default | Source §state.md |
|---|---|---|---|---|
| Spatial | `level` | u32 | 1 | §18-24 |
| Spatial | `move_speed` | f32 | 1.0 | §27-34 |
| Spatial | `move_speed_mult` | f32 | 1.0 | §27-34 |
| Combat | `shield_hp` | f32 | 0.0 | §39-48 |
| Combat | `armor` | f32 | 0.0 | §39-48 |
| Combat | `magic_resist` | f32 | 0.0 | §39-48 |
| Combat | `attack_damage` | f32 | 10.0 (matches step.rs constant) | §39-48 |
| Combat | `attack_range` | f32 | 2.0 (matches `ATTACK_RANGE`) | §39-48 |
| Combat | `mana` | f32 | 0.0 | §39-48 |
| Combat | `max_mana` | f32 | 0.0 | §39-48 |
| Psych-Needs | `safety` | f32 | 1.0 | §89-101 |
| Psych-Needs | `shelter` | f32 | 1.0 | §89-101 |
| Psych-Needs | `social` | f32 | 1.0 | §89-101 |
| Psych-Needs | `purpose` | f32 | 1.0 | §89-101 |
| Psych-Needs | `esteem` | f32 | 1.0 | §89-101 |
| Personality | `risk_tolerance` | f32 | 0.5 | §107-117 |
| Personality | `social_drive` | f32 | 0.5 | §107-117 |
| Personality | `ambition` | f32 | 0.5 | §107-117 |
| Personality | `altruism` | f32 | 0.5 | §107-117 (state.md calls it `compassion`; engine uses `altruism`) |
| Personality | `curiosity` | f32 | 0.5 | §107-117 |

### Cold fields added (9 new)

| Field | Type | Default | Source |
|---|---|---|---|
| `grid_id` | `Option<u32>` | `None` | §Physical State |
| `local_pos` | `Option<Vec3>` | `None` | §Physical State |
| `move_target` | `Option<Vec3>` | `None` | §Physical State |
| `status_effects` | `SmallVec<[StatusEffect; 8]>` | empty | §StatusEffect |
| `memberships` | `SmallVec<[Membership; 4]>` | empty | §Membership |
| `inventory` | `Inventory` | default | §Inventory |
| `memory` | `SmallVec<[MemoryEvent; 64]>` | empty | §Memory |
| `relationships` | `SmallVec<[Relationship; 8]>` | empty | §Relationship |
| `class_definitions` | `[ClassSlot; 4]` | zeroed | §Skill & Class |
| `creditor_ledger` | `SmallVec<[Creditor; 16]>` | empty | §Economic |
| `mentor_lineage` | `[Option<AgentId>; 8]` | all-None | §Relationships |

### Supporting types added (in `state/agent_types.rs`)

- `StatusEffect { kind: StatusEffectKind, source: AgentId, remaining_ticks: u32, payload_q8: i16 }`
- `StatusEffectKind` — `Stun/Slow/Root/Silence/Dot/Hot/Buff/Debuff` (u8 repr, 0..=7).
- `Membership { group: GroupId, role: GroupRole, joined_tick: u32, standing_q8: i16 }`
- `GroupRole` — `Member/Officer/Leader/Founder/Apprentice/Outcast` (u8 repr, 0..=5).
- `Inventory { gold: u32, commodities: [u16; 8] }`
- `MemoryEvent { source: AgentId, kind: u8, payload: u64, confidence_q8: u8, tick: u32 }`
- `Relationship { other: AgentId, valence_q8: i16, tenure_ticks: u32 }`
- `ClassSlot { class_tag: u32, level: u8 }`
- `Creditor { creditor: AgentId, amount: u32 }`
- `MentorLink { mentor: AgentId, discipline: u8 }` (defined, not yet stored —
  `mentor_lineage` currently holds bare `Option<AgentId>` per state.md;
  `MentorLink` is reserved for a future plan that tags lineage with
  discipline).

### Capabilities extension (`src/creature.rs`)

- `LanguageId(NonZeroU16)` — niche-optimised language id. Two built-in
  constants: `COMMON` (raw=1) and `DRACONIC` (raw=2). The full catalogue
  is a later plan.
- `Capabilities.languages: SmallVec<[LanguageId; 4]>`. Humans default to
  `[COMMON]`; Dragons to `[DRACONIC]`; Wolves/Deer empty.
- Existing fields on `Capabilities` (`can_fly`/`can_build`/`can_trade`/
  `can_climb`/`can_tunnel`/`can_marry`/`max_spouses`) were already
  present — Task F pinned them with a test so a refactor can't silently
  erase them.

## Task breakdown (executed)

Each task: failing test → fields + accessors + setters + bulk slice →
green test → clippy clean → commit. Spawn defaults + kill reset covered
per task.

- **A. Spatial extras** — `tests/state_spatial_extras.rs`.
- **B. Combat extras** — `tests/state_combat_extras.rs`.
- **C. Status effects** + stub types — `tests/state_status_effects.rs`,
  new `src/state/agent_types.rs`.
- **D. Psychological needs** (5 hot f32) — `tests/state_psych_needs.rs`.
  Doc comment on `SimState` documents the 3+5 split.
- **E. Personality** (5 hot f32) — `tests/state_personality.rs`. Uses
  `altruism` for what state.md calls `compassion`.
- **F. Capabilities.languages + LanguageId** —
  `tests/state_capabilities.rs`. Pins pre-existing Capabilities fields.
- **G. Memberships** + `GroupRole` — `tests/state_memberships.rs`.
- **H. Inventory stub** — `tests/state_inventory.rs`.
- **I. Memory stub** — `tests/state_memory.rs`. 64-slot inline (state.md
  caps at 20; ring semantics/eviction deferred).
- **J. Relationships stub** — `tests/state_relationships.rs`. 8-slot
  inline (state.md caps at 20).
- **K. Misc cold** (ClassSlot / Creditor / MentorLink) —
  `tests/state_misc_cold.rs`.
- **L. Schema hash re-baseline + docs** — this file, `status.md` flip,
  `state.md` 8-needs engine note.

## Schema hash delta

Before: `bbe4a3a2afe3618aa2d1c474fc0234437400af78f997c235f587819902ae6564`
After:  `d4c2c4b98f70ec3352c3ca4145fad6bd0d5390c85cd946db0614c3c3fa288008`

The fingerprint string now encodes every hot/cold field name + type and
every new stub type's layout, so a future rename or type change will
trigger a failure and force a conscious re-baseline.

## Test count delta

Before: 169 green (release). After: 210 green — +41 tests across 11 new
test files. Each group added 3–6 tests pinning default, round-trip,
bulk-slice length, and (for collections) kill-respawn clearing.

## Deferred fields

These state.md fields are **not** ported in this plan because they depend
on subsystems we don't have yet. They are tracked here for future plans.

- **AgentData fields (~50 of ~60)** — state.md §AgentData lists name,
  adventurer_id, economic (gold/debt/creditor_id/income_rate/
  credit_history/economic_intent/price_knowledge/trade_route_id/
  trade_history), location (home_settlement_id, home_building_id,
  work_building_id, inside_building_id, current_room, home_den),
  pathfinding cache (cached_path, path_index, goal_stack), work state
  machine (work_state, action, behavior_production), class system
  (class_tags, classes, behavior_profile), relationships
  (relationships, spouse_id, children, parents, apprentice_of,
  apprentices, pack_leader_id), perception (known_resources,
  known_voxel_resources, harvest_target), emotions (6d: joy, anger,
  fear, grief, pride, anxiety), aspiration, action_outcomes,
  price_beliefs, cultural_bias, campaign fields (morale/stress/fatigue/
  loyalty/injury/resolve/archetype/party_id/faction_id/mood/fears/deeds/
  guild_relationship), current_intention, equipment, equipped_items,
  passive_effects, world_abilities. Each either (a) duplicates a field
  we already have in a different shape (e.g. `gold` in
  `Inventory.gold`), (b) depends on subsystems not built (roomref,
  voxel system, class/ability system, quest pool, trade routes), or
  (c) is emergent (aspiration, action_outcomes). The port chose the
  top-level catalogue (things `state.md` lists at the same visual
  nesting depth as `id`/`pos`/`hp`) — the AgentData sub-catalogue is a
  separate future plan.
- **Emotions (6d transient)** — Plan deferred; not in the top-level
  catalogue (state.md lists them under §Emotions as a sub-structure
  carried by AgentData, not Agent). Ports with AgentData.
- **Aspiration** — same reason; emergent field recomputed every 500
  ticks. Needs the event-bus to be wired first.
- **Goal/GoalStack** — depends on goal-eval system. Storage shape is
  clear (Vec<Goal>) but the fields reference entity ids to buildings/
  quests we haven't pooled.
- **BuildingData / Room / Settlement / RegionState** — state.md §World /
  §Aggregate. Not agent state; needs the Aggregate-pool `AggregatePool<T>`
  instance data (shipped as Pod shapes in Plan 1 but no instance
  storage).
- **TileRef / RoomRef** — depends on the terrain + interiors system.
  `local_pos` is a `Vec3` stub, but the full `Option<RoomRef>` wait on
  a room-id type.
- **`known_voxel_resources`** — depends on voxel subsystem; not in
  engine crate today.
- **`believed_knowledge: Bitset<32>`** — depends on the knowledge-domain
  bitset plan (§theory-of-mind).

## Design friction

1. **Personality compassion vs altruism.** state.md names the fifth
   personality trait `compassion`; the plan prompt said `altruism`. We
   followed the prompt — the two names refer to the same
   helping/empathy trait; renaming later is a one-line find/replace.
   Flagged in the `Capabilities::for_creature` comment.
2. **Memory ring size.** state.md caps at 20 events; `SmallVec<[...; 64]>`
   was used for inline headroom. True ring-with-eviction semantics
   arrive with the memory plan; push-only today means the test can
   over-push without hitting the cap. Noted in the `cold_memory` field
   comment.
3. **MentorLink used vs not.** state.md `mentor_lineage: Vec<u32>` is a
   flat chain of entity IDs. `MentorLink { mentor, discipline }` is
   defined for a richer future shape but currently unused —
   `cold_mentor_lineage` holds `[Option<AgentId>; 8]` to match state.md.
4. **Inventory.gold u32 vs f32.** state.md §AgentData uses `f32`;
   Ability Plan 1 draft uses `i64` for future debt. MVP goes with `u32`
   — unsigned integer matches the "no negative balance without a
   creditor entry" model (separate `Creditor` ledger for debt). Future
   plans can widen to u64 or switch sign; accessors keep source
   compatibility.
5. **LanguageId const-unwrap warning.** The Rust 1.93 clippy lint
   `useless_nonzero_new_unchecked` forced us to use
   `NonZeroU16::new(1).unwrap()` in const context instead of
   `new_unchecked`. Safer and keeps the niche optimisation.
6. **Kill-then-respawn test for cold collections.** Slot reuse via the
   freelist meant every `SmallVec`-backed cold field needed an explicit
   `.clear()` in both `spawn_agent` and `kill_agent` (well, in
   `spawn_agent` because it overrides the slot). Tests under Task G, I
   pin this — without the clear, a re-spawned agent would inherit the
   previous tenant's memberships / memory / etc., a silent-correctness
   landmine.
7. **Repetition burden for hot f32 accessors.** Each of 10+ hot f32
   groups has near-identical accessor / setter / slice plumbing. A
   macro would cut ~300 lines but obscure grep-ability; we chose the
   explicit form to match the existing MVP style. A future refactor
   plan can collapse to a declarative macro if the pattern holds.

## Acceptance

- `cargo test -p engine --release` = **210 green** (up from 169).
- `cargo clippy -p engine --all-targets --release -- -D warnings` clean.
- `.schema_hash` baseline bumped and test passes.
- `status.md` row flipped from ⚠️ partial coverage to ✅ 🎯 full catalogue.
- `state.md` §Needs carries a one-line engine-note pointing at this plan.
- No existing test regressed (state_agent / state_needs / determinism
  suites all pass).

## What lands next

- **Plan 3.5 + Ability Plan 1 merge** (audit Proposal 1). Both plans
  wanted to add combat-adjacent SoA fields. The fields are now here as
  zero-default storage — merged plan adds only the semantics (mask
  gates, cast handlers, opportunity-attack cascade, engagement tracking).
  One schema bump → done for this phase.
- **Plan 3 persistence**. Snapshot writer's field list is now stable
  and covers the full `state.md` top-level catalogue.
