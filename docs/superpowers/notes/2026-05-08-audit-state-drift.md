# `state.md` Drift Audit (2026-05-08)

> Follow-up to the 2026-04-26 audit (`docs/superpowers/notes/2026-04-26-audit-state.md`).
> Two weeks of code work since then — Wave 2 piece 1 (control-verb expiries),
> Wave 2 piece 4 (lifesteal + damage_modify buff slots), Wave 3 ToM Phase 2
> (per-observer `cold_beliefs`), and the Lift A–D foundation merge (chronicle
> ordinals 70–76 + `AgentFieldId` variants for the future per-agent columns).
> Audit re-runs `crates/engine/src/state/mod.rs`, `crates/engine/src/schema_hash.rs`,
> `crates/engine/.schema_hash`, `crates/dsl_compiler/src/cg/data_handle.rs`, and
> the chronicle-event `EventKindId` table (`crates/engine/src/cascade/handler.rs`).
> Inline `> ⚠️ Audit 2026-05-08:` callouts at affected sections of `state.md` point
> back here. **Additive only — no prose was rewritten.**

---

## Summary

| Category | Count |
|---|---|
| `[CRITICAL]` (silently miscompiles, or stale field consumed today) | **0** |
| `[STATUS-FLIP]` (spec said missing, now landed) | **0** |
| `[UNDOCUMENTED]` (impl has SoA field spec catalog doesn't name) | **5** |
| `[MISSING]` (spec promises field with no impl) | **0** |
| `[STALE]` (spec callout / pointer no longer matches reality) | **3** |

No critical drifts. The 2026-04-26 catalog is structurally still accurate — the
big-rock gaps it called out (`AgentData` blob, `Aggregate`/`World` sections,
`Goal`/`Aspiration`/`Emotions`/`Source`/`MemEventType`) remain absent from
`crates/engine`. New drift is concentrated in (a) hot-SoA fields the catalog
doesn't name (Wave 2 + Wave 3 storage), and (b) Lift A–D vocabulary that landed
chronicle-event ordinals + `AgentFieldId` variants but **defers** the per-agent
SoA columns themselves to a follow-up slice.

The previous-audit count (42 ✅ / 18 ⚠️ / 31 ❌ / 7 🤔 / 4 ❓) is left intact;
this slice does not re-tally per-section status, only adds drift items that
landed since.

---

## `[UNDOCUMENTED]` items

### U1. Wave 2 piece 1 control-verb expiry mirrors

- **§ in spec**: §Combat/Vitality + §StatusEffect catalog `kind: StatusEffectKind`
  with enum variants including `Root`, `Silence` (StatusEffect §). The 2026-04-26
  callout already pinned the `remaining_ms` ↔ `hot_stun_expires_at_tick` shape
  flip, but only enumerated `hot_stun_expires_at_tick` / `hot_slow_expires_at_tick`.
- **What code does**: `state/mod.rs:107-110` adds four more hot expiry-tick mirrors
  for the Wave 2 piece 1 control verbs:
  - `hot_root_expires_at_tick:    Vec<u32>`
  - `hot_silence_expires_at_tick: Vec<u32>`
  - `hot_fear_expires_at_tick:    Vec<u32>`
  - `hot_taunt_expires_at_tick:   Vec<u32>`
  Same encoding as `hot_stun_expires_at_tick`: absolute expiry tick, `0` means
  "never applied / already elapsed". Schema-hash baseline registers them in the
  same hot block (`schema_hash.rs:21`).
- **Fix landing in this PR**: §StatusEffect callout enumerates the new four hot
  expiry mirrors. No catalog row added (these are the SoA-storage shape, not
  the per-agent typed `StatusEffect` row catalog already covers).

### U2. Wave 2 piece 4 buff-slot SoA columns (lifesteal + damage_taken_mult)

- **§ in spec**: not present. §StatusEffect `kind: StatusEffectKind` enumerates
  `Buff/Debuff{stat, factor}` (already pinned by the 2026-04-26 callout to
  `payload_q8: i16`), but the spec doesn't catalog the dedicated single-slot
  buff columns the engine carries.
- **What code does**: `state/mod.rs:125-138` adds four hot buff-slot fields:
  - `hot_lifesteal_frac_q8:                 Vec<i16>` — q8 fraction of damage healed; `0` = no lifesteal active
  - `hot_lifesteal_expires_at_tick:         Vec<u32>` — gate: `state.tick < expires_at`
  - `hot_damage_taken_mult_q8:              Vec<i16>` — q8 incoming-damage multiplier; `256` = `1.0×` identity
  - `hot_damage_taken_mult_expires_at_tick: Vec<u32>` — gate: `state.tick < expires_at`
  Schema-hash baseline registers all four in the hot block (`schema_hash.rs:21`).
  Stacking semantics ("incoming wins iff strictly greater magnitude OR equal
  magnitude with longer remaining duration") are recorded in `MEMORY.md` →
  `project_buff_stacking_rule.md`, not in `state.md`.
- **Fix landing in this PR**: §StatusEffect callout enumerates the four new
  buff-slot columns alongside U1.

### U3. Wave 3 ToM Phase 2 — `cold_beliefs` per-observer `BoundedMap`

- **§ in spec**: §PerceivedPersonality (Theory of Mind) callout (2026-04-26)
  notes `BeliefState` in `engine_data/src/belief.rs` stores observation data
  (`last_known_pos`, `last_known_hp`, etc.) — accurate. But the catalog itself
  never lists a `cold_beliefs` field on `SimState`.
- **What code does**: `state/mod.rs:175-176` declares
  `cold_beliefs: Vec<crate::pool::BoundedMap<AgentId, engine_data::belief::BeliefState, 8>>`
  per-agent (gated by the `theory-of-mind` feature flag). Per-(observer,
  target) lookup via `BoundedMap` keyed on observed `AgentId`, capacity `N=8`
  (matches `engine_data::belief::BELIEFS_PER_AGENT`). `BeliefState` is hand-written
  and carries 6 fields: `last_known_pos`, `last_known_hp`, `last_known_max_hp`,
  `last_known_creature_type`, `last_updated_tick`, `confidence`. Eviction
  threshold `EVICTION_THRESHOLD = 0.05`.
- **Fix landing in this PR**: §PerceivedPersonality callout updated with the
  actual `cold_beliefs` field shape + capacity. See also S1 below for the
  related comment-shape divergence in cascade/apply.

### U4. `ability_cooldowns` per-(agent, ability-slot) grid

- **§ in spec**: not present in §AgentData / §Goal / §Equipment. The 2026-04-26
  callout in §World already mentions `ability_cooldowns` in passing as one of
  several fields the agent catalog doesn't list, but no §Agent-side callout
  exists.
- **What code does**: `state/mod.rs:187`
  `pub ability_cooldowns: Vec<[u32; MAX_ABILITIES]>` — per-(agent, ability-slot)
  local cooldown cursor where `[slot * MAX_ABILITIES + ability_idx] = next_ready_tick`.
  Grouped with cold SoA fields (read only on cast-gate eval). Added 2026-04-22
  to fix a shared-cursor bug where all abilities on one agent were gated by
  the single global `hot_cooldown_next_ready_tick`.
- **Fix landing in this PR**: §Personality / Combat closing area gets a brief
  note pointing at this field. (Full §Abilities section is deferred — it would
  duplicate `ability.md`.)

### U5. `hot_mana` / `hot_max_mana`

- **§ in spec**: §Combat/Vitality catalogs `hp`, `max_hp`, `shield_hp`, `armor`,
  `magic_resist`, `attack_damage`, `attack_range`, `status_effects`. Spec-listed
  agent-resource catalog stops at HP / shield / armor. Mana is implicit in the
  `cost: <int>` `.ability` parser default (Mana resource — `dataset/abilities/`
  corpus convention) but never named at the SoA layer.
- **What code does**: `state/mod.rs:63-64` declares `hot_mana: Vec<f32>` and
  `hot_max_mana: Vec<f32>`. Schema-hash baseline registers both
  (`schema_hash.rs:17`). Initialised to `0.0` by `spawn_agent`. Read by the
  ability cast-gate when `EffectOp` consumers honour the ability's `cost`.
- **Note**: 2026-04-26 audit narrative mentions this in the §Combat per-section
  finding ("engine adds `hot_mana` / `hot_max_mana` which spec does not catalog
  (engine extension, not spec violation)"), but the spec body itself has no
  callout. Fix landing in this PR: §Combat/Vitality gets a one-line callout
  pointer to this audit.

---

## `[STALE]` items

### S1. Lift A–D `AgentFieldId` variants registered, SoA columns NOT yet allocated

- **§ in spec**: not present. The Lift A foundation merge (commit `7bb0929c`,
  PR #38) added `AgentFieldId` variants `BusyUntilTick`, `TravelDestX`,
  `TravelDestY`, `TravelDestZ` for the multi-tick-procedure / Travel surface,
  plus chronicle event ordinals 70–76 (`EffectTravelToApplied=70`,
  `EffectRecipeApplied=71`, `EffectWearToolApplied=72`, `EffectProposeApplied=73`,
  `EffectAnnounceApplied=74`, `EffectGainSkillApplied=75`,
  `EffectCreateObligationApplied=76`).
- **What's stale**: Doc-comments in `crates/engine/src/cascade/handler.rs` (e.g.
  L260–272) and `crates/engine/src/ability/apply.rs` (e.g. L186, L213–238)
  describe the *future-shape* downstream consumers will read — "writes
  `disguise_expires_at_tick = world.tick + duration_ticks` and
  `disguise_fake_type = fake_type` into per-agent SoA columns", "sets
  `busy_until_tick = world.tick + eta_ticks` and populates `travel_dest_{x,y,z}`
  SoA cells". **Those columns are NOT allocated** in the live `SimState` today
  — the only hot/cold SoA fields are the ones enumerated in `state/mod.rs`
  and in `schema_hash.rs`'s baseline. The dispatcher fires the chronicle event
  but the consumer rule that would *use* the SoA columns doesn't exist yet.
- **Cross-reference**: this is the same caveat the engine.md audit (PR #40)
  pinned at §Schema hash and §GPU resident-mirror. Foundation merges land
  the `EventKindId` ordinal + `EffectOp` variant + `AgentFieldId` vocabulary;
  the per-agent SoA columns themselves land in subsequent slices.
- **Fix landing in this PR**: state.md gets a top-level callout warning that
  the Lift A–D-era doc-comments in code refer to future SoA columns that
  the catalog correctly omits.

### S2. Wave 3 ToM Phase 2 `BeliefState` is per-observer `BoundedMap`, not flat 6-column SoA

- **§ in spec**: §PerceivedPersonality callout (2026-04-26) describes
  `BeliefState` as observation data with the right fields — accurate.
- **What's stale**: cascade / apply doc-comments
  (`cascade/handler.rs:271`, `ability/program.rs:515,630`,
  `ability/apply.rs:186`) describe the consumer-side as "the BeliefState SoA's
  6 columns at `[caster_slot * agent_cap + target_slot]`" — i.e. a flat
  `agent_cap × agent_cap × 6` column-major layout. The actual storage is
  `Vec<BoundedMap<AgentId, BeliefState, 8>>` per observer (`state/mod.rs:175-176`):
  per-observer hash-map keyed on observed `AgentId`, capacity 8, holding
  one `BeliefState` struct per pair. This is the same comment-shape vs
  reality divergence the engine.md audit pinned at the §Schema hash callout
  (`engine.md:124`).
- **Fix landing in this PR**: §PerceivedPersonality callout updated to point
  at this audit + the engine.md cross-reference.

### S3. Aggregate/World "live in headless_campaign / bevy_game layer" pointer

- **§ in spec**: top-of-file callout (`state.md:3`) and §Aggregate / §World
  callouts (lines 497–498, 885–886) say the missing structs "live in the
  `headless_campaign` / `bevy_game` layer (or in legacy worktrees)".
- **What's stale**: neither `headless_campaign` nor `bevy_game` exists as a
  workspace member today. Only `SettlementId` (and other ID stubs) survive in
  `crates/engine/src/ids.rs` — the Settlement / RegionState / Faction /
  GuildState / TradeRoute / EconomyState struct definitions don't live in
  *any* current crate. They were last present in legacy worktrees that have
  since been cleaned up.
- **Note for completeness**: this doesn't change the audit verdict (still ❌
  "not implemented in the engine crate"), but the "lives elsewhere" pointer
  is no longer accurate. The reader should not go looking for these in a
  sibling crate. Fix landing in this PR: top-of-file callout + §Aggregate +
  §World callouts get a "no longer present in any workspace crate" addendum
  via a single 2026-05-08 follow-up callout.

---

## Items that LOOK like drift but aren't

These were checked and confirmed correctly documented or out of scope:

- **`hot_engaged_with` / `hot_stun_expires_at_tick` / `hot_slow_expires_at_tick`
  / `hot_slow_factor_q8` / `hot_cooldown_next_ready_tick`** — listed in the
  2026-04-26 §World callout (line 885). Status unchanged.
- **`creature_type` enum (Human / Wolf / Deer / Dragon)** — already pinned at
  §Identity & Lifecycle callout. Status unchanged: Elf / Dwarf / Goblin still
  absent.
- **Needs scale (0–1 vs spec 0–100)** — already pinned at §Needs callout.
  Status unchanged.
- **`cold_creditor_ledger` / `cold_mentor_lineage` / `cold_class_definitions`** —
  listed in the §AgentData callout's "exists as separate cold fields" carve-out
  (line 282). Status unchanged.
- **`hot_root_expires_at_tick` / `hot_silence_expires_at_tick` etc.** were
  added 2026-05-04 (Wave 2 piece 1) — see U1 above. Wave 2 piece 4 buff slots
  added similarly.
- **`AgentFieldId::Vel`** (Phase 7 boids fixture, 2026-05-02) — used by
  per-fixture runtime crates that allocate their own velocity column outside
  `SimState`. The wolf-sim `SimState` has no velocity slot. This isn't a
  drift item — `state.md` is the wolf-sim catalog, not a per-fixture catalog.

---

## Recommended follow-up tasks

(Out of scope for this docs-only audit slice; tracked for the next pass.)

1. **§Combat/Vitality SoA-storage subsection** — explicit catalog of the hot
   expiry-tick + buff-slot mirrors (U1 + U2). Currently the spec catalogs the
   typed `StatusEffect` row but not the parallel per-status hot columns the
   engine actually reads in masks / scoring.
2. **§Theory of Mind subsection** — promote the §PerceivedPersonality callout
   to a real catalog row for `cold_beliefs` (U3). Pair with the `beliefs(...)`
   DSL surface (currently in the dsl.md drift audit's U3 / U4 deferred items).
3. **§Lift A–D forward-shape catalog** — when the Lift A–D consumer rules
   land per-agent SoA columns, backfill them to §Goal & Action Execution and
   add a §Multi-Tick Procedures section. S1 retires once SoA wiring lands.
4. **Aggregate/World provenance refresh** — either delete the "lives in
   headless_campaign" pointers or update them to "deferred; see ROADMAP" once
   the world-sim plan picks an owning crate. S3 retires.
