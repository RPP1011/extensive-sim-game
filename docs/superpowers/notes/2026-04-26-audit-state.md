# State Spec Audit (2026-04-26)

> Audit of `docs/spec/state.md` (SoA field catalog, 1164 lines) against the actual SimState in
> `crates/engine/src/state/mod.rs`, `crates/engine/src/state/agent_types.rs`, and
> `crates/engine_data/src/`.

---

## Summary

- **Sections audited:** 22 (Agent Identity/Physical/Combat/Needs/Personality/Emotions/Aspiration/Memory/Relationship/Goal/AgentData/BuildingData/Room/Inventory + all Aggregate/World sections)
- ✅ Implemented: 42
- ⚠️ Partial: 18
- ❌ Not implemented: 31
- 🤔 Spec mismatch: 7
- ❓ Needs judgment: 4

**Hot vs cold field split status:** The engine correctly classifies hot (every-tick) vs cold (spawn/debug) per `state.md`'s intent. Every field named "hot_" in SimState is a dense `Vec<T>` and every cold field is `Vec<Option<T>>` or `Vec<SmallVec<...>>`.

**DSL-emitted vs hand-written split:** `crates/engine_data/src/events/` and `crates/engine_data/src/entities/` are compiler-emitted (GENERATED header). `crates/engine/src/state/mod.rs` and `agent_types.rs` are hand-written. `crates/engine_rules/src/views/` (memory, standing) are compiler-emitted. `SimState` itself is hand-written and matches the top of state.md's agent-SoA commitment.

---

## Top gaps (ranked by impact)

1. **❌ Entire AgentData sub-struct not implemented** — The spec defines ~40 fields under `AgentData` (§AgentData: economic intent, campaign fields, emotions 6D, aspiration, behavior_profile, action_outcomes, price_beliefs, cultural_bias, current_intention, equipment, passive_effects, world_abilities, etc.). None of these appear in SimState or any sub-module. The engine carries only the SoA hot/cold fields (combat, needs, personality, memberships), not the rich per-agent blob state.md catalogs.

2. **❌ Aggregate state (WorldState) not implemented in this crate** — The spec's §Aggregate and §World sections catalog Settlement, RegionState, Faction, GuildState, Quest/QuestPosting, TradeRoute, ServiceContract, EconomyState, Group (full shape), ConstructionMemory, ChronicleEntry, WorldEvent, NavGrid, FidelityZone, BuildSeed, VoxelWorld, RegionPlan, GroupIndex, etc. Only a skeletal `Group` (3 fields) and `Quest` (5 fields) exist in `crates/engine/src/aggregate/`. The full WorldState and all Aggregate sub-structs are absent from this crate. They exist in the legacy `headless_campaign` worktree crate but not in the canonical engine crate.

3. **⚠️ Memory retired to DSL view — spec shape diverges** — `state.md §Memory` commits to `VecDeque<MemoryEvent>` (cap 20) with 7 typed fields: tick, MemEventType enum, location vec3, entity_ids Vec, emotional_impact f32, Source enum, confidence f32. The engine's retirement comment says Memory was moved to a `@per_entity_ring(K=64)` view in `crates/engine_rules/src/views/memory.rs`. The view's `MemoryEntry` has only `source: u32, value: f32, anchor_tick: u32` — a 3-field opaque shell vs 7 typed spec fields. `MemEventType` enum, `Source` enum, location, entity_ids, and emotional_impact are all absent.

---

## Per-section findings

### §Agent — Identity & Lifecycle
- Status: ⚠️ Partial
- Evidence: `state/mod.rs:50-55` (hot_alive, hot_level), `cold_creature_type`
- Notes: `id` (AgentId) exists via pool. `creature_type` cold. `alive` hot. `level` hot. Missing: `creature_type` enum is partial (only Human/Wolf/Deer/Dragon; spec lists Goblin/Elf/Dwarf/Dragon). Elf, Dwarf, Goblin are ❌ not in CreatureType enum (`engine_data/src/entities/mod.rs`).

### §Agent — Physical State
- Status: ✅ Implemented
- Evidence: `mod.rs:49` (hot_pos Vec3), `mod.rs:55-57` (hot_move_speed, hot_move_speed_mult), `mod.rs:107-110` (cold_grid_id, cold_local_pos, cold_move_target)
- Notes: All 6 fields present with correct types. Ground-snap cascade logic is trait-injected via `terrain: Arc<dyn TerrainQuery>`.

### §Agent — Combat/Vitality
- Status: ✅ Implemented + 🤔 mismatch on `status_effects`
- Evidence: `mod.rs:50-64` (hot_hp, hot_max_hp, hot_shield_hp, hot_armor, hot_magic_resist, hot_attack_damage, hot_attack_range)
- Notes: `status_effects` is `cold_status_effects: Vec<SmallVec<[StatusEffect; 8]>>` (cold) but spec says it should be per-agent. Shape mismatch: spec uses `Vec<StatusEffect>`, engine uses SmallVec cap 8. Also engine adds `hot_mana` / `hot_max_mana` which spec does not catalog (engine extension, not spec violation).

### §StatusEffect
- Status: ⚠️ Partial
- Evidence: `agent_types.rs:17-37`
- Notes: `StatusEffectKind` enum matches spec {Stun,Slow,Root,Silence,Dot,Hot,Buff,Debuff}. `source: AgentId` matches `source_id`. `remaining_ticks: u32` — spec says `remaining_ms: u32` (milliseconds). 🤔 Name diverges: engine uses ticks, spec uses ms. `payload_q8: i16` replaces per-kind payloads (Slow{factor}, Buff/Debuff{stat,factor}); opaque shell not typed.

### §Needs (8-dimensional)
- Status: ✅ Implemented (hot SoA), ⚠️ scale mismatch
- Evidence: `mod.rs:66-74`
- Notes: All 8 dims present as hot SoA (`hot_hunger`, `hot_thirst`, `hot_rest_timer`, `hot_safety`, `hot_shelter`, `hot_social`, `hot_purpose`, `hot_esteem`). Engine initializes to 1.0; spec says range 0–100. 🤔 Scale diverges: engine uses 0.0–1.0, spec says 0–100.

### §Personality (5D)
- Status: ✅ Implemented
- Evidence: `mod.rs:75-81` (hot_risk_tolerance, hot_social_drive, hot_ambition, hot_altruism, hot_curiosity)
- Notes: All 5 dims present. Spec uses `compassion`; engine uses `altruism` — acknowledged in comment as deliberate rename. Accessors exist for all 5 read+write.

### §Emotions (6D)
- Status: ❌ Not implemented
- Evidence: No `joy`, `anger`, `fear`, `grief`, `pride`, `anxiety` fields in SimState or any sub-module in `crates/engine`.
- Notes: Entire 6-dim emotion layer absent. High impact: emotions drive flee urgency, combat preference, productivity.

### §Aspiration
- Status: ❌ Not implemented
- Evidence: No `need_vector`, `vector_formed_at`, `crystal`, `crystal_progress`, `crystal_last_advanced` anywhere in engine crate.
- Notes: Medium-term goal orientation completely absent.

### §Memory
- Status: ⚠️ Partial (retired to DSL view; schema narrowed)
- Evidence: `engine_rules/src/views/memory.rs` (generated), retired comment `state/mod.rs:118-122`
- Notes: Storage moved to `@per_entity_ring(K=64)`. Ring is K=64 (spec cap=20 — larger, not a regression). But `MemoryEntry` has 3 fields vs spec's 7 typed fields. `MemEventType` enum, `Source` enum, location, entity_ids, emotional_impact all missing. Spec's `Memory.beliefs` (Vec<Belief>) layer entirely absent.

### §Source enum
- Status: ❌ Not implemented
- Evidence: No Source enum in any engine or engine_data file.
- Notes: Provenance tagging for memory entries (Witnessed/TalkedWith/Overheard/Rumor/Announced/Testimony) absent.

### §Relationship
- Status: ⚠️ Partial (MVP shell)
- Evidence: `agent_types.rs:98-102`
- Notes: `other: AgentId`, `valence_q8: i16` (trust only), `tenure_ticks: u32`. Missing: `familiarity`, `last_interaction`, `perceived_personality` (PerceivedPersonality struct), `believed_knowledge` (Bitset<32>). Cap is 8 (cold_relationships SmallVec), spec says 20. 🤔 Cap mismatch.

### §PerceivedPersonality (Theory of Mind)
- Status: ⚠️ Partial (feature-gated stub)
- Evidence: `state/mod.rs:137-138` (`cold_beliefs` behind `#[cfg(feature = "theory-of-mind")]`)
- Notes: `BeliefState` in `engine_data/src/belief.rs` has `last_known_pos`, `last_known_hp`, `last_known_max_hp`, `last_known_creature_type`, `last_updated_tick`, `confidence`. This is observation-belief not theory-of-mind personality modeling. Spec's `PerceivedPersonality.traits [f32;5]` + `confidence [f32;5]` + `observation_count` are absent.

### §Membership
- Status: ✅ Implemented
- Evidence: `agent_types.rs:58-63`
- Notes: `group: GroupId`, `role: GroupRole`, `joined_tick: u32`, `standing_q8: i16`. All 4 spec fields present. Standing is q8 fixed-point vs spec's f32 [-1,1] — acceptable encoding. GroupRole enum matches exactly {Member,Officer,Leader,Founder,Apprentice,Outcast}.

### §Goal & GoalStack
- Status: ❌ Not implemented (in engine SoA)
- Evidence: No GoalKind, GoalStack, or Goal struct in `crates/engine/src/state/`. `crates/tactical_sim/src/goap/goal.rs` has a `Goal` but it's the tactical GOAP planner, not the world-sim goal stack.
- Notes: The spec's `goal_stack` with push/pop and priority preemption is part of AgentData which is absent.

### §AgentData (full sub-struct)
- Status: ❌ Not implemented
- Evidence: No `AgentData` struct in engine crate.
- Notes: ~40 fields absent: name, adventurer_id, gold, debt, creditor_id, income_rate, credit_history, economic_intent, price_knowledge, trade_route_id, trade_history, home_settlement_id, home_building_id, work_building_id, inside_building_id, current_room, home_den, cached_path (nav), work_state, action, behavior_production, class_tags, behavior_profile, action_outcomes, price_beliefs, cultural_bias, morale, stress, fatigue, loyalty, injury, resolve, archetype, party_id, faction_id, mood, fears, deeds, guild_relationship, current_intention, intention_ticks, equipment, equipped_items, passive_effects, world_abilities. Only a few (classes via `cold_class_definitions`, creditors via `cold_creditor_ledger`, mentor_lineage via `cold_mentor_lineage`) are partially scaffolded as separate cold fields.

### §BuildingData
- Status: ❌ Not implemented
- Evidence: No BuildingData struct in engine crate.

### §Room
- Status: ❌ Not implemented
- Evidence: No Room struct in engine crate.

### §Inventory
- Status: ⚠️ Partial
- Evidence: `agent_types.rs:72-75`
- Notes: `gold: i32` (spec says `f32`; engine uses i32 for GPU atomics — deliberate divergence), `commodities: [u16; 8]` (spec says `[f32; 8]`). Missing: `capacity: f32`. 🤔 Gold and commodity types diverge from spec.

### §Settlement (Aggregate)
- Status: ❌ Not implemented in engine crate
- Evidence: Only found in `.claude/worktrees/agent-aa2db99f/` (worktree, not canonical).

### §RegionState (Aggregate)
- Status: ❌ Not implemented in engine crate

### §Faction (Aggregate)
- Status: ❌ Not implemented in engine crate

### §GuildState (Aggregate)
- Status: ❌ Not implemented in engine crate

### §Quest & QuestPosting (Aggregate)
- Status: ⚠️ Partial
- Evidence: `crates/engine/src/aggregate/quest.rs`
- Notes: `Quest` has `seq`, `poster`, `category`, `resolution`, `acceptors`, `posted_tick` (6 fields). Spec has 13 fields. Missing: name, destination, progress, status, accepted_tick, deadline_tick, threat_level, reward_gold, reward_xp. `QuestPosting` entirely absent. QuestCategory is 5-bucket enum not matching spec's 10-type QuestType.

### §Group (universal, Aggregate)
- Status: ⚠️ Partial (MVP stub)
- Evidence: `crates/engine/src/aggregate/group.rs`
- Notes: `Group` has `kind_tag: u32, members: SmallVec<[AgentId;8]>, leader: Option<AgentId>`. Spec has ~25 fields across Identity/Leadership/Resources/Standings/Recruitment/Activity. Missing: id (GroupId), name, founded_tick, dissolved_tick, founder_ids, leadership_chain, governance, charter, treasury, stockpile, facilities, military_strength, standings (Map<GroupId,Standing>), standing_history, recruitment_open, eligibility_predicate, dues, active_quests, recent_activity.

### §TradeRoute (Aggregate)
- Status: ❌ Not implemented in engine crate

### §ServiceContract (Aggregate)
- Status: ❌ Not implemented in engine crate

### §EconomyState (Aggregate)
- Status: ❌ Not implemented in engine crate

### §ChronicleEntry (Aggregate/World)
- Status: ⚠️ Partial (DSL-emitted, narrowed schema)
- Evidence: `engine_data/src/events/chronicle_entry.rs` (GENERATED)
- Notes: Emitted struct has `template_id: u32, agent: AgentId, target: AgentId, tick: u32`. Spec has `tick, category, text: String, entity_ids: Vec<u32>`. Missing: category, text (narrative human-readable), entity_ids array. Template approach encodes category implicitly in template_id; text is deferred to template lookup (not stored). ❓ Design intent may be deliberate for GPU compatibility.

### §WorldEvent (World)
- Status: ❌ Not implemented as dedicated struct in engine crate
- Evidence: Several individual event structs in `engine_data/src/events/` (agent_died, quest_posted, etc.) but no unified WorldEvent enum with 13 variants.

### §WorldState top-level scalar fields (tick, rng_state, next_id)
- Status: ⚠️ Partial
- Evidence: `mod.rs:34,37` (tick: u32, seed: u64)
- Notes: `tick` exists as `u32` (spec says `u64`). `seed` exists as `u64`. `rng_state` not separately tracked (seed used directly). `next_id` managed via `AgentSlotPool` not exposed as a named field.

### §VoxelWorld (World)
- Status: ❌ Not implemented in engine crate
- Evidence: No VoxelWorld in engine crate. Present in tactical_sim/bevy_game layers.

### §RegionPlan (World)
- Status: ❌ Not implemented in engine crate

### §NavGrid (World)
- Status: ❌ Not implemented in engine crate

### §FidelityZone (World)
- Status: ❌ Not implemented in engine crate

### §BuildSeed (World)
- Status: ❌ Not implemented in engine crate

### §StructuralEvent (World)
- Status: ❌ Not implemented in engine crate

### §SimScratch (World)
- Status: ❌ Not implemented in engine crate
- Notes: `crates/engine/src/scratch.rs` exists but likely covers different scratch buffers than the world-sim SimScratch.

### §GroupIndex (World)
- Status: ❌ Not implemented in engine crate

---

## Field-level inventory

| Spec field name | Spec §ref | Implementation site | Status |
|---|---|---|---|
| id (AgentId) | §Identity | AgentSlotPool (pool-managed) | ✅ |
| creature_type | §Identity | cold_creature_type Vec<Option<CreatureType>>, mod.rs:103 | ✅ |
| alive | §Identity | hot_alive Vec<bool>, mod.rs:52 | ✅ |
| level | §Identity | hot_level Vec<u32>, mod.rs:54 | ✅ |
| pos | §Physical | hot_pos Vec<Vec3>, mod.rs:49 | ✅ |
| grid_id | §Physical | cold_grid_id Vec<Option<u32>>, mod.rs:108 | ✅ |
| local_pos | §Physical | cold_local_pos Vec<Option<Vec3>>, mod.rs:109 | ✅ |
| move_target | §Physical | cold_move_target Vec<Option<Vec3>>, mod.rs:110 | ✅ |
| move_speed | §Physical | hot_move_speed Vec<f32>, mod.rs:55 | ✅ |
| move_speed_mult | §Physical | hot_move_speed_mult Vec<f32>, mod.rs:56 | ✅ |
| hp | §Combat | hot_hp Vec<f32>, mod.rs:50 | ✅ |
| max_hp | §Combat | hot_max_hp Vec<f32>, mod.rs:51 | ✅ |
| shield_hp | §Combat | hot_shield_hp Vec<f32>, mod.rs:58 | ✅ |
| armor | §Combat | hot_armor Vec<f32>, mod.rs:59 | ✅ |
| magic_resist | §Combat | hot_magic_resist Vec<f32>, mod.rs:60 | ✅ |
| attack_damage | §Combat | hot_attack_damage Vec<f32>, mod.rs:61 | ✅ |
| attack_range | §Combat | hot_attack_range Vec<f32>, mod.rs:62 | ✅ |
| status_effects | §Combat | cold_status_effects Vec<SmallVec<[StatusEffect;8]>>, mod.rs:111 | ⚠️ cap=8 not Vec; cold not hot |
| StatusEffect.kind | §StatusEffect | agent_types.rs:17-26 | ✅ |
| StatusEffect.source_id | §StatusEffect | agent_types.rs:34 (field: source) | ✅ |
| StatusEffect.remaining_ms | §StatusEffect | agent_types.rs:35 (field: remaining_ticks) | 🤔 name/unit diverges |
| StatusEffect payload (typed) | §StatusEffect | agent_types.rs:36 (payload_q8 opaque) | ⚠️ opaque not typed |
| hunger | §Needs | hot_hunger Vec<f32>, mod.rs:66 | ⚠️ range 0–1 vs spec 0–100 |
| thirst | §Needs (engine ext) | hot_thirst Vec<f32>, mod.rs:67 | ❓ not in spec Needs table |
| rest_timer | §Needs (engine ext) | hot_rest_timer Vec<f32>, mod.rs:68 | ❓ not in spec Needs table |
| safety | §Needs | hot_safety Vec<f32>, mod.rs:70 | ⚠️ range 0–1 vs spec 0–100 |
| shelter | §Needs | hot_shelter Vec<f32>, mod.rs:71 | ⚠️ range 0–1 vs spec 0–100 |
| social | §Needs | hot_social Vec<f32>, mod.rs:72 | ⚠️ range 0–1 vs spec 0–100 |
| purpose | §Needs | hot_purpose Vec<f32>, mod.rs:73 | ⚠️ range 0–1 vs spec 0–100 |
| esteem | §Needs | hot_esteem Vec<f32>, mod.rs:74 | ⚠️ range 0–1 vs spec 0–100 |
| risk_tolerance | §Personality | hot_risk_tolerance Vec<f32>, mod.rs:77 | ✅ |
| social_drive | §Personality | hot_social_drive Vec<f32>, mod.rs:78 | ✅ |
| ambition | §Personality | hot_ambition Vec<f32>, mod.rs:79 | ✅ |
| altruism (compassion) | §Personality | hot_altruism Vec<f32>, mod.rs:80 | 🤔 name differs from spec (altruism vs compassion) |
| curiosity | §Personality | hot_curiosity Vec<f32>, mod.rs:81 | ✅ |
| joy | §Emotions | — | ❌ |
| anger | §Emotions | — | ❌ |
| fear | §Emotions | — | ❌ |
| grief | §Emotions | — | ❌ |
| pride | §Emotions | — | ❌ |
| anxiety | §Emotions | — | ❌ |
| Aspiration.need_vector | §Aspiration | — | ❌ |
| Aspiration.crystal | §Aspiration | — | ❌ |
| Aspiration.crystal_progress | §Aspiration | — | ❌ |
| Memory.events (VecDeque<MemoryEvent>) | §Memory | DSL view MemoryEntry (3 fields), views/memory.rs | ⚠️ shape narrowed |
| Memory.beliefs (Vec<Belief>) | §Memory.beliefs | — | ❌ |
| Source enum | §Source | — | ❌ |
| Relationship.trust | §Relationship | valence_q8 (agent_types.rs:100) | ⚠️ opaque q8 not named trust |
| Relationship.familiarity | §Relationship | — | ❌ |
| Relationship.last_interaction | §Relationship | tenure_ticks (agent_types.rs:101) | ⚠️ tenure, not last interaction |
| Relationship.perceived_personality | §Relationship | — | ❌ |
| Relationship.believed_knowledge | §Relationship | — | ❌ |
| Membership.group_id | §Membership | agent_types.rs:59 | ✅ |
| Membership.role | §Membership | agent_types.rs:60 | ✅ |
| Membership.joined_tick | §Membership | agent_types.rs:61 | ✅ |
| Membership.standing | §Membership | standing_q8 (agent_types.rs:62) | ⚠️ q8 not f32 |
| data (AgentData) | §Sub-structures | — | ❌ not a sub-struct |
| inventory | §Sub-structures | cold_inventory Vec<Inventory>, mod.rs:115 | ⚠️ partial (see Inventory) |
| memberships | §Sub-structures | cold_memberships, mod.rs:113 | ✅ |
| capabilities | §Sub-structures | cold_channels (ChannelSet only), mod.rs:104 | ⚠️ only channels stored, full Capabilities not SoA |
| Inventory.commodities | §Inventory | [u16;8] (agent_types.rs:74) | 🤔 u16 not f32 |
| Inventory.gold | §Inventory | i32 (agent_types.rs:73) | 🤔 i32 not f32 (GPU reason) |
| Inventory.capacity | §Inventory | — | ❌ |
| GoalStack | §Goal | — | ❌ |
| AgentData.name | §AgentData | — | ❌ |
| AgentData.gold | §AgentData | Inventory.gold (partial) | ⚠️ |
| AgentData.home_settlement_id | §AgentData | — | ❌ |
| AgentData.work_state | §AgentData | — | ❌ |
| AgentData.class_tags | §AgentData | cold_class_definitions (tag hash only) | ⚠️ |
| AgentData.behavior_profile | §AgentData | — | ❌ |
| AgentData.morale | §AgentData.Campaign | — | ❌ |
| AgentData.stress | §AgentData.Campaign | — | ❌ |
| AgentData.fatigue | §AgentData.Campaign | — | ❌ |
| AgentData.current_intention | §AgentData | — | ❌ |
| AgentData.equipped_items | §AgentData | — | ❌ |
| AgentData.passive_effects | §AgentData | — | ❌ |
| AgentData.world_abilities | §AgentData | — | ❌ |
| ClassSlot.class_tag | §Skill & Class | cold_class_definitions[4], agent_types.rs:109 | ✅ |
| ClassSlot.level | §Skill & Class | agent_types.rs:110 | ✅ |
| ClassSlot.xp | §Skill & Class | — | ❌ (not in ClassSlot) |
| mentor_lineage | §Relationships | cold_mentor_lineage [Option<AgentId>;8], mod.rs:131 | ✅ |
| Creditor.creditor | §Economic | cold_creditor_ledger, agent_types.rs:116 | ✅ |
| Creditor.amount | §Economic | agent_types.rs:117 | ⚠️ u32 not f32 |
| PerceivedPersonality (ToM) | §Relationship | BeliefState (feature-gated), belief.rs | ⚠️ is observation not theory-of-mind |
| BuildingData | §BuildingData | — | ❌ |
| Settlement | §Aggregate | — (worktree only) | ❌ |
| RegionState | §Aggregate | — | ❌ |
| Faction | §Aggregate | — | ❌ |
| GuildState | §Aggregate | — | ❌ |
| Quest (full) | §Quest | aggregate/quest.rs (partial, 5 fields) | ⚠️ |
| QuestPosting | §Quest | — | ❌ |
| Group (full) | §Group | aggregate/group.rs (3 fields) | ⚠️ |
| TradeRoute | §Aggregate | — | ❌ |
| ServiceContract | §Aggregate | — | ❌ |
| EconomyState | §Aggregate | — | ❌ |
| ChronicleEntry (full) | §World | engine_data/events/chronicle_entry.rs (4 fields) | ⚠️ |
| WorldEvent (enum) | §World | individual event structs, no unified enum | ⚠️ |
| WorldState.tick | §World | SimState.tick: u32, mod.rs:34 | 🤔 u32 not u64 |
| WorldState.seed | §World | SimState.seed: u64, mod.rs:37 | ✅ |
| VoxelWorld | §World | — in engine crate | ❌ |
| RegionPlan | §World | — in engine crate | ❌ |
| NavGrid | §World | — in engine crate | ❌ |
| FidelityZone | §World | — in engine crate | ❌ |
| BuildSeed | §World | — in engine crate | ❌ |
| StructuralEvent | §World | — in engine crate | ❌ |
| SimScratch | §World | engine/src/scratch.rs (different scope) | ❌ |
| GroupIndex | §World | — | ❌ |

---

## Cross-cutting observations

1. **Scope mismatch: engine crate vs world_sim layer.** The spec's §Aggregate and §World sections describe the *WorldState* — the full simulation world object. The `crates/engine` crate is scoped to the *combat engine* (SimState = per-fight SoA for agents). WorldState, VoxelWorld, Settlement, Faction, etc. reside in a separate `headless_campaign` / `bevy_game` layer not audited here. This is a structural scope divergence — the spec treats both as one catalog; the implementation separates them into different crates.

2. **Needs scale divergence (0–1 vs 0–100).** Every Needs field initializes to `1.0` in SimState (`spawn_agent` writes 1.0) but the spec says "Range 0–100. Higher = more satisfied." Systems that compute urgency as `(100 - need) / 100` will be broken unless the formula is adjusted for the 0–1 range. This is a quiet semantic hazard.

3. **tick: u32 vs u64.** SimState uses `u32` tick. Spec says `u64`. At 10 Hz this wraps in ~13.6 years of sim time — not an immediate problem but a spec-to-impl gap. The comment in `state/mod.rs:34` does not note this as intentional.

4. **Memory storage retirement not spec-acknowledged.** The per-entity ring view (K=64) is more capacity than the spec's cap-20 VecDeque, but the 3-field `MemoryEntry` schema is narrower than the 7-field spec definition. The retirement comment acknowledges the change but state.md still shows the old shape. Either state.md needs updating or the view needs richer fields.

5. **Capabilities: only `channels` in SoA.** Spec §Sub-structures says `capabilities: Capabilities` (channels, languages, can_fly, can_build, can_trade, can_climb, can_tunnel, can_marry, max_spouses). Engine stores only `cold_channels: Vec<Option<ChannelSet>>` — the channel set only. `can_fly`, `can_build`, `can_trade`, etc. are reconstructed at runtime from `Capabilities::for_creature(creature_type)` (engine_data/src/entities/mod.rs). This is a deliberate engine optimization (reconstruct from creature_type) but means capability overrides on individual agents are not storable.

6. **DSL-emitted events cover many spec §Aggregate event shapes.** `engine_data/src/events/` contains ~30 generated event structs (quest_posted, quest_accepted, effect_damage_applied, effect_gold_transfer, etc.) that partially satisfy the WorldEvent / ChronicleEntry spec. They are fine-grained where the spec is coarse-grained.

7. **`hot_engaged_with` is an engine extension not in state.md.** Combat engagement lock (`hot_engaged_with: Vec<Option<AgentId>>`) is present in SimState but not mentioned in the spec's field catalog. Same for `hot_stun_expires_at_tick` / `hot_slow_expires_at_tick` / `hot_slow_factor_q8` / `hot_cooldown_next_ready_tick` / `ability_cooldowns` — these are combat-engine implementation details not cataloged in state.md's field tables.
