# World Sim State Catalog

> ⚠️ **Audit 2026-04-26 — scope mismatch:** This catalog covers both per-fight `SimState` (combat engine) **and** the full `WorldState` (campaign / DF-style world sim). The `crates/engine` crate implements only the SoA hot/cold fields under `Agent state`. **Aggregate state (Settlement, RegionState, Faction, GuildState, TradeRoute, ServiceContract, EconomyState, etc.) and World state (VoxelWorld, RegionPlan, NavGrid, FidelityZone, BuildSeed, GroupIndex, etc.) are not implemented in the engine crate** — they live in the `headless_campaign` / `bevy_game` layer (or in legacy worktrees), which the audit did not cover.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for the per-section findings (42 ✅ / 18 ⚠️ / 31 ❌ / 7 🤔 / 4 ❓).

Schema-of-record: every SoA field engine code touches, with semantics, writers, and readers.

## Contents
- [Agent state](#agent-state)
- [Aggregate state](#aggregate-state)
- [World state](#world-state)

---

## Agent state

### Agent (top-level)

Universal entity for any agentic actor (humans, wolves, dragons, goblins). Same struct, distinguished by `creature_type`.

#### Identity & Lifecycle

> ⚠️ **Audit 2026-04-26:** `creature_type` exists but `CreatureType` enum only covers Human/Wolf/Deer/Dragon — **Elf, Dwarf, and Goblin are absent** from the engine enum.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | AgentId | Unique agent ID, stable across ticks | spawn | all systems (lookups) |
| creature_type | CreatureType | Human, Elf, Dwarf, Wolf, Goblin, Dragon, ... | spawn | mask predicates (can_speak, predator_prey), observation features, narrative |
| alive | bool | Whether agent is active | death cascade | action_eval (filtering), movement |
| level | u32 | Power tier | progression (level-up), template init | combat damage scaling, action eval |

#### Physical State
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| pos | vec3 | World-space position (z authoritative) | movement system (advance_movement), post-movement ground-snap cascade, spawning | action_eval (distance / planar_distance / z_separation), threat assessment, pathfinding |
| grid_id | Option<u32> | Settlement grid assignment (for nav caching) | movement entry/exit, grid rebinding | pathfinding, movement cost |
| local_pos | Option<vec3> | Relative position within room (for interiors) | interiors system (room navigation) | rendering, room-based spatial queries |
| move_target | Option<vec3> | Desired destination (set by goal system) | goal system, movement subsystem | movement system (pathfinding goal) |
| move_speed | f32 | Base units/tick | template init, item bonuses, status effects | movement speed calc, action duration |
| move_speed_mult | f32 | Multiplier (default 1.0) | status effects (stun/slow) | effective movement calc |

**Ground-locked vs volumetric.** Ground-locked types (Human, Elf, Dwarf, Wolf, Goblin) get a post-movement `@phase(post)` cascade snapping `pos.z = surface_height(pos.xy) + creature_height/2` outdoors or `floor_height(pos, building) + creature_height/2` when `inside_building_id` is set. Volumetric types (Dragon, Fish, Bat) skip the snap. `creature_height` derived from `creature_type` config.

#### Combat/Vitality
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| hp | f32 | Current health | apply_flat (healing/damage), work (eat restores), resting | death check, threat urgency, combat eval |
| max_hp | f32 | Max health pool | template init, equipment bonuses, progression | hp clamp, healing cap |
| shield_hp | f32 | Absorb layer (damage shields) | status effects, abilities | damage application (absorb first) |
| armor | f32 | Damage reduction % | template init, equipment bonuses, item affixes | damage calc reduction |
| magic_resist | f32 | Magic-specific resist (reserved) | template init | magic damage calc (not yet used) |
| attack_damage | f32 | Outgoing damage | template init, equipment bonuses, status effects | combat damage calc |
| attack_range | f32 | Melee/ranged range (units) | template init | action targeting (distance gating) |
| status_effects | Vec<StatusEffect> | Active temporary effects (stun, slow, dot, buff, debuff) | apply_flat (effect application), cooldown system | status check, status decay, movement/action penalties |

#### Sub-structures

> ⚠️ **Audit 2026-04-26:** `data` (AgentData) does not exist as a sub-struct. `inventory` is partial (see §Inventory). `capabilities` stores only `cold_channels` (ChannelSet) — `can_fly`, `can_build`, `can_trade`, etc. are reconstructed from `creature_type` at runtime, not stored per-agent.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| data | AgentData | Agent state (see AgentData section below) | agent_inner, action_eval, social/economic systems | all behavioral systems |
| inventory | Inventory | Commodity + gold storage (every agent has one) | work (production), trade, eating, looting | action_eval (resource availability), trader intent |
| memberships | Vec<Membership> | Groups this agent belongs to (faction, family, guild, religion, party, pack, settlement) | JoinGroup/LeaveGroup, spawn defaults | mask predicates, observation features, `is_hostile`/`is_friendly` views |
| capabilities | Capabilities | `channels: SortedVec<CommunicationChannel, 4>` over `{Speech, PackSignal, Pheromone, Song, Telepathy, Testimony}`; plus `languages`, `can_fly`, `can_build`, `can_trade`, `can_climb`, `can_tunnel`, `can_marry`, `max_spouses`. Channels drive `PostQuest`/`Communicate`/`Announce`/overhear eligibility; cross-species comms require a shared channel. `can_fly` gates z-separation combat predicates. | template init | pathfinding, construction defense, mask predicates, observation features |

Equipped items referenced by ID from `data.equipped_items`.

---

### Membership

Agent → group association.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| group_id | GroupId | Which group | JoinGroup event | mask predicates, observation features |
| role | GroupRole | Member, Officer, Leader, Founder, Apprentice, Outcast | JoinGroup, PromoteEvent | mask predicates, reputation calc |
| joined_tick | u64 | When joined | JoinGroup event | tenure calc, observation feature |
| standing | f32 | -1..1 reputation within the group | social events involving the group | mask predicates (Outcast cannot vote, etc.) |

---

### StatusEffect

> 🤔 **Audit 2026-04-26 (spec mismatch):** Spec catalogs `remaining_ms: u32`; implementation stores **absolute expiry tick** in `hot_stun_expires_at_tick` / `hot_slow_expires_at_tick` (Task 143). Per-kind typed payload (Slow{factor}, Buff/Debuff{stat,factor}) is replaced by an opaque `payload_q8: i16`. SmallVec cap=8 (cold), not Vec.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` and `docs/superpowers/notes/2026-04-26-audit-language-stdlib.md`.

Temporary state modifier on an entity.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| kind | StatusEffectKind | Enum: Stun, Slow{factor}, Root, Silence, Dot/Hot{rate,interval}, Buff/Debuff{stat,factor} | apply_flat (effect dispatch) | movement (Slow/Root penalty), action (Silence blocks), damage (Dot/Hot tick) |
| source_id | u32 | Entity who applied this effect | apply_flat (effect source) | credit for damage/control |
| remaining_ms | u32 | Time left (milliseconds, converted from ticks) | cooldown system (decay per tick) | effect expiry check |

---

### Needs (8-dimensional)

> ⚠️ **Audit 2026-04-26 (scale mismatch):** Spec says range 0–100. **Implementation uses range 0.0–1.0** (initialized to 1.0 in `spawn_agent`). Quiet semantic hazard for any system that computes urgency as `(100 - need) / 100`.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Maslow-inspired primary drives. Range 0–100. Higher = more satisfied. Engine carries Maslow-5 (`safety/shelter/social/purpose/esteem`) plus 3 physiological (`hunger/thirst/rest_timer`); `hunger` is shared.

#### Need Dimensions
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| hunger | f32 | Satiation level (rises over time without food) | agent_inner (tick drift +1/tick), work (eat action -40), starvation events | action_eval (urgency: Eat at <30), goal selection (priority), focus modifier |
| safety | f32 | Threat proximity / security feeling | agent_inner (nearby hostiles -5, safe location +2), combat death -20 | threat assessment urgency, goal interrupt (Flee >80 urgency) |
| shelter | f32 | Quality of housing | agent_inner (inside good building +1, outside -1), settlement founding | rest location selection, settlement preference |
| social | f32 | Interaction fulfillment | social_gathering (trade +5, gathering +10, conflict -10), death of friend -30 | socializing goal urgency, isolation effects |
| purpose | f32 | Meaningful work | work (productive labor +2, idle -0.5), goal completion +10 | goal selection bias, mood modulation |
| esteem | f32 | Achievement recognition | legendary_deeds (+30 per deed), failed goal -5, social respect from relationships | pride emotion, ambition-driven actions |

**Emergent?** Triggers but not computed from other fields. Primary inputs to aspiration (Phase B).

---

### Personality (5D)

Experience-shaped behavioral profile, 0–1. Set at spawn, drifts via events.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| risk_tolerance | f32 | Engage-danger willingness | combat win +0.02, loss -0.02, near-death -0.05 | action selection, flee threshold |
| social_drive | f32 | Interaction preference | socializing +0.01, isolation -0.01, trade +0.01 | Socialize goal priority, relationship maintenance |
| ambition | f32 | Status/achievement drive | quest +0.02, goal failure -0.02, promotion witness +0.01 | Work/Build goal selection, competitive behavior |
| altruism | f32 | Empathy, healing/helping bias | healing +0.02, witnessing suffering +0.03, cruelty -0.05 | healing/rescue goal frequency, trust modifier |
| curiosity | f32 | Exploration drive | skill +0.02, discovery +0.01, repetitive work -0.01 | exploration frequency, trade diversity |

**Emergent?** Primary input (formed by events); not reconstructible from others.

---

### Emotions (6D, Transient)

> ❌ **Audit 2026-04-26:** Entire 6-dim emotion layer (`joy`, `anger`, `fear`, `grief`, `pride`, `anxiety`) is **not implemented** in `SimState` or any sub-module. High impact: emotions are spec'd to drive flee urgency, combat preference, productivity.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Short-term state, 0–1, decays per tick (≈0.01–0.05 depending on event).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| joy | f32 | Happiness (goal complete +0.3, good trade +0.2) | decay -0.02/tick | morale, productivity |
| anger | f32 | Rage (betrayal +0.4, theft +0.3, insult +0.2) | decay -0.02/tick | combat preference, hostility |
| fear | f32 | Dread (near-death +0.5, threat +0.3) | decay -0.01/tick, safety -0.1 | flee urgency, risk aversion |
| grief | f32 | Sorrow (ally death +0.6, home loss +0.4) | decay -0.01/tick | isolation, productivity penalty |
| pride | f32 | Self-regard (achievement +0.3, recognition +0.2) | decay -0.03/tick | social dominance, ambition |
| anxiety | f32 | Worry (shortage +0.2, goal fail +0.1) | decay -0.02/tick | goal hesitation, conservatism |

---

### Aspiration

> ❌ **Audit 2026-04-26:** `Aspiration` is **not implemented** in `SimState` or any sub-module. `need_vector`, `vector_formed_at`, `crystal`, `crystal_progress`, `crystal_last_advanced` are all absent.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Personality-weighted need-gap vector, 100-tick horizon bias. Recomputed every 500 ticks.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| need_vector | [f32; 6] | Normalized (100 - need[i]) × personality[i] | agent_inner (500-tick recompute) | action_eval bias, goal_eval priority |
| vector_formed_at | u64 | Last recompute tick | agent_inner | recompute interval check |
| crystal | Option<Crystal> | Concrete target (Entity/Class/Building/Location/Resource) bound to a need dim | salient events (quest, discovery, relationship); failure resets | concrete pursue vs. need balancing |
| crystal_progress | f32 | [0,1] completion toward crystal | systems progressing crystal goal | stall check, UI |
| crystal_last_advanced | u64 | Last tick progress increased | crystal progress events | stall detection (abandon crystal) |

---

### Memory

> ⚠️ **Audit 2026-04-26 (shape narrowed):** Memory storage was retired to a `@per_entity_ring(K=64)` view in `engine_rules/src/views/memory.rs`. The view's `MemoryEntry` has only 3 fields (`source: u32, value: f32, anchor_tick: u32`) versus the 7 typed fields documented below (tick, MemEventType enum, location vec3, entity_ids, emotional_impact f32, Source enum, confidence f32). `MemEventType` enum, `Source` enum, location, entity_ids, and emotional_impact are absent. The `Memory.beliefs: Vec<Belief>` layer is entirely absent.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Event log + semantic beliefs.

#### Memory.events
`events: VecDeque<MemoryEvent>` capped at 20. Updated by `memory.record_event`; read by belief formation, narrative, relationships.

#### MemoryEvent
| Field | Type | Meaning |
|---|---|---|
| tick | u64 | When event occurred |
| event_type | MemEventType | WasAttacked, AttackedEnemy, FriendDied, WonFight, CompletedQuest, LearnedSkill, WasHealed, WasBetrayedBy, TradedWith, BuiltSomething, LostHome, WasRescuedBy, Starved, FoundShelter, MadeNewFriend, BecameApprentice, ... |
| location | vec3 | Where it occurred (drives location-based beliefs, pathfinding fear) |
| entity_ids | Vec<u32> | Involved entities (relationship updates, grudge formation) |
| emotional_impact | f32 | [-1, 1] intensity (emotion boost, memory weight) |
| source | Source | Provenance (see Source enum) — drives confidence + observation `info_source_one_hot` |
| confidence | f32 ∈ [0,1] | Default from source: Witnessed=1.0, TalkedWith/Announced=0.8, Overheard=0.6, Rumor{hops}=0.8^hops, Testimony=doc trust_score |

---

### Source (enum)

> ❌ **Audit 2026-04-26:** `Source` enum is **not implemented** in any engine or engine_data file. Provenance tagging (Witnessed/TalkedWith/Overheard/Rumor/Announced/Testimony) is absent.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Provenance tag for a `MemoryEvent`. Determines default confidence and observation feature.

```rust
enum Source {
    Witnessed,              // confidence = 1.0
    TalkedWith(AgentId),    // confidence = 0.8
    Overheard(AgentId),     // confidence = 0.6
    Rumor { hops: u8 },     // confidence = 0.8^hops
    Announced(GroupId),     // confidence = 0.8
    Testimony(ItemId),      // confidence = document.trust_score
}
```

Observation-visible via `info_source_one_hot[5]` (Witnessed/TalkedWith/Overheard/Rumor/NeverMet).

#### Memory.beliefs (Semantic Layer)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| beliefs | Vec<Belief> | Semantic conclusions from events (LocationDangerous/Safe, EntityTrustworthy/Dangerous, SettlementProsperous/Poor, SkillValuable, FactionFriendly/Hostile, Grudge, HeardStory) | belief_formation system (from memory events, social gossip, trade info) | action_eval (destination safety), relationship trust init, goal location bias |

**Belief.confidence** [0.0, 1.0] — decays without update (~0.95/tick absent reinforcement).

---

### Relationship (Per-pair, NPC → NPC)

> ⚠️ **Audit 2026-04-26:** MVP shell only — `trust` (as opaque `valence_q8: i16`), `tenure_ticks`, `other: AgentId` exist. Missing: `familiarity`, `last_interaction` (tenure_ticks is tenure, not last interaction), `perceived_personality`, `believed_knowledge`. Cap is 8 (spec says 20).
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Directional relationship from one NPC toward another (asymmetric).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| trust | f32 | [-1.0, 1.0]. Positive = friendly, negative = hostile. Starts 0.0 | trade +0.1, combat -0.3, betrayal -0.5, rescue +0.3, slow decay | action_eval (trade partner, combat targeting), marriage likelihood |
| familiarity | f32 | [0.0, 1.0]. Grows with proximity and interaction | co-location, trade, work, socializing (+0.02–0.05/interaction) | relationship weight, interaction frequency |
| last_interaction | u64 | Tick of last trade/combat/work together | interaction events | weight decay, "forgotten" threshold |
| perceived_personality | PerceivedPersonality | Mental model of other's traits | observe_action (from witnessed NPC behavior) | compatibility, collaboration prediction, gossip basis |
| believed_knowledge | Bitset<32> | Theory-of-mind: bits self believes other knows, indexed by knowledge domain (Combat, Trade, Family, Politics, Religion, Craft, ...) | set when self witnesses/is told about domain-tagged actions; slow decay | mask predicates (e.g. `Deceive(t) when ¬believed_knowledge(t, Fact::X)`), observation, gossip targeting |

**Capped at 20 relationships per NPC** (evict lowest familiarity if over 20).

#### PerceivedPersonality (Theory of Mind)

> ⚠️ **Audit 2026-04-26:** Feature-gated stub (`#[cfg(feature = "theory-of-mind")]`). `BeliefState` in `engine_data/src/belief.rs` stores observation data (`last_known_pos`, `last_known_hp`, etc.) — **not** the theory-of-mind personality model. Spec's `traits [f32;5]`, `confidence [f32;5]`, and `observation_count` are absent.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| traits | [f32; 5] | Estimated [risk_tol, social, ambition, altruism, curiosity] | observe_action (action→trait signal, alpha learning) | compatibility calc with own personality, behavior prediction |
| confidence | [f32; 5] | Per-trait confidence [0.0, 1.0] | observe_action (+=0.01×attention per action) | trait update alpha (lower confidence → higher alpha), gossip weight |
| observation_count | u32 | Total action observations | observe_action (count++) | authority weight (more observations → less influence) |

**Emergent?** Derived entirely from observed actions. Reconstructible from action history.

---

### Goal & Action Execution

> ❌ **Audit 2026-04-26:** `GoalStack` and `Goal` struct are **not implemented** in the engine SoA. `crates/tactical_sim` has a GOAP `Goal` but it is the tactical planner, not the world-sim goal stack with push/pop priority preemption.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

#### Goal

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| kind | GoalKind | Idle, Work, Eat, Trade{sid}, Fight, Socialize, Rest, Quest{id,pos}, Flee{from}, Build{id}, Haul{commodity,amount,dest}, Relocate{sid}, Gather{commodity,amount} | goal_eval (needs + personality) | action dispatch, movement |
| priority | u8 | Urgency 0–95 (Flee=95, EatCritical=90, Fight=80, Work=50, Rest=20, Idle=0) | goal_eval (from needs urgency) | stack sorting (higher preempts) |
| started_tick | u64 | When pushed | goal_eval push | timeout, stall detection |
| progress | f32 | [0,1] completion | action execution | UI, adaptive urgency on stall |
| target_pos | Option<vec3> | Movement destination | goal_eval, movement (clear on arrival) | pathfinding goal |
| target_entity | Option<u32> | Building/NPC/item/resource ID | goal_eval | work assignment, interaction |
| plan | Vec<PlannedStep> | Multi-step (e.g., Travel → Work → Carry → Rest) | plan generation (action_eval / GOAP) | per-step state machine |
| plan_index | u16 | Active step | step completion (advance_plan) | current_step() for dispatch |

#### GoalStack
`goals: Vec<Goal>` priority-sorted descending, cap 8. Push (dedup+re-sort), pop (completion), remove_kind (override). High-priority goals preempt; on interrupt completion previous resumes (hysteresis).

---

### AgentData

> ❌ **Audit 2026-04-26 — entire sub-struct missing:** ~40 fields below (name, gold, debt, creditor_id, income_rate, economic_intent, price_knowledge, trade_route_id, work_state, behavior_profile, action_outcomes, price_beliefs, cultural_bias, morale, stress, fatigue, loyalty, injury, resolve, archetype, party_id, faction_id, mood, fears, deeds, current_intention, equipped_items, passive_effects, world_abilities, …) **do not exist in `crates/engine`**. The engine carries SoA hot/cold scalars only — there is no per-agent `AgentData` blob. A few related items exist as separate cold fields (`cold_class_definitions`, `cold_creditor_ledger`, `cold_mentor_lineage`).
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Container for all agent-specific state. Carried by every Agent.

#### Identity & Metadata
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| name | String | Generated from entity_id + seed deterministically | spawn | UI, speech, quest text |
| adventurer_id | u32 | Campaign Adventurer link (0 if unlinked) | party entry, class assignment | party queries, persistence |
| creature_type | CreatureType | Citizen (default), PackPredator, Territorial, Raider, Megabeast | constructor, monster spawn | personality preset, ability unlock, gating |

#### Economic & Trade
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| gold | i64 | Liquid currency (signed; debt = negative) | trade, work wages, looting | purchasing power, debt, bribe |
| debt | f32 | Gold owed to creditor | loans, trade deficit | work motivation, bankruptcy |
| creditor_id | Option<u32> | Entity ID of creditor | loans (grant) | collection events, relationship penalty |
| income_rate | f32 | EMA of gold/work cycle (α=0.1) | work completion | credit decision, wage negotiation |
| credit_history | u8 | [0, 255] reliability score | contract success +10, default -5, repayment | bid weight, loan availability |
| economic_intent | EconomicIntent | Produce{commodity,rate}, Trade, Idle, Loot, Haul, Work | goal_eval | action dispatch, building assignment, routing |
| price_knowledge | Vec<PriceReport> | Commodity prices heard via gossip | trade gossip | margin calc, speculation |
| trade_route_id | Option<usize> | Index into WorldState.trade_routes | trade system | repeatable path, profit tracking |
| trade_history | Vec<(u32, u32)> | (destination_sid, success_count) | trade completion | repeat tendency, exploration |

#### Location & Settlement Affiliation
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| home_settlement_id | Option<u32> | Where NPC is "based" for economics | residence, movement | work location, trade origin, relocation |
| home_building_id | Option<u32> | Primary residence (Rest goal target) | housing, family founding | rest location, shelter |
| work_building_id | Option<u32> | Assigned workplace | work assignment | work goal target, state machine |
| inside_building_id | Option<u32> | Currently inside | movement (enter/exit) | interior nav, room access, shelter |
| current_room | Option<u8> | Room index within building.rooms | interiors | room-based AI, visual location |
| home_den | Option<vec3> | Lair (Territorial/PackPredator monsters) | constructor (spawn) | return-home, territorial defense |

#### Movement & Pathfinding Cache
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| cached_path | Vec<(u16, u16)> | Waypoints on settlement grid (col, row) | pathfinding (A* result) | movement (follow path), path expiry on goal change |
| path_index | u16 | Current waypoint index | movement (advance on arrival), goal change (reset) | movement steering, arrival detection |
| goal_stack | GoalStack | Priority-sorted goal list (see GoalStack section) | goal_eval (push/pop), interrupts (prioritize) | agent_inner (execute top goal), action dispatch |

#### Work & Labor State Machine
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| work_state | WorkState | Enum: Idle, Travel, Work, Carry. State machine for work loop | work system (state transition) | work action dispatch, commodity accumulation, haul completion |
| action | NpcAction | Enum: Idle, Walking, Eating, Working{activity}, Hauling, Fighting, Socializing, Resting, Building, Fleeing, Trading, Harvesting | action system (per goal/state) | UI display, animation state, action duration (ticks_remaining) |
| behavior_production | Vec<(usize, f32)> | [(commodity_index, production_rate_per_tick), ...] | work assignment (set per job) | production accumulation, yield bonus, passive income |

#### Skill & Class System
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| class_tags | Vec<String> | Lowercase class names (e.g. ["miner", "farmer"]) | progression (class grant), apprenticeship | ability unlock, behavior profile bias, passive effect generation |
| classes | Vec<ClassSlot> | Acquired classes with levels and XP | progression (level-up from XP), class grant | ability unlock threshold, stat bonus (HP/armor/damage), behavior tag generation |
| behavior_profile | Vec<(u32, f32)> | Sorted tag_hash → accumulated_weight pairs | accumulate_tags per action (all systems), periodic xp_grant (class xp += behavior_profile[tag]) | passive effect calc, action bias weighting, tag lookup (binary search) |

#### Relationships & Social
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| relationships | HashMap<u32, Relationship> | Per-pair directional, capped at 20 | interaction events (modify_relationship) | action_eval (trust), collaboration |
| spouse_id | Option<u32> | Married-to entity ID | family system | family mechanics, shared housing |
| children | Vec<u32> | Child entity IDs | family birth | family tree, inheritance |
| parents | Vec<u32> | Parent entity IDs | family birth, succession | lineage, inherited skills |
| mentor_lineage | Vec<u32> | Mentor chain (most recent first) | apprenticeship | skill inheritance, teaching bonus |
| apprentice_of | Option<u32> | Current mentor entity ID | apprenticeship start | skill acceleration, teaching reputation |
| apprentices | Vec<u32> | Mentee entity IDs | apprenticeship | teaching labor, succession |
| pack_leader_id | Option<u32> | Leader entity ID (PackPredator) | pack formation | regrouping, coordination, morale |

#### Perception & Resource Knowledge
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| known_resources | HashMap<u32, ResourceKnowledge> | Discovered resource entities (pos, type, observation_tick); stale >2000 ticks | exploration scan, gossip | gatherer targeting, trade route planning |
| known_voxel_resources | Vec<VoxelResourceKnowledge> | Ore veins, stone (voxel coords) | voxel exploration, mining gossip | miner targeting, navigation |
| harvest_target | Option<(i32, i32, i32)> | Currently-being-mined voxel (chunk x,y,z) | harvesting system | harvest progress, completion |

#### Psychological & Emotional
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| needs | Needs | 8D vector | agent_inner drift, event bumps | goal_eval, aspiration, action bias |
| memory | Memory | Event log + beliefs | event recording, belief formation | belief-based decisions, relationships, reputation |
| personality | Personality | 5D trait vector | event-driven drift | aspiration weight, action bias, compatibility |
| emotions | Emotions | 6D transient state | events, per-tick decay | action urgency, goal preemption, mood |
| aspiration | Aspiration | Medium-term orientation + optional crystal | agent_inner (500-tick recompute), salient events | action bias, goal selection, concrete pursuit |
| action_outcomes | HashMap<(u8, u32), OutcomeEMA> | Adaptive learning: (action_type, target_type_hash) → EMA | action result (reward/penalty EMA) | action_eval, policy learning |
| price_beliefs | [PriceBelief; NUM_COMMODITIES] | Per-commodity estimated value + confidence | trade experience, decay | trade margin, speculation, gossip |
| cultural_bias | [f32; 12] | Per-action-type conformity bias [-0.3, 0.3] | social_gathering, goal success | action utility weighting |

#### Campaign System Fields
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| morale | f32 | [0, 100] combat morale | combat events, settlement aura, quest completion | combat urgency, retreat threshold |
| stress | f32 | [0, 100] psychological stress | combat damage, near-death, quest failures | focus penalty (morale × (1 - stress/100)) |
| fatigue | f32 | [0, 100] physical fatigue | work, travel, combat | action speed penalty, rest urgency |
| loyalty | f32 | [0, 100] faction/guild loyalty | faction events, betrayal, party failure | faction quest difficulty, behavior gating |
| injury | f32 | [0, 100] incapacity (≥90 → idle only) | combat damage, healing, rest recovery | action capability, combat penalty |
| resolve | f32 | [0, 100] willpower under pressure | fear events, morale pool, leadership aura | flee threshold, action commitment |
| archetype | String | Hero class ("knight", "ranger", "mage", ...) | constructor (template.archetype) | ability unlock, stat scaling |
| party_id | Option<u32> | Party membership | party formation, expedition | party AI, shared objectives, culture drift |
| faction_id | Option<u32> | Faction/guild membership | faction recruitment, succession | faction quests, goals, relationship modifier |
| mood | u8 | Mood discriminant (0=neutral) | mood system | action preference, social openness |
| fears | Vec<u8> | Active fear type indices | phobia system | avoidance behavior, anxiety |
| deeds | Vec<u8> | Legendary deed types earned | deed system | reputation, pride, story arc |
| guild_relationship | f32 | [-100, 100] reputation with guild | guild quests, betrayal | guild benefits, quest priority |

#### Current Action & Intention
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| current_intention | Option<(NpcAction, f32)> | Committed action + utility score at commitment time | action system (action commit with utility), goal change (clear) | action stickiness (hysteresis), action_type_id feedback |
| intention_ticks | u32 | Ticks executing current intention | action system (increment per tick), action reset (clear on change) | action switching penalty, commitment duration (prevents thrashing) |

#### Equipment & Items
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| equipment | Equipment | Legacy quality levels (weapon/armor/accessory) | progression (quality grant), crafting | stat bonus calc (mostly replaced by item entities) |
| equipped_items | EquippedItems | Item entity IDs in slots (weapon/armor/accessory) | equipping system (pick up item), unequipping (drop/swap) | stat bonus lookup (effective_quality × rarity × durability) |

#### Passive & Active Abilities
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| passive_effects | PassiveEffects | Simulation-level bonuses (production_mult, trade_bonus, aura_radius, etc.) | PassiveEffects::compute from behavior_profile (on class level-up) | work production (yield boost), trade (price boost), aura application |
| world_abilities | Vec<WorldAbility> | Parsed DSL abilities usable in world sim | ability parsing (from class_tags), cooldown reset | ability application (rally, fortify, reveal, etc.), cooldown check |

---

### BuildingData

> ❌ **Audit 2026-04-26:** `BuildingData` struct is **not implemented** in the engine crate.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| building_type | BuildingType | Shelter, Farmhouse, Mine, Laboratory, ... (20+ types) | constructor | function, capability unlock |
| settlement_id | Option<u32> | Owning settlement | constructor, settlement system | economic integration, affiliation |
| name | String | Display name | constructor, renaming | UI, narrative |
| grid_col, grid_row | u16 | Position on settlement 128×128 grid | constructor | layout, collision, interior nav |
| footprint_w, footprint_h | u16 | Size in grid cells | constructor | extent, placement collision |
| rooms | Vec<Room> | Interior room definitions | construction growth | NPC occupation, function distribution |
| tier | u8 | Upgrade level | construction tier advancement | capacity/stat scaling |
| resident_ids | Vec<u32> | NPC residents (Shelter type) | housing assignment | count, social location |
| worker_ids | Vec<u32> | NPC workers assigned | work assignment | production, wages, location |
| residential_capacity | u32 | Max residents | type default, upgrades | housing scarcity |
| work_capacity | u32 | Max workers | type default, upgrades | labor bottleneck |
| worker_class_ticks | Vec<...> | Per-class worker tick counts | work system | quality calc, class yields |
| storage | [f32; NUM_COMMODITIES] | Commodity inventory | production, trade, haul | cap check, supply, source |
| storage_capacity | f32 | Total commodity cap | type default, upgrades | overflow, trade limits |
| construction_progress | f32 | [0,1] completion (functional at >0.5) | construction (work ticks += progress) | capability, display |
| built_tick | u64 | When completed | construction completion | age bonuses, decay start |
| builder_id | Option<u32> | Primary builder | construction | reputation, inheritance |
| builder_modifiers | Vec<...> | Builder quality modifiers (tag-based) | construction | production buff, effects |
| owner_modifiers | Vec<...> | Owner quality modifiers (tag-based) | ownership | production buff |
| temporary | bool | Expires at TTL | temporary building system | expiry, permanence |
| ttl_ticks | Option<u64> | Time-to-live (if temporary) | temporary system | expiry trigger |
| specialization_tag | Option<u32> | FNV-1a hash of specialization tag | specialization system | yield bonus, focus |
| specialization_strength | f32 | Bonus strength [0, 2] | specialization | multiplicative yield |

---

### Room (Interior Floor Plan)

> ❌ **Audit 2026-04-26:** `Room` struct is **not implemented** in the engine crate.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

| Field | Type | Meaning |
|---|---|---|
| id | u8 | Unique room ID within building |
| kind | RoomKind | Bedroom, Workshop, Storage, Gathering, Shrine, ... |
| interior_x, interior_y | u16 | Position within building grid |
| interior_w, interior_h | u16 | Size |
| occupants | Vec<u32> | NPCs in room |
| furnishing_level | u8 | Comfort/quality (0–100) |
| blessing | Option<...> | Special effect (e.g., fertility for bedrooms) |

---

### Inventory (Portable Commodity Storage)

> 🤔 **Audit 2026-04-26 (type mismatch):** `Inventory.gold` is `i32` (not `f32` — deliberate i32 for GPU atomics). `commodities` is `[u16; 8]` (not `[f32; 8]`). `capacity: f32` field is absent.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

| Field | Type | Meaning |
|---|---|---|
| commodities | [f32; NUM_COMMODITIES] | Per-commodity quantity (work, eat, trade, haul) |
| gold | f32 | Currency (trade, wages, loot) |
| capacity | f32 | Weight cap (NPC 50.0, building 500.0) |

---

### Derivation Graph

```
PRIMARY: alive, pos, hp, status_effects, needs, personality, emotions, memory.events
SECONDARY (emergent): aspiration.need_vector (recompute 500t), aspiration.crystal,
  perceived_personality, behavior_profile (accumulate_tags), goal_stack priority, PassiveEffects
INFRASTRUCTURE: entity_index, group_index, cached_path, price_knowledge, SimScratch

TRANSFORMATIONS:
- Focus = (avg_needs / 100).clamp(0.5, 1.5)
- Urgency(need) = (100 - need) / 100 * personality[dim]
- Effective Quality = base * rarity_multiplier * durability_fraction
- Action Utility = goal_priority + aspiration_bias + cultural_bias[action] + action_outcomes EMA
```

---

### Notes on Determinism & Serialization

- **hot/cold split**: `Entity` unpacked into `HotEntity` (cache-line scalars) + `ColdEntity` (heap refs); serialized via entity.split() / Entity::from_parts().
- **SimScratch / entity_index / group_index**: `#[serde(skip)]`, rebuilt on load via `rebuild_all_indices()`.
- **Determinism**: all entity randomness via `entity_hash(id, tick, salt)`.
- **Tag hashing**: compile-time FNV-1a `hash(tag_name)` for tag_hash constants.

---

## Aggregate state

> ❌ **Audit 2026-04-26:** This entire section is **not implemented in the engine crate**. Only skeletal `Group` (3 fields) and `Quest` (5–6 fields) exist in `crates/engine/src/aggregate/`. Settlement, RegionState, Faction, GuildState, TradeRoute, ServiceContract, EconomyState, ConstructionMemory, full `Group` (~25 fields), full `Quest` (13 fields), QuestPosting, etc. are absent — they live in legacy `headless_campaign` worktrees not in canonical engine.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

`Group` is the universal social-collective primitive. Settlements, factions, families, guilds, religions, packs, cabals, parties, courts, leagues, monasteries are all `Group` discriminated by `kind`. The Group section at the end gives the canonical shape; per-kind sections describe additional fields.

### Settlement (Group with `kind=Settlement`)

> ❌ **Audit 2026-04-26:** `Settlement` is **not implemented** in the canonical engine crate. Only found in a legacy worktree (`agent-aa2db99f`), not in `crates/engine`.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Political/economic/structural hub. Carries stockpiles, prices, treasury, population, group affiliations, fixed location with facilities.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique settlement ID | spawn/init | all settlement-aware systems |
| name | String | Display name | init | UI, chronicle, logs |
| pos | vec3 | 2D footprint center; `pos.z = surface_height(pos.xy)` | init | all systems using location |
| grid_id | Option<u32> | NavGrid this settlement owns | init | navigation, pathfinding |
| specialty | SettlementSpecialty | Production/npc focus | init | production, npc_spawn, resource nodes |
| treasury | f32 | Gold reserves | taxes, economy, contracts, looting, conquest | economy, contracts, warfare, bankruptcy |
| stockpile | [f32; 8] | Commodity reserves (FOOD, WOOD, IRON, COPPER, HERBS, CRYSTAL, HIDE, MEDICINE, EQUIPMENT) | production, consumption, trade, looting | economy, npc_decisions, trade |
| prices | [f32; 8] | Local market prices per commodity | trade_goods, price_controls, seasons, debasement | economy, trades, npc_decisions, arbitrage |
| population | u32 | Count of alive NPCs with home_settlement_id == id | npc birth/death, migration | economy scaling, consumption, threat |
| faction_id | Option<u32> | Owning faction, if controlled | conquest, civil_war, diplomacy | faction_ai, warfare, taxes |
| threat_level | f32 | Danger score (0–1) | monster_density, quest_posting, attacks | quest urgency, recruitment, upgrades |
| infrastructure_level | f32 | Building/defense tier (0–5) | construction, upgrades | defense, garrison capacity |
| context_tags | Vec<(u32, f32)> | Contextual action modifiers (plague, festival, war) | events (crisis, seasons, diplomacy) | action_system, skill resolution |
| treasury_building_id | Option<u32> | Entity ID of Treasury building | init (ensure_treasury_buildings) | resource movement, gold transfers |
| service_contracts | Vec<ServiceContract> | Active contracts | npc_decisions, contract_lifecycle | contract resolution |
| construction_memory | ConstructionMemory | Per-settlement building event history | building_ai event logging | building_ai pattern learning |

---

### RegionState

> ❌ **Audit 2026-04-26:** `RegionState` is **not implemented** in the engine crate.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Large map areas with terrain, monster populations, faction control, dungeon sites.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique region ID | spawn | all systems referencing regions |
| name | String | Display name | spawn | UI, chronicle |
| pos | vec3 | World-space center; `z = sea_level` | spawn | region queries, travel distance |
| terrain | Terrain | Biome type (see Terrain Enum) | spawn | resource spawning, threat, travel |
| sub_biome | SubBiome | Variant (see SubBiome Enum) | spawn | resource yields, threat mods, travel |
| elevation | u8 | Height tier (0–4: sea, foothills, highlands, peaks, sky) | spawn | resource rarity, threat, building placement |
| monster_density | f32 | Local spawn rate multiplier | random_events, spawning, scaling | encounter gen, threat_level calc |
| threat_level | f32 | Aggregate danger (0–1) | monster density, battles, decay | threat_reports, quest_posting, npc decisions |
| has_river | bool | River flows through | spawn | river_travel, trade routes |
| has_lake | bool | Lake exists | spawn | water resources, water travel |
| is_coastal | bool | Borders ocean | spawn | coastal trade, sea encounters |
| dungeon_sites | Vec<DungeonSite> | Procedural dungeon entrances | dungeon_discovery, exploration | quest_posting, adventuring |
| neighbors | Vec<u32> | 4-connected grid neighbor IDs | spawn | travel, movement, region_graphs |
| river_connections | Vec<u32> | River-connected region IDs | spawn | river_travel, trade routes |
| is_chokepoint | bool | Only 1–2 passable neighbors | spawn | warfare, blockade tactics |
| is_floating | bool | FlyingIslands terrain | spawn | access control, encounters |
| faction_id | Option<u32> | Controlling faction | conquest, civil_war | faction_ai, warfare, taxes |
| control | f32 | Faction control strength (0–1) | diplomacy, conquest, unrest | faction power assessment |
| unrest | f32 | Civil unrest (0–1) | diplomacy, oppression, riots | civil_war triggers, loyalty |

---

### Faction (Group with `kind=Faction`)

> ❌ **Audit 2026-04-26:** `Faction` is **not implemented** in the engine crate.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Macro-political collective (governments, military powers, organized cults). Carries military_strength, standings, tech_level, governance.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique faction ID | spawn | all faction-aware systems |
| name | String | Faction name | init | UI, chronicle |
| relationship_to_guild | f32 | Player-guild relation (-100..100) | diplomacy, quest_completion, betrayals | quest_board filtering, npc morale |
| at_war_with | Vec<u32> | Faction IDs in open warfare | civil_war, diplomacy | warfare, conquest, movement |
| military_strength | f32 | Current fighting capacity | recruitment, training, losses, desertion | conquest, diplomacy, battle strength |
| max_military_strength | f32 | Theoretical max from population | population growth, tech_level | scaling, recruitment limits |
| territory_size | u32 | Regions/settlements controlled | conquest, loss_of_control | power, tax base, recruitment |
| treasury | f32 | Gold reserves | taxes, war loot, quests, reparations | military, mercenary, diplomacy |
| diplomatic_stance | DiplomaticStance | Friendly/Neutral/Hostile/AtWar/Coalition | diplomacy system | npc decisions, interaction modifiers |
| coup_risk | f32 | Stability (0–1, higher = upheaval risk) | oppression, morale, succession | faction_ai decisions, succession crises |
| escalation_level | u32 | Conflict intensity (0–5) | warfare, diplomacy escalation | warfare intensity cap, treaty enforcement |
| tech_level | u32 | Research tier | research, quests | military bonus, production, ability unlock |
| recent_actions | Vec<String> | Bounded log of recent events | all major systems | narrative, NPC knowledge, diplomacy memory |

---

### GuildState

> ❌ **Audit 2026-04-26:** `GuildState` is **not implemented** in the engine crate.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Player faction state (independent from NPC factions).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| gold | f32 | Guild treasury | quest_rewards, missions, expenses | quest_board filtering, upgrades |
| supplies | f32 | Supply reserve | production, usage, looting | mission readiness |
| reputation | f32 | Fame/standing (0–100) | quests, battles, diplomacy | quest_board filtering, npc_recruitment, prices |
| tier | u32 | Guild level (0–5, unlocks features) | reputation milestones | mission gen, capacity |
| credit_rating | f32 | Borrowing capacity (0–100) | loans, repayment | loan eligibility, interest |
| active_quest_capacity | u32 | Max simultaneous active quests | tier unlocks | quest acceptance |

---

### Quest & Quest Lifecycle

> ⚠️ **Audit 2026-04-26:** `Quest` is partial — engine has 6 fields (`seq`, `poster`, `category`, `resolution`, `acceptors`, `posted_tick`) vs spec's 13. Missing: name, destination, progress, status, accepted_tick, deadline_tick, threat_level, reward_gold, reward_xp. `QuestPosting` is entirely absent. `QuestCategory` is a 5-bucket enum vs spec's 10-type `QuestType`.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

#### Quest (Active)

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique quest ID | quest_board acceptance | party, quest_lifecycle |
| name | String | Quest title | posted | UI |
| quest_type | QuestType | See QuestType enum | posted | party decisions, threat calc |
| party_member_ids | Vec<u32> | Assigned party entity IDs | acceptance, deaths, replacements | party_ai, completion checks |
| destination | vec3 | Objective location | posted | pathfinding, progress |
| progress | f32 | Completion ratio (0–1) | quest_lifecycle | UI, completion |
| status | QuestStatus | See QuestStatus enum | quest_lifecycle | transitions, rewards |
| accepted_tick | u64 | When accepted | acceptance | deadline, duration |
| deadline_tick | u64 | Tick deadline (0 = none) | posted | failure condition, reward scaling |
| threat_level | f32 | Difficulty (0–1) | posted | party composition, reward scaling |
| reward_gold | f32 | Gold on completion | posted | motivation, board display |
| reward_xp | u32 | XP on completion | posted | party leveling |

#### QuestType / QuestStatus
`QuestType`: `Hunt`, `Escort`, `Deliver`, `Explore`, `Defend`, `Gather`, `Rescue`, `Assassinate`, `Diplomacy`, `Custom`.
`QuestStatus`: `Traveling`, `InProgress`, `Completed`, `Failed`, `Returning`.

#### QuestPosting (Board)
| Field | Type | Meaning |
|---|---|---|
| id | u32 | Unique posting ID |
| name | String | Title |
| quest_type | QuestType | Category |
| settlement_id | u32 | Posting settlement |
| destination | vec3 | Objective |
| threat_level | f32 | Difficulty |
| reward_gold | f32 | Bounty |
| reward_xp | u32 | Experience |
| expires_tick | u64 | When posting removed |

---

### TradeRoute

> ❌ **Audit 2026-04-26:** `TradeRoute` is **not implemented** in the engine crate.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Emergent trade connection from repeated profitable NPC trading.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| settlement_a, settlement_b | u32 | Endpoints | npc_trade | trade_logic, decay |
| established_tick | u64 | When created | npc_trade | history |
| total_profit | f32 | Cumulative gold profited | npc_trade | route viability |
| trade_count | u32 | Successful trades | npc_trade | route activity |
| strength | f32 | Health (0–1, decays without activity, abandoned < 0.1) | npc_trade, decay | npc_decisions, discovery |

---

### ServiceContract

> ❌ **Audit 2026-04-26:** `ServiceContract` is **not implemented** in the engine crate.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

NPC service request (build, gather, craft, heal, guard, haul, teach, barter).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| requester_id | u32 | NPC posting the contract | npc_decisions | contract_lifecycle |
| service | ServiceType | Work type (see ServiceType enum) | npc_decisions | contractor matching |
| max_payment | Payment | Max NPC will pay | npc_decisions, morale | bid filtering |
| payment | f32 | Agreed payment (resolution) | contract_resolution | contractor reward |
| provider_id | Option<u32> | Accepting NPC | acceptance | contract_lifecycle, reputation |
| posted_tick | u64 | When posted | posting | age for cleanup |
| completed | bool | Done | contract_resolution | archive |
| bidding_deadline | u64 | Tick bidding closes (urgency-driven: crit +5, high +15, med +30, low +100) | posting | bid window |
| bids | Vec<ContractBid> | Offers (bidder_id, bid_amount, skill_value, credit_history) | npc_decisions | winner selection |
| accepted_bid | Option<usize> | Index of accepted offer | contract_resolution | payment |

---

### PriceReport
`settlement_id: u32`, `prices: [f32; 8]`, `tick_observed: u64`. Snapshot for arbitrage / trend.

---

### SettlementSpecialty Enum
`General`, `MiningTown` (iron/crystal), `TradeHub` (price discovery), `MilitaryOutpost` (iron/equipment), `FarmingVillage` (food/hide), `ScholarCity` (herbs/medicine, XP bonus), `PortTown` (food/wood, coastal), `CraftingGuild` (equipment/medicine).

---

### DungeonSite
| Field | Type | Meaning |
|---|---|---|
| pos | vec3 | Entrance location |
| name | String | Procedural name |
| explored_depth | u8 | Deepest level reached (0 = entrance only) |
| max_depth | u8 | Total levels |
| is_cleared | bool | Fully cleared (no respawn) |
| last_explored_tick | u64 | When last entered |
| threat_mult | f32 | Danger modifier from terrain + depth |

---

### DiplomaticStance Enum
`Friendly` (+morale, trade bonuses), `Neutral`, `Hostile` (-morale, embargo possible), `AtWar` (open conflict), `Coalition` (shared military).

---

### Terrain Enum (17 types)
Biome type carrying resource yields, threat, travel speed, elevation tier: `Plains`, `Forest`, `Mountains`, `Coast`, `Swamp`, `Desert`, `Tundra`, `Volcano`, `DeepOcean`, `Jungle`, `Glacier`, `Caverns`, `Badlands`, `FlyingIslands`, `DeathZone`, `AncientRuins`, `CoralReef`. Per-variant multipliers in `terrain.rs`.

---

### SubBiome Enum (16 types)
Variants within biome: `Standard`, `LightForest`, `DenseForest`, `AncientForest`, `SandDunes`, `RockyDesert`, `HotSprings`, `GlowingMarsh`, `TempleJungle`, `NaturalCave`, `LavaTubes`, `FrozenCavern`, `MushroomGrove`, `CrystalVein`, `Aquifer`, `BoneOssuary`. Multipliers stack with parent terrain.

---

### EconomyState

> ❌ **Audit 2026-04-26:** `EconomyState` is **not implemented** in the engine crate.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

`total_gold_supply: f32` (sum of all gold), `total_commodities: [f32; 8]` (supply per commodity). Drives inflation, scarcity, price scaling.

---

### ChronicleEntry & WorldEvent

> ⚠️ **Audit 2026-04-26:** DSL-emitted `ChronicleEntry` has only 4 fields (`template_id, agent, target, tick`) — missing `category`, `text` (human-readable narrative), `entity_ids`. `WorldEvent` exists as ~30 fine-grained individual event structs in `engine_data/src/events/` but **no unified `WorldEvent` enum** with 13 variants.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

`ChronicleEntry { tick, category, text, entity_ids }` — bounded narrative log. Categories: Battle, Quest, Diplomacy, Economy, Death, Discovery, Crisis, Achievement, Narrative.

`WorldEvent` (13 variants): `Generic{category,text}`, `EntityDied`, `QuestChanged`, `FactionRelationChanged`, `RegionOwnerChanged`, `BondGrief`, `SeasonChanged`, `BattleStarted`, `BattleEnded`, `QuestPosted`, `QuestAccepted`, `QuestCompleted`, `SettlementConquered`.

---

### WorldState Top-Level Collections

> ⚠️ **Audit 2026-04-26:** `regions`, `groups` (3-field stub), `quests` (partial), and `quest_board` are the only partially-present collections. `trade_routes`, `economy`, `adventurer_bonds`, `guild`, `relations`, `prophecies` are **absent** from the engine crate. These collections live in the `headless_campaign`/`bevy_game` layer.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Spatial/cache fields (`tick`, `rng_state`, indices, `tiles`, `voxel_world`, `nav_grids`, `surface_cache`, `surface_grid`, `cell_census`, `sim_scratch`, `build_seeds`, `structural_events`, `region_plan`, `chronicle`, `world_events`, `fidelity_zones`) are documented in the World state section below.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| regions | Vec<RegionState> | Regional population, faction control, terrain | spawn | all region-aware systems |
| groups | Vec<Group> | All social-collective state, kind-discriminated | spawn, JoinGroup/LeaveGroup, FoundGroup, DissolveGroup, conquest, diplomacy | every group-aware system |
| quests | Vec<Quest> | Active quests being pursued | quest_board acceptance | quest_lifecycle, party_ai |
| quest_board | Vec<QuestPosting> | Available quests not yet accepted | settlement/threat posting | guild quest selection, expiration cleanup |
| trade_routes | Vec<TradeRoute> | Emergent trading paths (strength decays) | npc_trade | npc_decisions, trade analysis |
| economy | EconomyState | Global totals (gold supply, commodity sum) | all economic systems | inflation, scarcity calc |
| adventurer_bonds | HashMap<(u32, u32), f32> | NPC-to-NPC bond strength (0–100) | relationships, quests, deaths | morale, party cohesion, grief |
| guild | GuildState | Player faction state | quest completion, events | quest board filtering, upgrades |
| relations | HashMap<(u32, u32, u8), f32> | Entity-to-entity relations by kind (relationship, bond, romance, rivalry, grudge, mentorship) | interactions, events | npc decisions, morale, romance |
| prophecies | Vec<Prophecy> | Generated at init, fulfilled by events | init, prophecy system | narrative hooks |
| registry | Option<Arc<Registry>> | Data-driven entity/ability/item definitions | init | entity spawning, ability execution |
| skip_resource_init | bool | Skip resource node spawning (building-AI scenarios) | init | resource_nodes system |

---

### Derivation Graph

```
DERIVABLE FROM ENTITY STATE (cached for perf):
- population[sid]            ← count(entities where home_settlement_id == sid && alive)
- threat_level[region]       ← (monster_density * count + recent_attacks + recent_deaths) / (1 + time_since_crisis)
- threat_level[settlement]   ← from region.threat + local entity queries
- is_at_war[a,b]             ← faction.at_war_with list (vs O(n²) relation scan)

EMERGENT (require history):
- prices                     ← base / (1 + stockpile / (pop * halflife))
- strength[trade_route]      ← exponential decay without activity, abandoned < 0.1
- trade_count, total_profit  ← accumulated by npc_trade

CACHED, REGENERABLE (pure funcs):
- region_plan                ← seed
- surface_cache, surface_grid← (vx, vy, region_plan, seed)
- nav_grids                  ← voxel_world
- cell_census                ← chunk contents in cell band
- entity_index/settlement_index/group_index ← rebuilt from entities/settlements
```

---

### ConstructionMemory

Three-tier per-settlement construction event log. Updated by building_ai event logging; read by pattern learning.

| Tier | Field | Capacity | Decay |
|---|---|---|---|
| short_term | RingBuffer<ConstructionEvent> | 64 | None (circular) |
| medium_term | RingBuffer<AggregatedPattern> (importance > 0.3) | 256 | Halves every 500 ticks |
| long_term | RingBuffer<StructuralLesson> (importance > 0.7) | 64 | Permanent until contradicted |

---

### Payment, ContractBid, ServiceType

**Payment**: `gold: f32`, `commodities: Vec<(u8, f32)>`. Methods: `gold_only()`, `commodity()`, `estimated_value()`, `is_empty()`.

**ContractBid**: `bidder_id`, `bid_amount: f32`, `skill_value: f32`, `credit_history: u8 (0–100)`.

**ServiceType**: `Build(BuildingType)`, `Gather(commodity_idx, amount)`, `Craft`, `Heal`, `Guard(target_id)`, `Haul(commodity_idx, amount, (x,y))`, `Teach(npc_id)`, `Barter { offer, want }`.

---

### Group (universal)

> ⚠️ **Audit 2026-04-26:** MVP stub only — engine's `Group` has 3 fields (`kind_tag: u32, members: SmallVec<[AgentId;8]>, leader: Option<AgentId>`). The ~25 fields across Identity/Leadership/Resources/Standings/Recruitment/Activity documented below are **absent**.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Social-collective primitive — any collection of agents with shared identity. Kind + optional field presence differentiates factions, families, guilds, religions, packs, cabals, parties, settlements, leagues, monasteries, courts.

#### Identity & Membership
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | GroupId | Unique group ID | spawn / FoundGroup event | all group-aware systems |
| kind | GroupKind | Faction, Family, Guild, Religion, Pack, Party, Settlement, Court, Cabal, Other | spawn | mask predicates, observation features |
| name | String | Display name | init / RenameEvent | UI, chronicle, narrative |
| founded_tick | u64 | When formed | FoundGroup event | tenure, history derivations |
| dissolved_tick | Option<u64> | When (or if) dissolved | DissolveGroup event | active filter, narrative |
| members | Vec<AgentId> | Materialized list of agents whose memberships include this group | derived view from agent.memberships (or cached) | quest party expansion, observation, mask |
| founder_ids | Vec<AgentId> | Original founders (immutable) | spawn | history, narrative |

#### Leadership / Governance
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| leader_id | Option<AgentId> | Current leader (None for headless groups like ad-hoc parties) | succession / promotion events | mask predicates (leader actions), observation |
| leadership_chain | Vec<AgentId> | Ordered succession queue (for groups with formal succession) | succession events | succession resolution |
| governance | GovernanceKind | Hereditary, Elective, Council, Theocratic, Ad-hoc, ... | init / charter changes | succession mechanism, mask |
| charter | Option<CharterRef> | Settlement-only: governing charter | charter quests | tax rules, eligibility, recruitment |

#### Resources & Capacity
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| treasury | f32 | Group gold reserves | tax events, contract payouts, war loot | mask (can-afford predicates), action eval |
| stockpile | Option<[f32; NUM_COMMODITIES]> | Bulk material stores (settlements, guilds with inventory) | production/consumption events | mask (can-fulfill predicates), trade |
| facilities | Option<FacilityRef> | Settlements only: production buildings, infrastructure links | construction events | production eligibility, settlement bonuses |
| military_strength | Option<f32> | For groups that field forces (factions, packs, mercenary guilds) | recruitment, casualties, training | war predicates, mask, observation |

#### Standings (relations to other groups)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| standings | Map<GroupId, Standing> | Per-other-group disposition: Allied/Neutral/Tense/AtWar/Vassal/Suzerain/Excommunicate | diplomacy events (AllianceFormed, WarDeclared, VassalSworn, ...) | mask predicates (can-attack-other-group, trade-with), observation, is_hostile derivation |
| standing_history | Vec<StandingEvent> | Bounded history of standing changes | diplomacy events | narrative, AI memory |

#### Recruitment & Eligibility
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| recruitment_open | bool | Whether agents can JoinGroup unsolicited | charter / leadership decisions | mask (JoinGroup) |
| eligibility_predicate | EligibilityRef | Reference to the rule(s) for membership eligibility (e.g. "must be Smith family", "must complete oath quest", "must be hostile-creature-type") | charter / init | mask (JoinGroup) |
| dues | Option<DuesSpec> | Recurring contribution required of members | charter | tax events, mask (insolvency check) |

#### Activity
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| active_quests | Vec<QuestId> | Quests this group has Posted or Accepted as a party | PostQuest / AcceptQuest cascade | quest lifecycle, observation |
| recent_activity | Vec<EventRef> | Bounded log of recent group-level events | various | narrative, observation, AI memory |

**Derived views (not stored):** `population` (`members.len()`), `is_at_war(other)` / `is_allied(other)` (read `standings[other]`), `wealth_per_member` (`treasury / members.len()`), `cultural_descriptor` (aggregate behavior_profiles), `reputation_among(other_group)` (cross-group event history).

#### Per-kind shapes
| Kind | Typical fields |
|---|---|
| Faction | `military_strength`, `standings`, `governance`, `tech_level`, recruitment_open |
| Settlement | `facilities`, `charter`, `stockpile`, `treasury`; paired with spatial record (`pos`, `grid_id`, `region_id`) |
| Guild | `treasury`, `recruitment_open`, `dues`, `eligibility_predicate`, settlement-bound charter |
| Family | `leader_id` (household head), `members` from kin events |
| Party | `leader_id`, `founded_tick`, `dissolved_tick` |
| Religion | `charter` (scripture/pantheon), `eligibility_predicate`, leadership chain |
| Pack | `leader_id` (alpha), `eligibility_predicate` keyed on `creature_type` |
| Cabal | `recruitment_open=false`, restrictive eligibility, secret standings |
| Court | `governance`, `leadership_chain` |
| League | flat governance, `standings` heavy |

---

## World state

> ❌ **Audit 2026-04-26:** This entire section is **not implemented in the engine crate**. VoxelWorld, RegionPlan, NavGrid, FidelityZone, BuildSeed, StructuralEvent, SimScratch, GroupIndex, etc. are absent from `crates/engine`. They live in `tactical_sim`/`bevy_game` layers. `WorldState.tick` is `u32` in implementation, not `u64` as spec'd. Several hot fields exist on agents that this catalog doesn't list (`hot_engaged_with`, `hot_stun_expires_at_tick`, `hot_slow_expires_at_tick`, `hot_slow_factor_q8`, `hot_cooldown_next_ready_tick`, `ability_cooldowns`).
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

Environment layer: terrain, tiles, voxels, spatial indices, shared caches. Not per-entity, not per-settlement.

---

### WorldState top-level fields (world/spatial only)

> ⚠️ **Audit 2026-04-26 (spec mismatch):** `SimState.tick` is `u32` (spec says `u64`). `rng_state` is not separately tracked — `seed: u64` is used directly. `next_id` is managed via `AgentSlotPool`, not as a named scalar field.
> See `docs/superpowers/notes/2026-04-26-audit-state.md` for detail.

#### Scalar / identity (Primary)
| Field | Type | Meaning |
|---|---|---|
| tick | u64 | Monotonic tick counter (incremented by `WorldSim::tick`) |
| rng_state | u64 | PCG-style RNG state; sole randomness source (`next_rand_u32`) |
| next_id | u32 | Monotonic entity ID counter |
| max_entity_id | u32 | Highest ID seen (sizes `entity_index`); derived from rebuild |

#### Derived indices (`#[serde(skip)]`, rebuilt on load)
| Field | Type | Purpose |
|---|---|---|
| entity_index | `Vec<u32>` | `id → idx` into entities/hot/cold (size `max_entity_id+1`, sentinel `u32::MAX`) |
| group_index | `GroupIndex` | Contiguous per-settlement + per-party entity ranges |
| settlement_index | `Vec<u32>` | `settlement_id → idx` into settlements |

All derived from `entities`/`settlements`; avoid linear scans.

#### Spatial / terrain state (Primary unless noted)
| Field | Type | Purpose |
|---|---|---|
| tiles | `HashMap<TilePos, Tile, ahash>` | Sparse 2D tile grid |
| fidelity_zones | `Vec<FidelityZone>` | Sim-fidelity bubbles (entity_ids membership recomputed each tick) |
| build_seeds | `Vec<BuildSeed>` | Pending room-growth seeds |
| voxel_world | `VoxelWorld` | 3D chunked voxel world (physical truth) |
| nav_grids | `Vec<NavGrid>` | Baked 2D walkable surfaces (derived cache) |
| region_plan | `Option<Arc<RegionPlan>>` | Biome/elevation plan; `#[serde(skip)]`, regen from seed |
| structural_events | `Vec<StructuralEvent>` | Per-tick collapse/fracture buffer |
| chronicle | `Vec<ChronicleEntry>` | Narrative log (bounded ring) |
| world_events | `Vec<WorldEvent>` | Recent events (bounded) |

#### Cache fields (all `#[serde(skip)]`)
| Field | Type | Rebuilt by |
|---|---|---|
| surface_cache | `HashMap<u64, i32>` | `scan_voxel_resources_cached` (lazy) |
| surface_grid | `FlatSurfaceGrid` | `warm_surface_cache` |
| cell_census | `HashMap<(i32,i32),[u32;6]>` | `scan_voxel_resources_cached` (lazy) |
| sim_scratch | `SimScratch` | Caller clears before each use (NOT persistent) |

---

### WorldState.tiles

**Type:** `HashMap<TilePos, Tile, ahash::RandomState>`
**Purpose:** Sparse 2D grid of placed tiles (floors, walls, doors, furniture, farmland, workspaces, ditches). Only modified tiles stored. 2.0 world units per tile.

#### Tile fields
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tile_type | TileType | What kind of tile | construction (wall/door placement), action_eval (PlaceTile) | flood_fill, interiors, buildings, pathing, movement_cost |
| placed_by | Option\<u32\> | Entity that placed it | construction, action_eval | rarely queried |
| placed_tick | u64 | Tick placed | construction | decay/age checks |
| floor_level | u8 | Floor index in multi-story building (0=ground, outdoor=0). Multi-story buildings store one `Tile` per (TilePos, floor_level) | construction (per-floor) | interior nav (`floor_height(pos, building)`), rendering, mask predicates (overhear: same building → any floor) |

#### TileType variants
- **Terrain:** `Dirt`, `Stone`, `Water`
- **Structural:** `Floor(TileMaterial)`, `Wall(TileMaterial)`, `Door`, `Window`
- **Infrastructure:** `Path`, `Bridge`, `Fence` (blocks monsters only)
- **Agricultural:** `Farmland`
- **Furniture:** `Workspace(WorkspaceType)`, `Bed`, `Altar`, `Bookshelf`, `StorageContainer`, `MarketStall`, `WeaponRack`, `TrainingDummy`, `Hearth`
- **Defensive:** `Moat`, `TowerBase`, `GateHouse`, `ArcherPosition`, `Trap`

Per-variant role via `movement_cost()`, `is_solid()`, `is_wall()`, `is_floor()`, `is_furniture()`, `blocks_monsters_only()`.

#### TileMaterial: `Wood`, `Stone`, `Iron`.
#### WorkspaceType: `Forge`, `Anvil`, `Loom`, `AlchemyBench`, `Kitchen`, `Sawbench`.
#### BuildingFunction: `Shelter`, `Production`, `Worship`, `Knowledge`, `Defense`, `Trade`, `Storage`. Used by `BuildSeed.intended_function` and `minimum_interior()`.

#### Characteristics
- Sparse writes (PlaceTile / construction room-growth). Primary state.
- GPU-hostile: HashMap lookup per neighbor in flood_fill. Candidate for flat-grid conversion.

#### TilePos
`TilePos { x: i32, y: i32 }` with hand-packed `Hash` (single `write_u64`). Helpers: `from_world`, `to_world`, `neighbors8()`, `neighbors4()`.

---

### WorldState.voxel_world (VoxelWorld)

**Type:** `VoxelWorld`
**Purpose:** 3D chunked voxel world — physical source of truth. Sparse, only loaded chunks stored.

#### VoxelWorld fields
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| chunks | `HashMap<ChunkPos, Chunk, ahash>` (type alias `ChunkMap`) | Sparse chunk storage | `generate_chunk`, `set_voxel`, `damage_voxel`, `mine_voxel`, `remove_box/sphere`, `fill_box/sphere`, `replace_in_box` | `get_voxel`, surface_height chunk path, structural_tick, exploration scan, voxel_construction, voxel_harvest |
| sea_level | i32 | Global water z-level (default 350 at 0.10 VOXEL_SCALE = 35m) | Init only | terrain gen, surface_height fallback, flood detection |
| region_plan | `Option<RegionPlan>` | Biome plan driving materializer | Init only | `surface_height` fast path, `generate_chunk` |

#### Constants (constants.rs)
| Constant | Value | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 64 | Voxels per chunk edge |
| `CHUNK_VOLUME` | 262,144 | `CHUNK_SIZE³` — voxels per chunk |
| `VOXEL_SCALE` | 0.10 | Meters per voxel |
| `MAX_SURFACE_Z` | 2000 | Upper bound on terrain Z (for surface scans) |
| `SEA_LEVEL` | 350 | Default water level in voxel-z |
| `MEGA` | (chunks-per-mega-axis) | Mega-chunk grouping (rendering) |

#### ChunkPos
`{ x, y, z: i32 }` — chunk-space coord (each chunk covers CHUNK_SIZE³ voxels). Uses derive-Hash + ahash.

#### Chunk
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| voxels | `Vec<Voxel>` (len = CHUNK_VOLUME = 262,144) | Dense voxel array, row-major `(z·CS + y)·CS + x` | `Chunk::set`, VoxelWorld mutations | renderer, structural_tick, scans |
| pos | ChunkPos | Identity | Constructor | mesh gen |
| dirty | bool | Any voxel changed — regen SDF/mesh | mutations; cleared by structural_tick | mesh regen, structural_tick |

#### Voxel
`#[repr(C)]`, ~16 bytes/voxel.
| Field | Type | Meaning |
|---|---|---|
| material | VoxelMaterial (u8 repr) | Material enum |
| light | u8 | Light level 0–15 |
| damage | u8 | Mining damage (breaks when ≥ hardness) |
| flags | u8 | bits 0–3: water level; 4–5: flow direction; 6: is_source; 7: is_support |
| integrity | f32 | Structural health [0,1]. 1.0=intact, 0.0=collapsed (load-bearing voxels preserve material at 0.0) |
| building_id | Option\<u32\> | Building this voxel belongs to, if any |
| zone | VoxelZone | Residential/Commercial/Industrial/Military/Agricultural/Sacred/Underground/None |

`effective_hp() = integrity * material.properties().hp_multiplier`. `damage_voxel` at ≤0 HP sets integrity=0 (load-bearing) or replaces with Air, then triggers `cascade_collapse` upward.

#### VoxelMaterial variants
~45 variants: Air, natural terrain (Dirt/Stone/Granite/Sand/Clay/Gravel/Grass), fluids (Water/Lava/Ice/Snow), ores (IronOre/CopperOre/GoldOre/Coal/Crystal), placed (WoodLog/WoodPlanks/StoneBlock/StoneBrick/Thatch/Iron/Glass), agricultural (Farmland/Crop), additional (Basalt/Sandstone/Marble/Bone/Brick/CutStone/Concrete/Ceramic/Steel/Bronze/Obsidian), biome surfaces (JungleMoss/MudGrass/RedSand/Peat/TallGrass/Leaves), entity markers (non-solid, rendering only).

Each material: `is_solid()`, `is_fluid()`, `is_transparent()`, `hardness()`, `mine_yield()`, `properties()` (hp_multiplier, fire_resistance, load_bearing, weight, rubble_move_cost, construction_cost, blast_resistance).

#### VoxelZone
`None`, `Residential`, `Commercial`, `Industrial`, `Military`, `Agricultural`, `Sacred`, `Underground`.

#### Surface-height paths

`surface_height: vec2 → f32` (outdoor): topmost solid voxel surface z. Interior nav uses `floor_height: (vec3, building_id) → f32`.

Three code paths, priority order:
1. **Analytical fbm** (`surface_height_at(vx, vy, plan, seed)`) — when `region_plan` is `Some`. Pure function; zero chunk lookups.
2. **Chunk-walking fallback** (`surface_height_from_chunks`) — walks chunk-z-slices top-down; returns `sea_level` if no solid.
3. **Cached** (`surface_grid` dense → `surface_cache` sparse) — used by exploration scans.

The ground-snap cascade reads `surface_height(pos.xy)` outdoor or `floor_height(pos, building_id)` indoor, then sets `pos.z = h + creature_height/2`.

#### Characteristics
- Chunks added lazily (`generate_chunk`/`ensure_loaded_around`). Voxels mutated by harvest, construction, structural_tick, damage.
- GPU: chunks are inherently chunked; HashMap wrapper is hostile. Dense `Vec<Voxel>` per chunk is GPU-upload-ready (16 B/voxel, 4 MB/chunk).
- Primary state.

**3D agent positions.** `Agent.pos: vec3` is authoritative. Volumetric `creature_type` (Dragon, Fish, Bat) place anywhere in voxel grid without indoor/outdoor flag. Ground-locked resolve `pos.z` via snap cascade.

---

### WorldState.region_plan (RegionPlan)

**Type:** `Option<RegionPlan>` in WorldState; also stored by value in `VoxelWorld.region_plan`.
**Purpose:** Continental-scale biome plan — grid of `RegionCell`s + polyline rivers + road segments. Referenced by chunk generation.

#### RegionPlan fields
| Field | Type | Meaning |
|---|---|---|
| cols, rows | usize | Grid dimensions |
| cells | `Vec<RegionCell>` | Row-major biome grid |
| rivers | `Vec<RiverPath>` | Polyline rivers in voxel-space |
| roads | `Vec<RoadSegment>` | Road segments in voxel-space |
| seed | u64 | Worldgen seed (determinism anchor) |

#### RegionCell (region_plan.rs:52)
| Field | Type | Meaning |
|---|---|---|
| height | f32 | Normalised elevation [0,1] |
| moisture | f32 | Normalised moisture [0,1] |
| temperature | f32 | Normalised temperature [0,1] |
| terrain | Terrain | Biome classification (Plains, Forest, Mountains, Coast, Swamp, Desert, Tundra, Volcano, DeepOcean, Jungle, Glacier, Caverns, Badlands, FlyingIslands, DeathZone, AncientRuins, CoralReef) |
| sub_biome | SubBiome | Variant within biome (Standard, LightForest, DenseForest, AncientForest, SandDunes, RockyDesert, HotSprings, GlowingMarsh, TempleJungle, …) |
| settlement | `Option<SettlementPlan>` | Planned settlement in this cell |
| dungeons | `Vec<DungeonPlan>` | Planned dungeons |
| has_road | bool | Whether a road passes through |

#### SettlementPlan (region_plan.rs:12)
| Field | Type | Meaning |
|---|---|---|
| kind | SettlementKind | `Town` / `Castle` / `Camp` / `Ruin` |
| local_pos | (f32, f32) | Normalised position within cell [0,1) |

#### DungeonPlan (region_plan.rs:27)
| Field | Type | Meaning |
|---|---|---|
| local_pos | (f32, f32) | Normalised position within cell |
| depth | u8 | Dungeon depth tier |

#### RiverPath (region_plan.rs:35)
| Field | Type | Meaning |
|---|---|---|
| points | `Vec<(f32, f32)>` | Polyline in voxel-space |
| widths | `Vec<f32>` | Per-vertex widths (parallel array to points) |

#### RoadSegment (region_plan.rs:42)
Straight-line segment: `from`, `to` as `(f32, f32)` in voxel-space.

Static after init. `#[serde(skip)]` — regenerated from seed on load. Cells flat `Vec`, rivers/roads flattenable via `to_gpu_cells()`/`to_gpu_rivers()`. Deterministic function of seed.

---

### WorldState.nav_grids

**Type:** `Vec<NavGrid>`
**Purpose:** Baked 2D walkable surfaces derived from `VoxelWorld`, one per settlement area. Pathfinding (A*, flow fields) operates on NavGrid.

#### NavGrid fields
| Field | Type | Meaning |
|---|---|---|
| origin_vx, origin_vy | i32 | Min corner in voxel-space |
| width, height | u32 | Grid dimensions |
| nodes | `Vec<NavNode>` | Row-major `(dy·width + dx)` |

#### NavNode
| Field | Type | Meaning |
|---|---|---|
| walkable | bool | Whether a walker can stand here |
| surface_z | i32 | Z of walkable surface |
| move_cost | f32 | Material-based cost; 0 for non-walkable |

Rebuilt when `VoxelWorld` has structural changes; `buildings.rs:rebake_nav_grids` walks columns top-down. GPU-friendly (flat row-major). Pure function of `voxel_world`; rebuildable via `NavGrid::bake`.

---

### FidelityZone

**Type:** `FidelityZone` in `WorldState.fidelity_zones: Vec<FidelityZone>`
**Purpose:** Proximity bubble controlling sim fidelity. Entities inside run at zone level (High/Medium/Low/Background).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique zone ID | spawn | grid/zone lookups |
| fidelity | Fidelity | `High`/`Medium`/`Low`/`Background` | exploration EscalateFidelity, threat systems | tick fidelity dispatch |
| center | vec3 | World-space center | spawn | `update_grid_membership` (proximity, 3D Euclidean) |
| radius | f32 | Zone radius | spawn | membership |
| entity_ids | `Vec<u32>` | Entity IDs inside (recomputed each tick) | `update_grid_membership` | `has_hostiles`, `has_friendlies` |

`entity_ids` is derived; rest is primary.

---

### BuildSeed

**Type:** `BuildSeed` in `WorldState.build_seeds: Vec<BuildSeed>`
**Purpose:** Placed room-growth seed. NPC sets a seed; automaton enlarges outward until enclosed.

| Field | Type | Meaning |
|---|---|---|
| pos | TilePos | Seed tile position |
| intended_function | BuildingFunction | Room function (drives `minimum_interior`) |
| minimum_interior | u32 | Required interior tile count for completion |
| placed_by | u32 | Entity that set the seed |
| tick | u64 | Tick placed |
| complete | bool | True when grown OR stalled past `MAX_SEED_ATTEMPTS` |
| attempts | u16 | Room-growth attempts (stall detection) |
| last_interior_size | u16 | Last observed interior size for stall detection |

Added by NPC PlaceBuildSeed action; mutated by construction room-growth; pruned when complete. Primary state.

---

### StructuralEvent

**Type:** `StructuralEvent` in `WorldState.structural_events: Vec<StructuralEvent>`
**Purpose:** Per-tick events from voxel collapse/fracture. **Cleared at tick start** — ephemeral.

#### Variants
- `FragmentCollapse { chunk_x, chunk_y, chunk_z, fragment_voxel_count: u32, cause: CollapseCase }`
- `StressFracture { chunk_x, chunk_y, chunk_z, cluster_mass: f32, material_strength: f32 }`

#### CollapseCase
- `NpcHarvest` — voxel_harvest (tree felling, mining)
- `NpcConstruction` — placed voxel destabilised neighbours
- `Natural` — organic collapse from structural_tick

---

### SimScratch (NOT persistent state)

**Type:** `SimScratch`. Pooled scratch buffers reused across tick systems. Cleared + refilled within a single function call. `Clone` returns `Default` (don't duplicate allocations).

#### Sub-buffer ownership
- `action_eval`: `snaps` (entity snapshots), `snap_grid` (indices by cell), `snap_grids_typed` (kind-typed grids), `deferred` (action decisions)
- `exploration`: `npc_indices` (scan list), `npc_pos_voxel` (cached positions), `visible_cells`
- `construction::flood_fill_with_boundary`: `flood_visited` (128×128 generational), `flood_current_gen`, `flood_queue`, `flood_interior`, `flood_boundary`

---

### GroupIndex

**Type:** `GroupIndex`
**Purpose:** Contiguous per-settlement/per-party entity ranges. After `rebuild_group_index()`, entities sorted by `(settlement_id, party_id)`; systems iterate a slice.

#### Fields
| Field | Type | Meaning |
|---|---|---|
| settlement_ranges | `Vec<(u32,u32)>` | `[sid] = (start, end)` into entities — all kinds |
| settlement_npc_ranges | `Vec<(u32,u32)>` | Per-settlement NPC sub-range |
| settlement_building_ranges | `Vec<(u32,u32)>` | Per-settlement Building sub-range |
| settlement_monster_ranges | `Vec<(u32,u32)>` | Per-settlement Monster sub-range |
| party_ranges | `Vec<(u32,u32)>` | Per-party ranges |
| unaffiliated_range | (u32,u32) | Entities not assigned to any settlement |

#### Accessors
`settlement_entities(sid)`, `settlement_npcs(sid)`, `settlement_buildings(sid)`, `settlement_monsters(sid)`, `party_entities(pid)`, `unaffiliated_entities()` → `Range<usize>`.

Rebuilt by `rebuild_group_index()` after structural entity changes. Derived from `entities` + `settlement_id`/`party_id`.

---

### SurfaceCache

**Type:** `HashMap<u64, i32, ahash::RandomState>` keyed by packed `(vx, vy)`.

Lazily populated by `scan_voxel_resources_cached` on miss. Fallback for positions outside `surface_grid`. Pure function of `(vx, vy, region_plan, seed)`; `#[serde(skip)]`.

### FlatSurfaceGrid / FlatSurfaceTile

#### FlatSurfaceTile
| Field | Type | Meaning |
|---|---|---|
| origin_x, origin_y | i32 | Tile origin in voxel-space |
| width, height | i32 | Tile dimensions |
| heights | `Vec<i16>` | Row-major `(dy·width + dx)` surface z-heights |

#### FlatSurfaceGrid
`tiles: Vec<FlatSurfaceTile>` — one tile per settlement region.

Populated by `warm_surface_cache` per settlement. `get(vx,vy)` linearly scans tiles. ~30× faster than HashMap lookup, 10× less memory. `#[serde(skip)]`. Derived, GPU-friendly.

### CellCensus

**Type:** `HashMap<(i32,i32), [u32; 6], ahash::RandomState>` keyed by `(cell_x, cell_y)` where each cell spans `RESOURCE_CELL_SIZE = 128` voxels.
**Value:** `[count_wood, count_iron, count_copper, count_gold, count_coal, count_crystal]`.

Lazily populated when an NPC sees the cell. Persistent for the run; invalidation on voxel edits is tech debt (resources change slowly). Derived, `#[serde(skip)]`. GPU-hostile (HashMap); flat-grid candidate.

---

### WorldState.chronicle

**Type:** `Vec<ChronicleEntry>` — bounded ring of narrative events, human-readable.

#### ChronicleEntry
| Field | Type | Meaning |
|---|---|---|
| tick | u64 | When it happened |
| category | ChronicleCategory | Battle/Quest/Diplomacy/Economy/Death/Discovery/Crisis/Achievement/Narrative |
| text | String | Human-readable text |
| entity_ids | `Vec<u32>` | Entities involved |

Appended by ~20 systems. GPU-hostile (`String`).

### WorldState.world_events

**Type:** `Vec<WorldEvent>` — recent events for system queries (bounded).

#### Variants
`Generic{category,text}`, `EntityDied{entity_id,cause}`, `QuestChanged`, `FactionRelationChanged`, `RegionOwnerChanged`, `BondGrief`, `SeasonChanged`, `BattleStarted`, `BattleEnded`, `QuestPosted`, `QuestAccepted`, `QuestCompleted`, `SettlementConquered`.

---

### Summary

- **Primary state** (irreplaceable): `tick`, `rng_state`, `next_id`, `tiles`, `voxel_world.chunks`, `voxel_world.sea_level`, `region_plan` (regenerable from seed), `build_seeds`, `chronicle`, `world_events`, `fidelity_zones` (zone defs), `structural_events` (per-tick).
- **Derived state**: `entity_index`, `group_index`, `settlement_index`, `surface_cache`, `surface_grid`, `cell_census`, `nav_grids`, `max_entity_id`, `fidelity_zones[].entity_ids`.
- **Scratch (ephemeral)**: `SimScratch` + sub-buffers.
