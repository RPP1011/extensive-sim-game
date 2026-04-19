# World Sim State Catalog

Unified field-level inventory across agent, aggregate, and world state. Structure is per-domain; each subsection preserves the authoring of its source doc.

## Contents
- [Agent state](#agent-state) — (formerly `state.md`)
- [Aggregate state](#aggregate-state) — (formerly `state.md`)
- [World state](#world-state) — (formerly `state.md`)

---

## Agent state

### Agent (top-level)

Universal entity for any agentic actor — humans, wolves, dragons, goblins. The same struct, distinguished by `creature_type` and configuration.

#### Identity & Lifecycle
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

**Ground-locked vs volumetric creature_types.** Ground-locked types (Human, Elf, Dwarf, Wolf, Goblin) carry a post-movement constraint: a `@phase(post)` cascade rule keyed on `MovementApplied` snaps `pos.z = surface_height(pos.xy) + creature_height/2` outdoors, or `floor_height(pos, building) + creature_height/2` when `inside_building_id` is set. The constraint is applied automatically based on `creature_type` config. Volumetric types (Dragon, Fish, Bat) skip the snap and retain free z motion. `creature_height` is a derived field from `creature_type` config.

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
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| data | AgentData | Agent state (see AgentData section below) | agent_inner, action_eval, social/economic systems | all behavioral systems |
| inventory | Inventory | Commodity + gold storage (every agent has one) | work (production), trade, eating, looting | action_eval (resource availability), trader intent |
| memberships | Vec<Membership> | Groups this agent belongs to (faction, family, guild, religion, party, pack, settlement). Multi-membership produces emergent loyalty conflicts. | JoinGroup / LeaveGroup actions; spawn-time defaults | mask predicates (group-gated actions), observation features (memberships slot), `is_hostile` / `is_friendly` views |
| capabilities | Capabilities | Jump/climb/tunnel/fly/siege flags + can_speak / can_hear / hearing_range / can_build / can_trade flags derived from `creature_type`. `can_speak` / `can_hear` / `hearing_range` drive `Communicate` / `Announce` / overhear mask eligibility; `fly` gates z-separation combat predicates | template init (from `creature_type` config) | pathfinding obstruction, construction defense, mask predicates (`Communicate`, `Announce`, overhear), observation features |

Items the agent has equipped are referenced by ID from `data.equipped_items`. The Item entities themselves are catalogued in this doc's Item section.

---

### Membership

A single membership relating an agent to a group (faction, family, guild, religion, party, pack, settlement, etc.).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| group_id | GroupId | Which group | JoinGroup event | mask predicates, observation features |
| role | GroupRole | Member, Officer, Leader, Founder, Apprentice, Outcast | JoinGroup, PromoteEvent | mask predicates (leader-gated actions), reputation calculations |
| joined_tick | u64 | When joined | JoinGroup event | tenure calculation, observation feature |
| standing | f32 | -1..1 reputation within the group | various social events involving the group | mask predicates (Outcast cannot vote, etc.) |

`Group` itself (with leadership chain, treasury, standing relations to other groups, internal rules) is documented in `state.md`.

---

### StatusEffect

Temporary state modifier on an entity.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| kind | StatusEffectKind | Enum: Stun, Slow{factor}, Root, Silence, Dot/Hot{rate,interval}, Buff/Debuff{stat,factor} | apply_flat (effect dispatch) | movement (Slow/Root penalty), action (Silence blocks), damage (Dot/Hot tick) |
| source_id | u32 | Entity who applied this effect | apply_flat (effect source) | credit for damage/control |
| remaining_ms | u32 | Time left (milliseconds, converted from ticks) | cooldown system (decay per tick) | effect expiry check |

---

### Needs (6-dimensional)

Maslow-inspired primary drives. Range 0–100. Higher = more satisfied.

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

### Personality (5-dimensional)

Experience-shaped behavioral profile. Range 0–1. Set at spawn, drifts via events.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| risk_tolerance | f32 | Willingness to engage danger | combat win +0.02, combat loss -0.02, near death -0.05 | action selection (riskier options higher), flee threshold variance |
| social_drive | f32 | Preference for interaction | socializing event +0.01, isolation -0.01, trade completion +0.01 | goal selection bias (Socialize priority), relationship maintenance |
| ambition | f32 | Drive for status/achievement | quest completion +0.02, goal failure -0.02, witnessing promotion +0.01 | goal selection (higher ambition → more Work/Build), competitive behavior |
| compassion | f32 | Empathy, healing/helping bias | healing others +0.02, witnessing suffering +0.03, cruelty -0.05 | healing/rescue goal frequency, relationship trust modifier |
| curiosity | f32 | Information-seeking, exploration | learning skill +0.02, discovery of new location +0.01, repetitive work -0.01 | exploration goal frequency, trade diversity preference |

**Emergent?** Primary input (formed by events), but can derive behavior urgencies from it. Not reconstructible from others.

---

### Emotions (6-dimensional, Transient)

Short-term emotional state. Range 0–1, decays each tick (rate ≈0.01–0.05 depending on event).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| joy | f32 | Happiness spike (goal completion +0.3, good trade +0.2) | agent_inner decay (-0.02/tick), events (positive outcomes) | action urgency (joy increases morale & productivity) |
| anger | f32 | Rage/aggression (betrayal +0.4, theft +0.3, insult +0.2) | agent_inner decay (-0.02/tick), social conflict | combat preference, hostile action likelihood |
| fear | f32 | Dread/anxiety (near death +0.5, threat sighting +0.3) | agent_inner decay (-0.01/tick → slower decay), safety increase -0.1 | flee urgency, risk aversion, action caution |
| grief | f32 | Sorrow (ally death +0.6, home loss +0.4) | agent_inner decay (-0.01/tick), time passage | isolation preference, productivity penalty |
| pride | f32 | Self-regard (achievement +0.3, public recognition +0.2) | agent_inner decay (-0.03/tick), esteem events | social dominance in interaction, ambition boost |
| anxiety | f32 | Worry/uncertainty (resource shortage +0.2, goal failure +0.1) | agent_inner decay (-0.02/tick), aspiration drift | goal hesitation, conservative decisions |

---

### Aspiration (Phase B: Medium-term Orientation)

Personality-weighted need gap vector. Recomputed every 500 ticks. Provides 100-tick horizon behavioral bias.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| need_vector | [f32; 6] | Weighted gap (100 - need[i]) × personality[i], normalized | agent_inner (recompute every 500 ticks from needs + personality) | action_eval (action bias scoring), goal_eval (priority weighting) |
| vector_formed_at | u64 | Tick when last recomputed | agent_inner (formed tick) | recompute interval check (500 tick delta) |
| crystal | Option<Crystal> | Concrete target (Entity/Class/Building/Location/Resource) bound to a need dimension | salient events (quest, discovery, relationship), failure resets | action goals (concrete pursue path vs. need balancing) |
| crystal_progress | f32 | Completion toward crystal (0.0–1.0) | systems progressing crystal goal (work toward building, gather resource) | crystal_last_advanced tick check, UI display |
| crystal_last_advanced | u64 | Last tick crystal_progress increased | crystal progress events | stall detection (no progress → abandon crystal) |

---

### Memory

Event log + semantic beliefs formed from experience.

#### Memory.events (Ring Buffer)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| events | VecDeque<MemoryEvent> | Capped at 20 entries (oldest discarded) | event recording (memory.record_event) | belief formation, narrative, relationship updates |

#### MemoryEvent
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tick | u64 | When event occurred | event recorder | temporal reasoning, forgetting rate |
| event_type | MemEventType | Variant: WasAttacked, AttackedEnemy, FriendDied, WonFight, CompletedQuest, LearnedSkill, WasHealed, WasBetrayedBy, TradedWith, BuiltSomething, LostHome, WasRescuedBy, Starved, FoundShelter, MadeNewFriend, BecameApprentice, etc. | event system (combat, death, quest, learning) | belief update, reputation effect |
| location | vec3 | Where event occurred | event recorder | location-based belief (place dangerous/safe), pathfinding fear |
| entity_ids | Vec<u32> | Entities involved (attacker, victim, rescuer, etc.) | event recorder | relationship update, grudge formation |
| emotional_impact | f32 | -1.0 to +1.0 intensity | event recorder | emotion boost, memory weight |
| source | Source | Provenance of the event record (see Source enum below) | event recorder (Witnessed) / Communicate cascade / Announce cascade / Read cascade / rumor propagation | confidence derivation, mask predicates (`knows_*`, `confident_about`), observation `info_source_one_hot`, relationship trust |
| confidence | f32 ∈ [0, 1] | Derived from `source` by default (Witnessed=1.0, TalkedWith/Announced=0.8, Overheard=0.6, Rumor{hops}=0.8^hops, Testimony=doc trust_score); may be set explicitly | event recorder | mask predicates (`confident_about`), action weighting, decay |

---

### Source (enum)

Provenance tag for a `MemoryEvent`. Determines default confidence and observation feature.

```rust
enum Source {
    Witnessed,              // saw it directly; confidence = 1.0
    TalkedWith(AgentId),    // told by an agent; confidence = 0.8
    Overheard(AgentId),     // caught someone else's Communicate/Announce; confidence = 0.6
    Rumor { hops: u8 },     // N-hop propagation; confidence = 0.8^hops
    Announced(GroupId),     // broadcast from a group; confidence = 0.8
    Testimony(ItemId),      // written record (Document); confidence = document.trust_score
}
```

`Source` is observation-visible: the per-actor slot feature `info_source_one_hot[5]` (Witnessed / TalkedWith / Overheard / Rumor / NeverMet) projects this enum down to a 5-way one-hot.

#### Memory.beliefs (Semantic Layer)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| beliefs | Vec<Belief> | Semantic conclusions from events (LocationDangerous/Safe, EntityTrustworthy/Dangerous, SettlementProsperous/Poor, SkillValuable, FactionFriendly/Hostile, Grudge, HeardStory) | belief_formation system (from memory events, social gossip, trade info) | action_eval (destination safety), relationship trust init, goal location bias |

**Belief.confidence** [0.0, 1.0] — decays without update (~0.95/tick absent reinforcement).

---

### Relationship (Per-pair, NPC → NPC)

Directional relationship from one NPC toward another (asymmetric).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| trust | f32 | [-1.0, 1.0]. Positive = friendly, negative = hostile. Starts 0.0 | interaction events (trade +0.1, combat -0.3, betrayal -0.5, rescue +0.3), time passage (slow decay) | action_eval (trade partner preference, combat targeting), marriage likelihood |
| familiarity | f32 | [0.0, 1.0]. Grows with proximity and interaction | co-location, trade, work, socializing (+0.02–0.05/interaction) | relationship influence weight, interaction frequency |
| last_interaction | u64 | Tick of last trade/combat/work together | interaction events | relationship weight decay, "forgotten" threshold |
| perceived_personality | PerceivedPersonality | Mental model of other's traits (see section below) | observe_action method (from witnessed NPC behavior) | compatibility check, collaboration prediction, gossip basis |
| believed_knowledge | Bitset<32> | Theory-of-mind: bits self believes the other agent knows, indexed by knowledge domain (Combat, Trade, Family, Politics, Religion, Craft, ...). Same bit layout as `self.knowledge_domain_bits`. | set when self witnesses or is told about the other's domain-tagged actions ("saw them fight → Combat bit"); slow decay without reinforcement | mask predicates (e.g. `Deceive(t) when ¬believed_knowledge(t, Fact::X)`), observation features, gossip targeting |

**Capped at 20 relationships per NPC** (evict lowest familiarity if over 20).

#### PerceivedPersonality (Theory of Mind)

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| traits | [f32; 5] | Estimated [risk_tol, social, ambition, compassion, curiosity] | observe_action (action→trait signal, alpha learning) | compatibility calc with own personality, behavior prediction |
| confidence | [f32; 5] | Per-trait confidence [0.0, 1.0] | observe_action (+=0.01×attention per action) | trait update alpha (lower confidence → higher alpha), gossip weight |
| observation_count | u32 | Total action observations | observe_action (count++) | authority weight (more observations → less influence) |

**Emergent?** Derived entirely from observed actions. Reconstructible from action history.

---

### Needs-Driven Behavioral State

#### Personality-Weighted Need Gap
**Emergent.** Aspiration.need_vector = personality-weighted normalization of (100 - needs[i]).

---

### Goal & Action Execution

#### Goal (Priority-Sorted Stack)

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| kind | GoalKind | Enum: Idle, Work, Eat, Trade{settlement}, Fight, Socialize, Rest, Quest{id,pos}, Flee{from}, Build{id}, Haul{commodity,amount,dest}, Relocate{settlement}, Gather{commodity,amount} | goal_eval system (interrupt/assign based on needs + personality) | action system (dispatches per goal), movement pathfinding |
| priority | u8 | Urgency level (0–95). Flee=95, EatCritical=90, Fight=80, Work=50, Rest=20, Idle=0 | goal_eval (computed from needs urgency) | goal stack sorting (higher priority preempts) |
| started_tick | u64 | When this goal was pushed | goal_eval (push) | timeout detection, goal age for stall detection |
| progress | f32 | [0.0, 1.0] completion | action execution (work/haul/build incrementally advance) | UI display, adaptive urgency (goals regain priority if progress stalls) |
| target_pos | Option<vec3> | Movement destination for this goal | goal_eval (from location urgency), movement (arrival clears) | movement system (pathfinding goal) |
| target_entity | Option<u32> | Building/NPC/item/resource entity ID | goal_eval (from entity availability scans) | work assignment (building ID), interaction (NPC ID) |
| plan | Vec<PlannedStep> | Multi-step decomposition (e.g., Travel → Work → Carry → Rest) | plan generation (action_eval or GOAP-style planner) | action dispatch (per-step state machine) |
| plan_index | u16 | Current active step | step completion (advance_plan) | current_step() accessor for action dispatch |

#### GoalStack (Interrupt/Resume Mechanism)

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| goals | Vec<Goal> | Priority-sorted (descending), cap 8 | push (dedup + re-sort), pop (completion), remove_kind (override) | current() accessor (top goal drives agent), interrupt check (new high-priority goal preempts) |

**Behavior:** Goals are processed by priority. Eating/fleeing can interrupt work. On interrupt completion, the previous goal resumes (hysteresis).

---

### AgentData

Container for all agent-specific state — needs, emotions, personality, relationships, classes, equipment, work state, social ties, learned biases. Carried by every Agent regardless of `creature_type`.

#### Identity & Metadata
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| name | String | Generated from entity_id + seed deterministically | name generation (spawn) | UI display, speech, quest text |
| adventurer_id | u32 | Campaign Adventurer link (0 if unlinked) | party entry, class assignment | party queries, campaign persistence |
| creature_type | CreatureType | Enum: Citizen (default), PackPredator, Territorial, Raider, Megabeast | constructor (template.kind), monster spawn | personality preset, ability unlock, action gating |

#### Economic & Trade
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| gold | f32 | Liquid currency | trade (earn/spend), work (wages accumulate), looting | trade purchasing power, debt payment, bribe ability |
| debt | f32 | Gold owed to creditor | loans system, trade deficit | work motivation (debt priority), bankruptcy check |
| creditor_id | Option<u32> | Entity ID of creditor | loans system (loan grant) | debt collection events, relationship penalty |
| income_rate | f32 | EMA of gold earned per work cycle (alpha=0.1) | work completion (update_ema), production bonus | income belief for credit decision, wage negotiation |
| credit_history | u8 | [0, 255] reliability score | contract completion (+10 on success, -5 on default), repayment | bid weight in service contracts, loan availability |
| economic_intent | EconomicIntent | Enum: Produce{commodity,rate}, Trade, Idle, Loot, Haul, Work | goal_eval (derived from needs + personality) | action dispatch, work building assignment, trader routing |
| price_knowledge | Vec<PriceReport> | Commodity prices heard via gossip (settlement_id, price_estimate) | trade gossip updates | trade margin calculation, speculation intent |
| trade_route_id | Option<usize> | Index into WorldState.trade_routes (if assigned) | trade system (route assignment) | repeatable trade path, route profit tracking |
| trade_history | Vec<(u32, u32)> | (destination_settlement_id, success_count) | trade completion (record profit) | repeat trade tendency, exploration bias |

#### Location & Settlement Affiliation
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| home_settlement_id | Option<u32> | Settlement where NPC is "based" for economics | settlement residence, movement | work assignment building location, trade route origin, relocation trigger |
| home_building_id | Option<u32> | Primary residence (where Rest goal leads) | settlement housing, family founding | rest location, shelter need satisfaction |
| work_building_id | Option<u32> | Assigned workplace (building entity ID) | work assignment system | work goal target, work_state machine |
| inside_building_id | Option<u32> | Currently inside building (entity ID) | movement (enter/exit interiors) | interior navigation, room access, shelter provision |
| current_room | Option<u8> | Room index within building.rooms | interiors system (room occupation) | room-based AI (separate grid), visual location |
| home_den | Option<vec3> | Lair position for monsters (Territorial/PackPredator) | constructor (spawn position) | monster return-home behavior, territorial defense |

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

#### Relationships & Social (Phase C/D)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| relationships | HashMap<u32, Relationship> | Per-pair directional relationships | interaction events (modify_relationship), capped at 20 | action_eval (trust-based decisions), collaboration preference |
| spouse_id | Option<u32> | Married-to entity ID | family system (marriage event) | family mechanics, shared housing, co-parenthood |
| children | Vec<u32> | Child entity IDs | family system (birth) | family tree, inheritance, legacy |
| parents | Vec<u32> | Parent entity IDs | family system (birth parent link), succession | lineage, inherited skills, family duty |
| mentor_lineage | Vec<u32> | Mentor chain (most recent first) | apprenticeship (mentee links back to mentor) | skill inheritance, teaching bonus, apprenticeship completion |
| apprentice_of | Option<u32> | Current mentor entity ID | apprenticeship start | skill acceleration, teaching reputation |
| apprentices | Vec<u32> | Mentee entity IDs | apprenticeship (apprentice links forward) | teaching labor, succession planning, legacy |
| pack_leader_id | Option<u32> | Leader entity ID (for PackPredator monsters) | pack formation system | regrouping behavior, combat coordination, pack morale |

#### Perception & Resource Knowledge
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| known_resources | HashMap<u32, ResourceKnowledge> | Discovered resource entities (position, type, observation_tick) | exploration (scan for resources nearby), gossip (hear of resource), stale >2000 ticks | gatherer targeting, trade route planning, known good source preference |
| known_voxel_resources | Vec<VoxelResourceKnowledge> | Ore veins, stone, etc. (voxel coordinates) | voxel exploration, mining gossip | miner targeting, mining site navigation |
| harvest_target | Option<(i32, i32, i32)> | Currently-being-mined voxel (chunk x, y, z) | harvesting system (voxel mining) | harvest progress tracking, harvest completion |

#### Psychological & Emotional
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| needs | Needs | 6D Maslow vector (see Needs section) | agent_inner (tick drift, event bumps) | goal_eval (urgency), aspiration, action bias |
| memory | Memory | Event log + beliefs (see Memory section) | event recording, belief formation | belief-based decisions, relationship updates, reputation effects |
| personality | Personality | 5D trait vector (see Personality section) | event-driven drift (small deltas) | aspiration weight, action selection bias, relationship compatibility |
| emotions | Emotions | 6D transient state (see Emotions section) | events, decay per tick | action urgency, goal preemption, mood-based behavior |
| aspiration | Aspiration | Medium-term orientation with optional concrete crystal (see Aspiration section) | agent_inner recompute (500 ticks), salient events (form crystal) | action bias scoring, goal selection, concrete pursue objective |
| action_outcomes | HashMap<(u8, u32), OutcomeEMA> | Adaptive learning: (action_type, target_type_hash) → EMA of outcomes | apply action result (reward/penalty EMA update) | action_eval (adaptive urgency interrupt, policy learning) |
| price_beliefs | [PriceBelief; NUM_COMMODITIES] | Per-commodity estimated value + confidence | trade experience (direct/trade/secondhand update), decay | trade margin calc, speculation decision, price gossip |
| cultural_bias | [f32; 12] | Per-action-type conformity bias [-0.3, 0.3] | social_gathering (from peer actions), goal success (party influence) | action utility weighting, decision smoothing toward party norms |

#### Campaign System Fields (Migrated)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| morale | f32 | [0, 100] combat morale | combat events, settlements (aura), quest completion | combat action urgency, retreat threshold |
| stress | f32 | [0, 100] psychological stress | combat damage, near-death, failing quests | focus penalty (morale × (1 - stress/100)), mental state |
| fatigue | f32 | [0, 100] physical fatigue | work, travel, combat | action speed penalty, rest need urgency |
| loyalty | f32 | [0, 100] faction/guild loyalty | faction events, betrayal, party failure | faction quest difficulty, party behavior gating |
| injury | f32 | [0, 100] incapacity level (≥90 incapacitated) | combat damage, healing, rest recovery | action capability (injury ≥90 → only idle), combat penalty |
| resolve | f32 | [0, 100] willpower under pressure | fear events, morale pool, leadership aura | flee threshold modifier, action commitment |
| archetype | String | Hero class name ("knight", "ranger", "mage", etc.) | constructor (template.archetype) | ability unlock, stat scaling, action gating |
| party_id | Option<u32> | Party membership | party formation, expedition | party AI coordination, shared objectives, culture drift |
| faction_id | Option<u32> | Faction/guild membership | faction recruitment, succession | faction quests, faction goal priority, relationship modifier |
| mood | u8 | Current mood discriminant (0=neutral) | mood system (events shift mood) | action preference weighting, social openness |
| fears | Vec<u8> | Active fear type indices | phobia system (trigger fear) | fear-induced behavior (avoidance), anxiety emotion |
| deeds | Vec<u8> | Legendary deed types earned | deed system (achievement event) | reputation boost, pride emotion, story arc |
| guild_relationship | f32 | [-100, 100] reputation with guild | guild quests (complete +10, fail -5), betrayal | guild benefits access, quest assignment priority |

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

Structure definitions for player-placed buildings.

#### Identity & Type
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| building_type | BuildingType | Enum: Shelter, Farmhouse, Mine, Laboratory, etc. (20+ types) | constructor (building placement) | function determination, capability unlock |
| settlement_id | Option<u32> | Owning settlement | constructor (placement), settlement system | economic integration, NPC affiliation |
| name | String | Display name | constructor (type default), renaming system | UI, narrative, identity |

#### Spatial & Structure
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| grid_col, grid_row | u16 | Position on settlement 128×128 grid | constructor | room/workspace layout, collision, interior navigation |
| footprint_w, footprint_h | u16 | Size in grid cells | constructor (type-dependent) | building extent, placement collision check |
| rooms | Vec<Room> | Interior room definitions (floor plan) | construction growth (room addition) | NPC room occupation, function distribution |
| tier | u8 | Upgrade level | construction (tier advancement) | capacity scaling, stat scaling |

#### Occupancy & Workforce
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| resident_ids | Vec<u32> | NPC residents (for Shelter-type buildings) | housing assignment system | resident count, social interaction location |
| worker_ids | Vec<u32> | NPC workers assigned to this building | work assignment system | production efficiency, wage payment, work location |
| residential_capacity | u32 | Max residents | type default, upgrades | housing scarcity, population pressure |
| work_capacity | u32 | Max workers | type default, upgrades | labor bottleneck, staffing urgency |
| worker_class_ticks | Vec<...> | Per-class worker tick counts (for class-specific production) | work system (time logged by class) | production quality calc, class-specific yields |

#### Production & Storage
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| storage | [f32; NUM_COMMODITIES] | Commodity inventory | production (accumulate), trade (buy/sell), haul (receive) | inventory cap check, supply for trade, haul source |
| storage_capacity | f32 | Total commodity cap | type default, upgrades | inventory overflow management, trade limiting |

#### Construction & Status
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| construction_progress | f32 | [0.0, 1.0] completion | construction system (work ticks += progress) | building capability (functions at >0.5), visual display |
| built_tick | u64 | When completed (progress == 1.0) | construction system (record on completion) | age-based bonuses (ancient buildings), decay start |
| builder_id | Option<u32> | Primary builder entity ID | construction system (record who completed) | builder reputation, inheritance after death |
| builder_modifiers | Vec<...> | Quality modifiers from builder (tag-based bonuses) | construction (apply builder class tags) | production yield buff, special effects |
| owner_modifiers | Vec<...> | Quality modifiers from owner (tag-based bonuses) | ownership system | production buff, special mechanics |
| temporary | bool | If true, building expires at TTL | temporary building system (build seed) | expiry check, building permanence |
| ttl_ticks | Option<u64> | Time-to-live (if temporary) | temporary system (set at build), decrement per tick | expiry event trigger |
| specialization_tag | Option<u32> | FNV-1a hash of specialized production tag (e.g., tag(b"farming")) | specialization system (set by producer) | production yield bonus, economic focus |
| specialization_strength | f32 | Strength of specialization bonus [0.0, 2.0] | specialization system (increase with successful production) | multiplicative yield boost, economic development |

---

### Room (Interior Floor Plan)

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u8 | Unique room ID within building | building construction | room occupation, room-specific queries |
| kind | RoomKind | Enum: Bedroom, Workshop, Storage, Gathering, Shrine, etc. | constructor (floor plan generation) | function assignment, NPC role in room |
| interior_x, interior_y | u16 | Position within building grid | layout generation | room boundary, navigation within interior |
| interior_w, interior_h | u16 | Size | layout generation | room extent, capacity calculation |
| occupants | Vec<u32> | NPC entity IDs currently in room | interiors system (room entry/exit) | social gathering location, social need satisfaction |
| furnishing_level | u8 | Comfort/quality (0–100) | construction upgrades, furniture placement | shelter need bonus, social gathering success |
| blessing | Option<...> | Special effect (e.g., fertility for bedrooms) | ritual system | gameplay bonus, thematic benefit |

---

### Inventory (Portable Commodity Storage)

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| commodities | [f32; NUM_COMMODITIES] | Per-commodity quantity | work (produce), eating (consume), trade (transfer), haul (load) | trader intent (have/want calc), action_eval (resource availability) |
| gold | f32 | Currency | trade (cost transfer), wage payment, looting | purchasing power, loan eligibility |
| capacity | f32 | Total weight cap | constructor (NPC 50.0, building 500.0) | encumbrance check, overflow prevention |

**Updated by:** Nearly all economic/production systems.

---

### Derivation Graph

Showing which fields are primary inputs vs. derived/reactive:

```
PRIMARY INPUTS (Sources of Truth):
├─ Entity.alive ← death events, resource depletion
├─ Entity.pos (vec3) ← movement system (post-snap for ground-locked creature_types)
├─ Entity.hp ← combat damage, healing, work eating
├─ Entity.status_effects ← ability application, cooldown decay
├─ Needs ← agent_inner drift, event bumps
├─ Personality ← event-driven trait deltas
├─ Emotions ← events, decay per tick
└─ Memory.events ← event recording (combat, death, quest, etc.)

SECONDARY (Computed/Emergent):
├─ Aspiration.need_vector ← personality-weighted needs (recomputed 500-tick)
├─ Aspiration.crystal ← salient event (quest, discovery, interaction)
├─ Relationship.perceived_personality ← observe_action from behavior
├─ Emotions (derived from needs) ← [safety → fear], [purpose fail → grief], [combat win → joy]
├─ Morale ← personality + needs average, quest reward
├─ Behavior_profile ← accumulate_tags (all action tags), class xp grants
├─ Goal_stack.current_priority ← urgency(needs, personality, threat)
└─ PassiveEffects ← compute(behavior_profile) on class upgrade

INFRASTRUCTURE (Caches/State):
├─ Entity.entity_index ← rebuild_entity_cache (O(1) ID lookup)
├─ Entity.group_index ← rebuild_group_index (settlement grouping)
├─ NpcData.cached_path ← pathfinding result
├─ NpcData.price_knowledge ← trade gossip snapshot
└─ SimScratch pools ← pooled buffer reuse

TRANSFORMATIONS:
- Focus = (avg_needs / 100).clamp(0.5, 1.5) ← Needs average
- Urgency(need) = (100 - need) / 100 * personality[dim] ← Needs + Personality
- Effective Quality = base * rarity_multiplier * durability_fraction ← ItemData
- Action Utility = goal_priority + aspiration_bias + cultural_bias[action] + action_outcomes EMA
```

---

### Update Frequency Summary

| Pattern | Frequency | Systems Involved |
|---|---|---|
| Needs drift (hunger, safety, etc.) | Every tick | agent_inner, entity_drift |
| Emotions decay | Every tick | agent_inner |
| Aspiration recompute | Every 500 ticks | agent_inner |
| Status effect cooldown | Every tick | cooldown system |
| Work production | Every tick (in work state) | work system |
| Goal interrupt/assign | Every tick (if idle or need urgency spikes) | goal_eval, agent_inner |
| Relationship interaction | When trade/combat/co-work happens | interaction systems |
| Memory event record | When significant event happens | event systems |
| Belief update | When memory event recorded or trade gossip heard | belief_formation system |
| Behavior profile accumulate | When action completes | action system |
| Class XP grant | When action matches class tag | progression system |
| Price belief update | When trade completes or gossip heard | trade/social systems |
| Passive effects compute | On class level-up | progression system |

---

### Notes on Determinism & Serialization

- **hot/cold split**: `Entity` is unpacked into `HotEntity` (cache-line scalars) and `ColdEntity` (heap-heavy refs) for iteration performance. Both are serialized via entity.split() / Entity::from_parts().
- **Scratch buffers (SimScratch)**: #[serde(skip)] — ephemeral per-tick, regenerated on demand. Not serialized.
- **entity_index, group_index**: #[serde(skip)] — rebuilt on load via rebuild_all_indices().
- **Determinism**: All entity randomness via entity_hash(id, tick, salt) — never thread_rng(). Same (id, tick) always gives same result.
- **Tag hashing**: Compile-time FNV-1a hash(tag_name) for tag_hash constants — used for personality trait matching, class lookup, behavior_profile binary search.

---

## Aggregate state

`Group` is the universal social-collective primitive. Every settlement, faction, family, guild, religion, hunting pack, criminal cabal, adventuring party, court, league, or monastery is a `Group` distinguished by its `kind` discriminator. The `Group` section at the end of this doc gives the canonical shape; the per-kind sections that follow describe the additional fields each kind carries.

### Settlement (Group with `kind=Settlement`)
Political, economic, and structural hub for agents. Carries stockpiles, prices, treasury, population, group affiliations, and a fixed location with facilities.

#### Identity & Metadata
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique settlement ID | spawn/init | all settlement-aware systems |
| name | String | Display name | init | UI, chronicle, logs |
| pos | vec3 | World-space position. 2D footprint center; `pos.z` derived as `surface_height(pos.xy)` at init. [OPEN] whether multi-level settlements (underground + aboveground) warrant storing a per-layer z. | init | all systems using settlement location |
| grid_id | Option<u32> | NavGrid this settlement owns | init | navigation, pathfinding |
| specialty | SettlementSpecialty | Production/npc focus (Mining, Trade, Farming, Military, Scholar, Port, Crafting) | init | production, npc_spawn, resource nodes |

#### Economy
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| treasury | f32 | Gold reserves in coffers | taxes, economy, contracts, looting, theft, conquest, betrayal | economy, contracts, warfare, bankruptcy |
| stockpile | [f32; 8] | Commodity reserves (FOOD, WOOD, IRON, COPPER, HERBS, CRYSTAL, HIDE, MEDICINE, EQUIPMENT) | production, consumption, trade, looting | economy, npc_decisions, trade, production |
| prices | [f32; 8] | Local market prices per commodity | trade_goods, price_controls, seasons, commodity_futures, currency_debasement | economy, trades, npc_decisions, arbitrage |

#### Population & Morale
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| population | u32 | Count of alive NPCs with home_settlement_id == id | npc birth/death, migration | economy scaling, resource consumption, threat assessment |

#### Politics & Security
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| faction_id | Option<u32> | Owning faction, if controlled | conquest, civil_war, diplomacy | faction_ai, warfare, diplomacy, taxes |
| threat_level | f32 | Danger score (0–1) from nearby monsters, recent losses | monster_density (regional), quest_posting, attacks, threat_scaling | quest_posting urgency, recruitment, building upgrades |
| infrastructure_level | f32 | Building/defense tier (0–5) | construction, upgrades | defense strength, garrison capacity, building work |

#### Organization
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| context_tags | Vec<(u32, f32)> | Contextual action modifiers (e.g., plague, festival, war) | events (crisis, seasons, diplomacy) | action_system, skill resolution |
| treasury_building_id | Option<u32> | Entity ID of Treasury building | init (ensure_treasury_buildings) | resource movement, gold transfers |
| service_contracts | Vec<ServiceContract> | Active contracts posted by settlement NPCs | npc_decisions, contract_lifecycle | npc_decisions, contract resolution |
| construction_memory | ConstructionMemory | Per-settlement building event history (short/medium/long-term) | building_ai event logging | building_ai pattern learning |

---

### RegionState
Territorial and environmental layer. Represents large map areas with terrain type, monster populations, faction control, and dungeon sites.

#### Identity & Geographic
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique region ID | spawn | all systems referencing regions |
| name | String | Display name | spawn | UI, chronicle |
| pos | vec3 | World-space center. Synthesized at init from the sin/cos layout formula with `z = sea_level` (regions don't have an interior). | spawn | region queries, travel distance |
| terrain | Terrain | Biome type (Plains, Forest, Mountains, Coast, Swamp, Desert, Tundra, Volcano, DeepOcean, Jungle, Glacier, Caverns, Badlands, FlyingIslands, DeathZone, AncientRuins, CoralReef) | spawn | resource spawning, threat, travel speed |
| sub_biome | SubBiome | Terrain variant (Standard, LightForest, DenseForest, AncientForest, SandDunes, RockyDesert, HotSprings, GlowingMarsh, TempleJungle, NaturalCave, LavaTubes, FrozenCavern, MushroomGrove, CrystalVein, Aquifer, BoneOssuary) | spawn | resource yields, threat mods, travel speed |
| elevation | u8 | Terrain height tier (0–4: sea level, foothills, highlands, peaks, summit/sky) | spawn | resource rarity, threat, travel, building placement |

#### Ecology
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| monster_density | f32 | Local monster spawn rate multiplier | random_events, monster_spawning, threat_scaling | encounter generation, threat_level calc |
| threat_level | f32 | Current aggregate danger (0–1, accounting for monster density, elevation, recent battles) | monster density, battles, time decay | threat_reports, quest_posting, npc decisions |
| has_river | bool | Whether a river flows through this region | spawn | river_travel, trade_route generation |
| has_lake | bool | Whether a lake exists here | spawn | water-dependent resources, water travel |
| is_coastal | bool | Whether region borders ocean (map edge or adjacent DeepOcean) | spawn | coastal trade, sea creature encounters |

#### Dungeon Sites
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| dungeon_sites | Vec<DungeonSite> | Entrances to procedural dungeons (pos, name, explored_depth, max_depth, is_cleared, last_explored_tick, threat_mult) | dungeon_discovery, exploration | quest_posting, adventuring |

#### Connectivity
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| neighbors | Vec<u32> | Region IDs of 4-connected grid neighbors | spawn | travel, npc_movement, region_graphs |
| river_connections | Vec<u32> | Region IDs connected by river | spawn | river_travel, trade routes |
| is_chokepoint | bool | True if only 1–2 passable neighbors (strategic) | spawn | warfare, blockade tactics, npc_pathing |
| is_floating | bool | True if FlyingIslands terrain (requires special access) | spawn | access control, encounter generation |

#### Politics & Conflict
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| faction_id | Option<u32> | Controlling faction, if any | conquest, civil_war | faction_ai, warfare, taxes, recruitment |
| control | f32 | Faction control strength (0–1) | diplomacy, conquest, unrest | faction power assessment |
| unrest | f32 | Civil unrest (0–1, fuels rebellion) | diplomacy, oppression, riots | civil_war triggers, faction loyalty |

---

### Faction (Group with `kind=Faction`)
Macro-political collective representing governments, military powers, organized cults. Carries military_strength + standings to other groups + tech_level + governance.

#### Identity & Relationships
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique faction ID | spawn | all faction-aware systems |
| name | String | Faction name | init | UI, chronicle, logs |
| relationship_to_guild | f32 | Relationship to player guild (-100 to 100) | diplomacy, quest_completion, betrayals | quest_board filtering, npc morale, diplomacy |
| at_war_with | Vec<u32> | List of faction IDs in open warfare | civil_war, diplomacy | warfare, settlement conquest, movement restrictions |

#### Military
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| military_strength | f32 | Current fighting capacity (troops, equipment, morale) | recruitment, training, losses, desertion | conquest, diplomacy, battle strength |
| max_military_strength | f32 | Theoretical max capacity from population | population growth, tech_level | scaling, recruitment limits |

#### Territory & Resources
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| territory_size | u32 | Count of regions/settlements controlled | conquest, loss_of_control | power assessment, tax base, recruitment |
| treasury | f32 | Faction gold reserves | taxes (from settlements), wars (looting), quests, reparations | military building, mercenary recruitment, diplomacy |

#### Politics & Stability
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| diplomatic_stance | DiplomaticStance | Toward another faction (Friendly, Neutral, Hostile, AtWar, Coalition) | diplomacy system | npc decisions, interaction modifiers |
| coup_risk | f32 | Stability metric (0–1, higher = more likely internal upheaval) | oppression, morale, succession events | faction_ai decisions, succession crisis triggers |
| escalation_level | u32 | Conflict intensity (0–5, caps NPC conflict actions) | warfare, diplomacy escalation | warfare intensity cap, treaty enforcement |
| tech_level | u32 | Research/development tier | research system, quests | military bonus, production speeds, ability unlock |
| recent_actions | Vec<String> | Bounded log of recent faction events | all major systems | narrative, NPC knowledge, diplomacy memory |

---

### GuildState
Player faction state — independent from NPC factions.

#### Resources
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| gold | f32 | Guild treasury | quest_rewards, missions, expenses | quest_board filtering, upgrades, expenses |
| supplies | f32 | Supply reserve (used for missions) | production, usage, looting | mission readiness, supply checks |

#### Reputation & Capacity
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| reputation | f32 | Fame/standing (0–100) | quests, battles, diplomacy | quest_board filtering, npc_recruitment, prices |
| tier | u32 | Guild level (0–5, unlocks features) | reputation milestones | mission generation, capacity unlocks |
| credit_rating | f32 | Borrowing capacity (0–100) | loans, repayment | loan eligibility, interest rates |
| active_quest_capacity | u32 | Max simultaneous active quests | tier unlocks | quest acceptance checks |

---

### Quest & Quest Lifecycle

#### Quest (Active)
Active quest being pursued by party.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique quest ID | quest_board->quests | party, quest_lifecycle |
| name | String | Quest title | posted | UI |
| quest_type | QuestType | Hunt, Escort, Deliver, Explore, Defend, Gather, Rescue, Assassinate, Diplomacy, Custom | posted | party decisions, threat calc |
| party_member_ids | Vec<u32> | Assigned party entity IDs | acceptance, deaths, replacements | party_ai, completion checks |
| destination | vec3 | Objective location | posted | pathfinding, progress calc |
| progress | f32 | Completion ratio (0–1) | quest_lifecycle (tick updates) | UI, completion determination |
| status | QuestStatus | Traveling, InProgress, Completed, Failed, Returning | quest_lifecycle state machine | quest_lifecycle transitions, rewards |
| accepted_tick | u64 | When party accepted | acceptance | deadline calc, duration tracking |
| deadline_tick | u64 | Tick deadline (0 = none) | posted (urgency-driven) | failure condition, reward scaling |
| threat_level | f32 | Quest difficulty (0–1) | posted (from threat_reports) | party composition, reward scaling, npc recruitment |
| reward_gold | f32 | Gold upon completion | posted (from settlement treasury scaling) | party motivation, quest_board display |
| reward_xp | u32 | Experience upon completion | posted | party leveling |

#### QuestType Enum
`Hunt`, `Escort`, `Deliver`, `Explore`, `Defend`, `Gather`, `Rescue`, `Assassinate`, `Diplomacy`, `Custom` — tag for behavior mechanics.

#### QuestStatus Enum
- `Traveling`: party moving to destination
- `InProgress`: at location, pursuing objective
- `Completed`: objective achieved
- `Failed`: party wiped or deadline passed
- `Returning`: returning to home settlement

#### QuestPosting (Board)
Unapplied quest available for guild to accept.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique posting ID | spawn/generation | quest_board filtering |
| name | String | Title | generated | UI |
| quest_type | QuestType | Category | generated | NPC decisions |
| settlement_id | u32 | Posting settlement | generated | settlement queries, faction affiliation |
| destination | vec3 | Objective | generated | party routing |
| threat_level | f32 | Difficulty | threat_reports | party selection, reward scaling |
| reward_gold | f32 | Bounty | settlement treasury, threat | guild motivation |
| reward_xp | u32 | Experience | threat scaling | party motivation |
| expires_tick | u64 | When posting removed | settlement urgency | quest_board cleanup |

---

### FidelityZone
Proximity bubble controlling simulation fidelity level (rich NPC detail vs. statistical abstraction).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique zone ID | spawn | fidelity_control system |
| fidelity | Fidelity | Richness level enum (High, Medium, Low) | fidelity_control | entity update frequency, detail depth |
| center | vec3 | Zone center (usually settlement pos) | fidelity_control, settlement movement | zone queries |
| radius | f32 | Zone extent in world units | fidelity_control, threat radius | proximity checks |
| entity_ids | Vec<u32> | Entities currently in this zone | entity_movement, spawn/despawn | fidelity state sync |

---

### TradeRoute
Emergent trade connection established by repeated profitable NPC trading.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| settlement_a | u32 | First settlement | npc_trade establishment | trade_logic, route strength decay |
| settlement_b | u32 | Second settlement | npc_trade establishment | trade_logic, route strength decay |
| established_tick | u64 | When route was created | npc_trade | historical tracking |
| total_profit | f32 | Cumulative gold profited | npc_trade | route viability assessment |
| trade_count | u32 | Number of successful trades | npc_trade | route activity metric |
| strength | f32 | Route health (0–1, decays without activity, abandoned < 0.1) | npc_trade, time decay | npc_decisions, route discovery |

---

### ServiceContract
NPC service request posted by settlement NPCs (build, gather, craft, heal, guard, haul, teach, barter).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| requester_id | u32 | NPC posting the contract | npc_decisions | contract_lifecycle |
| service | ServiceType | Type of work (Build(BuildingType), Gather(commodity, amount), Craft, Heal, Guard(target_id), Haul(commodity, amount, dest), Teach(npc_id), Barter{offer, want}) | npc_decisions | contractor capability matching |
| max_payment | Payment | Max NPC will pay (gold + commodities) | npc_decisions, morale | bid filtering |
| payment | f32 | Actual agreed payment (set on resolution) | contract_resolution | contractor reward |
| provider_id | Option<u32> | NPC accepting the contract | acceptance | contract_lifecycle, reputation |
| posted_tick | u64 | When posted | posting | contract age for cleanup |
| completed | bool | Whether work is done | contract_resolution | archive |
| bidding_deadline | u64 | Tick bidding closes (urgency-driven: critical +5, high +15, medium +30, low +100) | posted from urgency | bid window end |
| bids | Vec<ContractBid> | Offers from interested NPCs (bidder_id, bid_amount, skill_value, credit_history) | npc_decisions | winner selection |
| accepted_bid | Option<usize> | Index into bids of accepted offer | contract_resolution | payment disbursement |

---

### PriceReport
Historical price snapshot for arbitrage and market analysis.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| settlement_id | u32 | Which settlement | price_discovery | npc_trade arbitrage |
| prices | [f32; 8] | Prices at snapshot time | settlement price state | npc_trade profit calc |
| tick_observed | u64 | When recorded | price_discovery | stale check, trend calc |

---

### SettlementSpecialty Enum
Production focus for settlement NPCs and economy:
- `General`: baseline (no bonuses)
- `MiningTown`: +2.0 iron, +1.5 crystal production
- `TradeHub`: price discovery, merchant NPCs
- `MilitaryOutpost`: +1.3 iron, +0.5 equipment; warrior NPCs, threat reduction
- `FarmingVillage`: +2.0 food, +1.5 hide
- `ScholarCity`: +1.0 herbs, +0.5 medicine; research XP bonus
- `PortTown`: +1.5 food, +1.0 wood; coastal trade
- `CraftingGuild`: +2.0 equipment, +1.5 medicine

---

### DungeonSite
Procedural dungeon entrance in a region.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| pos | vec3 | World-space entrance location | spawn/discovery | quest routing, encounter gen |
| name | String | Procedural name (e.g., "Sunken Halls") | spawn | UI, quest descriptions |
| explored_depth | u8 | Current deepest level reached (0 = entrance only) | exploration_system | loot tier, discovery progress |
| max_depth | u8 | Total levels in dungeon | spawn | loot tier upper bound |
| is_cleared | bool | Whether fully cleared (no respawn) | exploration_system | encounter generation, loot refresh |
| last_explored_tick | u64 | When last entered | exploration_system | respawn timer, activity tracking |
| threat_mult | f32 | Danger modifier from terrain + depth | spawn | encounter difficulty |

---

### DiplomaticStance Enum
- `Friendly`: +morale, trade bonuses
- `Neutral`: standard interactions
- `Hostile`: -morale, embargo possible
- `AtWar`: open conflict, unit restrictions
- `Coalition`: shared military benefits

---

### Terrain Enum (17 types)
Biome type with resource yields, threat, travel speed, elevation tier:
- `Plains`: food 1.5x, hide 1.0x; threat 1.0x
- `Forest`: wood 1.5x, herbs 1.0x; threat 1.2x, travel 0.7x
- `Mountains`: iron 1.5x, crystal 1.0x; threat 1.5x, travel 0.5x, elevation 3
- `Coast`: food 1.2x, wood 0.8x; threat 0.8x, unsettleable
- `Swamp`: herbs 1.5x, medicine 1.0x; threat 1.3x, travel 0.4x
- `Desert`: crystal 1.2x; threat 1.4x, travel 0.6x
- `Tundra`: hide 1.2x; threat 1.3x, travel 0.6x
- `Volcano`: iron 2.0x, crystal 2.0x; threat 3.0x, travel 0.3x, elevation 3, unsettleable
- `DeepOcean`: food 0.5x (fishing); threat 2.5x, travel 0.0 (impassable), elevation 0, unsettleable
- `Jungle`: food 1.8x, herbs 1.5x, wood 1.0x; threat 1.8x, travel 0.4x
- `Glacier`: crystal 1.5x; threat 2.0x, travel 0.3x, elevation 3
- `Caverns`: iron 2.0x, crystal 1.5x; threat 2.0x, travel 0.5x, underground
- `Badlands`: iron 0.8x; threat 1.6x, travel 0.7x
- `FlyingIslands`: crystal 3.0x; threat 1.5x, travel 0.0 (special access), elevation 4, unsettleable, floating
- `DeathZone`: crystal 2.0x, medicine 1.5x; threat 5.0x, travel 0.5x, elevation 2, unsettleable, 5x threat multiplier
- `AncientRuins`: crystal 1.0x (artifacts); threat 2.5x, travel 0.6x, has dungeons
- `CoralReef`: food 1.5x, crystal 0.8x; threat 1.0x, travel 0.0 (underwater), unsettleable

---

### SubBiome Enum (16 types)
Terrain variant for detail within biome category:
- `Standard`: no variant
- `LightForest`: +1.2 travel, 0.6x wood (sparse)
- `DenseForest`: 0.5x travel, 1.8x wood, +1.5 threat (monsters ambush)
- `AncientForest`: 0.7x travel, 1.2x wood, 2.0x herbs, +1.3 threat
- `SandDunes`: 0.4x travel, +1.2 threat (exhausting)
- `RockyDesert`: standard travel, exposed ore
- `HotSprings`: 1.8x herbs (near heat)
- `GlowingMarsh`: 0.6x travel, 2.5x herbs (fungi), +1.4 threat
- `TempleJungle`: 0.5x travel, 1.5x herbs, +1.6 threat (hidden temples)
- `NaturalCave`: standard cave
- `LavaTubes`: basalt walls, lava pools (volcanic)
- `FrozenCavern`: ice walls, frozen lakes
- `MushroomGrove`: bioluminescent (organic caves)
- `CrystalVein`: crystal clusters, high ore (+1.5 ore mult)
- `Aquifer`: flooded chambers (underwater caves)
- `BoneOssuary`: ancient remains (death theme)

---

### EconomyState
Global economic aggregate.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| total_gold_supply | f32 | Sum of all gold (guild + factions + settlements) | economy system | inflation, market sentiment |
| total_commodities | [f32; 8] | Total supply of each commodity | production, consumption, trades | resource scarcity calc, price scaling |

---

### ChronicleEntry
Narrative log entry for historical record.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tick | u64 | When event occurred | event logging | timeline, quest narratives |
| category | ChronicleCategory | Event type (Battle, Quest, Diplomacy, Economy, Death, Discovery, Crisis, Achievement, Narrative) | event logging | filtering, narrative theming |
| text | String | Human-readable description | event logging | UI, lore |
| entity_ids | Vec<u32> | Involved entity IDs | event logging | relationship tracking |

---

### WorldEvent Enum (13 variants)
Immediate game events processed during tick:
- `Generic { category, text }`: catch-all
- `EntityDied { entity_id, cause }`: npc/monster death
- `QuestChanged { quest_id, new_status }`: quest state machine
- `FactionRelationChanged { faction_id, old, new }`: diplomatic shift
- `RegionOwnerChanged { region_id, old_owner, new_owner }`: territorial control
- `BondGrief { entity_id, dead_id, bond_strength }`: emotional reaction to death
- `SeasonChanged { new_season }`: annual cycle
- `BattleStarted { grid_id, participants }`: encounter begin
- `BattleEnded { grid_id, victor_team }`: encounter resolution
- `QuestPosted { settlement_id, threat_level, reward_gold }`: new opportunity
- `QuestAccepted { entity_id, quest_id }`: npc/guild accepts
- `QuestCompleted { entity_id, quest_id, reward_gold }`: quest reward
- `SettlementConquered { settlement_id, new_faction_id }`: faction conquest

---

### WorldState Top-Level Fields (not covered by per-entity docs)

#### Time & RNG
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tick | u64 | Current simulation tick | time system | all systems (time-based logic) |
| rng_state | u64 | Deterministic RNG seed state | next_rand_u32() | all probabilistic systems |

#### Indexes & Caches
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| entity_index | Vec<u32> | entity_id → index into entities/hot/cold (rebuilt) | rebuild_entity_cache() | O(1) entity lookups |
| max_entity_id | u32 | Highest entity ID seen (sizes entity_index) | rebuild_entity_cache() | index allocation |
| next_id | u32 | Monotonic entity ID counter | next_entity_id() | entity creation |
| group_index | GroupIndex | Contiguous ranges by (settlement_id, party_id) for batch iteration | rebuild_group_index() | settlement-scoped loops |
| settlement_index | Vec<u32> | settlement_id → index into settlements (rebuilt) | rebuild_settlement_index() | O(1) settlement lookups |

#### Spatial
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tiles | HashMap<TilePos, Tile> | Sparse tile modifications (walls, floors, ditches, farmland) | building/construction systems | pathfinding, rendering, collision |
| surface_cache | SurfaceCache | Cached surface height results (vx, vy) → height; lazily populated | exploration system | npc_decisions, resource scanning |
| surface_grid | FlatSurfaceGrid | Dense grid of surface heights (fallback cache for perf) | warm_surface_cache() | npc resource cell census, fast height lookups |
| cell_census | CellCensus | Per-resource-cell material counts (populated by NPC scans) | npc scanning, exploration | resource node discovery, NPC targeting |
| voxel_world | VoxelWorld | 3D chunked terrain (materials, elevation source of truth) | terrain generation, voxel destruction | navigation grid generation, building placement |
| nav_grids | Vec<NavGrid> | Walkable surfaces per settlement (derived from voxel_world) | voxel_world changes | npc pathfinding, building AI |

#### Memory & Scratch
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| sim_scratch | SimScratch | Pooled buffers (Vec/HashMap) reused across systems to reduce allocs | all systems | all systems (borrowed, cleared per tick) |

#### Collections
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| fidelity_zones | Vec<FidelityZone> | Proximity bubbles controlling detail level | fidelity_control | entity update frequency, detail depth |
| regions | Vec<RegionState> | Regional population, faction control, terrain | spawn | all region-aware systems |
| groups | Vec<Group> | All social-collective state — settlements, factions, families, guilds, religions, parties, packs (discriminated by `kind`) | spawn, JoinGroup/LeaveGroup, FoundGroup, DissolveGroup, conquest, diplomacy events | every group-aware system; queryable by kind via index |
| quests | Vec<Quest> | Active quests being pursued | quest_board acceptance | quest_lifecycle, party_ai |
| quest_board | Vec<QuestPosting> | Available quests not yet accepted | settlement/threat posting | guild quest selection, expiration cleanup |
| trade_routes | Vec<TradeRoute> | Emergent trading paths (strength decays) | npc_trade | npc_decisions, trade analysis |
| economy | EconomyState | Global totals (gold supply, commodity sum) | all economic systems | inflation, scarcity calc |
| adventurer_bonds | HashMap<(u32, u32), f32> | NPC-to-NPC bond strength (0–100) | relationships, quests, deaths | morale, party cohesion, grief events |
| guild | GuildState | Player faction state (gold, reputation, tier, capacity) | quest completion, events | quest board filtering, upgrades |
| relations | HashMap<(u32, u32, u8), f32> | Entity-to-entity relations by kind (relationship, bond, romance, rivalry, grudge, mentorship) | interaction, events | npc decisions, morale, romance events |
| chronicle | Vec<ChronicleEntry> | Narrative log (bounded ring buffer) | all major systems | UI history, lore, prophecy checking |
| prophecies | Vec<Prophecy> | Generated at init, fulfilled by world events | init, prophecy system | narrative hooks, prophecy fulfillment |
| world_events | Vec<WorldEvent> | Recent events (bounded, cleared/pushed per tick) | all systems emitting events | event-driven systems, prophecy matching |

#### Metadata
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| build_seeds | Vec<BuildSeed> | Room automaton queue waiting for processing | building_ai | room_growth, building construction |
| structural_events | Vec<StructuralEvent> | Collapses, fractures logged this tick (cleared at tick start) | building/structural systems | event cascade, building damage |
| registry | Option<Arc<Registry>> | Data-driven entity/ability/item definitions (loaded from dataset/) | init | entity spawning, ability execution |
| region_plan | Option<RegionPlan> | Continental terrain plan (stored for post-init chunk generation) | init | chunk generation reference |
| skip_resource_init | bool | Flag to skip resource node spawning (for building AI scenarios) | init | resource_nodes system |

---

### Derivation Graph

Fields that are derived (computable on-demand from entity state) vs. cached:

```
-- Derivable from entity state, currently cached:
population[sid] 
  ← count(entities where home_settlement_id == sid && alive)
  ** Could be recomputed each tick in O(n_entities) **

threat_level[region] 
  ← (monster_density * monster_count_in_region + recent_attack_count + recent_death_count) / (1 + time_since_crisis)
  ** Could be recomputed from entity positions + region history **

threat_level[settlement]
  ← 0–1 score from nearby (region.threat, monster_density, recent_losses, attacks)
  ** Could be recomputed from region.threat_level + local entity queries **

-- Emergent (require history to compute, not just current state):
prices[settlement, commodity]
  ← base_price / (1 + stockpile[c] / (population * price_halflife))
  ** Supply-demand feedback. Requires stockpile + population (emergent from entity count) **

strength[trade_route]
  ← 1.0 exponentially decays toward 0 without activity, abandoned < 0.1
  ** Requires temporal tracking; could be recomputed if trade_count + last_trade_tick stored **

trade_count, total_profit
  ← accumulated by npc_trade decisions over time
  ** Purely emergent, no on-demand recomputation path **

is_at_war[faction_a, faction_b]
  ← faction_a.at_war_with.contains(faction_b)
  ** Cached list; could iterate all factions each time (O(n²)) **

-- Cached for perf but stateless:
region_plan (continental terrain layout)
  ← generated at world init, stored for chunk reference
  ** Stateless; could be regenerated from seed if forgotten, but expensive **

surface_cache (height field lookups)
  ← lazily computed from voxel_world
  ** Pure function of (vx, vy, region_plan, seed); safe to recompute **

cell_census (material counts per resource cell)
  ← populated by NPC scans
  ** Could be recomputed by scanning all resource nodes, but expensive (~10-20ms per full rescan) **

nav_grids (walkable surfaces)
  ← derived from voxel_world
  ** Pure function of voxel_world; safe to recompute if voxel_world changes **

entity_index, settlement_index, group_index
  ← indices for O(1) lookups
  ** Pure functions of entities/settlements; must be rebuilt when those change **

-- Immutable (truly stored):
id, name, pos, terrain, sub_biome (all per-region/settlement/faction)
  ← set at spawn, never change
  ** Safe to store permanently **

treasury, stockpile (settlement/faction gold & goods)
  ← modified by economic systems
  ** Requires history; cannot derive without transaction log **

faction.at_war_with (war list)
  ← set by diplomacy/conquest systems
  ** Could be queried from relation graph, but cached for O(1) war checks **
```

#### Candidates for On-Demand Recomputation

1. **population** — Currently stored; could be recomputed in O(n_entities) per tick. Trade-off: 10–50 cycle loop vs. one integer field. **Keep cached** (too frequent queries).

2. **threat_level** (region & settlement) — Computable from monster_density + recent events + decay. **Keep cached** (used in quest posting hot loop).

3. **surface_cache** — Pure function of voxel_world; safe to invalidate and regenerate on demand, but expensive. **Keep cached** (used 16K+ times per cell scan).

4. **nav_grids** — Pure function of voxel_world; safe to regenerate when voxel changes. **Keep cached** (pathfinding is hot).

5. **cell_census** — Could be recomputed by iterating all resource nodes; takes 10–20ms per full rescan. **Keep cached** (used for target discovery, not latency-sensitive).

6. **trade_route.strength** — Decay is time-based; could be recomputed if last_activity_tick tracked. **Consider: store last_activity_tick, recompute strength on access**.

7. **is_at_war[faction_a, faction_b]** — Currently list lookup; could query relations graph. **Keep list** (hot path in movement, O(1) vs. O(n)).

---

### ConstructionMemory (Building AI Integration)

Per-settlement construction event log with three tiers:

| Tier | Field | Meaning | Capacity | Decay |
|---|---|---|---|---|
| Short-term | short_term: RingBuffer<ConstructionEvent> | Raw events (all types) | 64 entries | None (circular overwrite) |
| Medium-term | medium_term: RingBuffer<AggregatedPattern> | Patterns with importance > 0.3 | 256 entries | Halves every 500 ticks |
| Long-term | long_term: RingBuffer<StructuralLesson> | Structural lessons, importance > 0.7 | 64 entries | Permanent until contradicted |

Updated by: building_ai event logging system  
Read by: building_ai pattern learning (decide build strategies based on past successes/failures)

---

### Payment, ContractBid, ServiceType

**Payment**: Represents gold + commodities in a single transaction.
- `gold: f32` — gold amount
- `commodities: Vec<(u8, f32)>` — (commodity_index, amount) pairs
- Methods: `gold_only()`, `commodity()`, `estimated_value()`, `is_empty()`

**ContractBid**: An NPC's offer to perform a ServiceContract.
- `bidder_id: u32` — offering NPC
- `bid_amount: f32` — gold they'll accept
- `skill_value: f32` — skill at time of bid
- `credit_history: u8` — trust score (0–100)

**ServiceType** Enum: Work category for contracts.
- `Build(BuildingType)` — construction
- `Gather(commodity_idx, amount)` — resource collection
- `Craft` — manufacturing
- `Heal` — medical care
- `Guard(target_entity_id)` — protection
- `Haul(commodity_idx, amount, (x, y))` — transport goods
- `Teach(npc_id)` — education/training
- `Barter { offer: (idx, amt), want: (idx, amt) }` — commodity swap

---

### Summary: Which Fields Are Emergent vs. Stored

| Category | Field | Type | Emergent? | Updaters |
|---|---|---|---|---|
| Settlement pop. | population | u32 | **YES** (count alive at sid) | npc birth/death events |
| Threat | threat_level | f32 | Quasi (derived + cached) | monster_density, attacks, time |
| Prices | prices | [f32; 8] | Derived (supply-demand model) | trade_goods, seasons, price controls |
| Resources | stockpile | [f32; 8] | State (not derivable) | production, consumption, trade, looting |
| Treasury | treasury | f32 | State (not derivable) | taxes, maintenance, contracts, warfare |
| Faction war | at_war_with | Vec<u32> | Cached (queryable from relations) | diplomacy, civil_war |
| Quest progress | progress | f32 | Derived (from quest_lifecycle) | quest_lifecycle tick updates |
| Trade strength | strength | f32 | Derived (exponential decay) | npc_trade activity, time |
| Entity index | entity_index | Vec<u32> | **YES** (rebuild from entities) | rebuild_entity_cache() |

---

### Group (universal)

The social-collective primitive. A Group represents any collection of agents with shared identity and group-level state: factions, families, guilds, religions, hunting packs, criminal cabals, adventuring parties, settlements, leagues, monasteries, courts. The `kind` discriminator + presence/absence of optional fields differentiates them.

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
| standings | Map<GroupId, Standing> | Per-other-group disposition: Allied / Neutral / Tense / AtWar / Vassal / Suzerain / Excommunicate. **Replaces faction.at_war_with + diplomatic_stance.** | diplomacy events (AllianceFormed, WarDeclared, VassalSworn, ...) | mask predicates (can-attack-other-group, eligible for trade with), observation features, is_hostile derivation |
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

#### What's NOT on Group (derived views)

- `population` (use `members.len()`)
- `is_at_war(other)` — read from `standings[other]`
- `is_allied(other)` — read from `standings[other]`
- `wealth_per_member` — `treasury / members.len()`
- `cultural_descriptor` — derived from members' aggregate behavior_profiles
- `reputation_among(other_group)` — derived from cross-group event history

#### Per-kind shapes

| Kind | Typical fields populated |
|---|---|
| `Faction` | `military_strength`, `standings`, `governance`, `tech_level`, recruitment_open |
| `Settlement` | `facilities`, `charter`, `stockpile`, `treasury`, paired with a spatial record holding `pos`, `grid_id`, `region_id` |
| `Guild` | `treasury`, `recruitment_open`, `dues`, eligibility_predicate, often a settlement-bound charter |
| `Family` | `leader_id` (head of household), `members` derived from kin events |
| `Party` | `leader_id`, `founded_tick`, `dissolved_tick` set at quest completion or disbandment |
| `Religion` | `charter` holding scripture / pantheon, eligibility_predicate, leadership chain |
| `Pack` | `leader_id` is the alpha, eligibility_predicate keyed on `creature_type` |
| `Cabal` | `recruitment_open=false`, restrictive eligibility, secret standings |
| `Court` | `governance`, `leadership_chain` for ministerial succession |
| `League` | flat governance, `standings` heavy with member-group relations |

---

## World state

The environment layer: data defining terrain, structural tiles, voxels, spatial indices, and shared caches. Not per-entity, not per-settlement/faction.

Primary files:
- `src/world_sim/state.rs` — `WorldState`, `Tile`, `FidelityZone`, `BuildSeed`, `SimScratch`, `GroupIndex`, `StructuralEvent`
- `src/world_sim/voxel.rs` — `VoxelWorld`, `Chunk`, `Voxel`, `VoxelMaterial`
- `src/world_sim/terrain/region_plan.rs` — `RegionPlan`, `RegionCell`, `SettlementPlan`, `DungeonPlan`, `RiverPath`, `RoadSegment`
- `src/world_sim/systems/exploration.rs` — `FlatSurfaceGrid`, `SurfaceCache`, `CellCensus`
- `src/world_sim/nav_grid.rs` — `NavGrid`, `NavNode`
- `src/world_sim/constants.rs` — grid/voxel constants

---

### WorldState top-level fields (world/spatial only)

Per-entity and per-settlement/faction fields are documented in the other two docs. This section covers the rest.

#### Scalar / identity

| Field | Type | Meaning | Persistence | Updated by | Read by |
|---|---|---|---|---|---|
| tick | u64 | Monotonic tick counter | Primary | `WorldSim::tick` (once per step) | all time-gated systems |
| rng_state | u64 | PCG-style RNG state; sole randomness source | Primary | `next_rand_u32` / `next_rand` | all stochastic systems |
| next_id | u32 | Monotonic entity ID counter | Primary | `next_entity_id()` | entity spawn |
| max_entity_id | u32 | Highest ID seen (sizes `entity_index`) | Derived (cache sizing) | `rebuild_entity_cache`, `rebuild_group_index` | indexed lookups |

#### Derived indices (all `#[serde(skip)]` — rebuilt on load)

| Field | Type | Purpose | Rebuilt by | Read by |
|---|---|---|---|---|
| entity_index | `Vec<u32>` | `id → idx` into `entities/hot/cold`. Size = `max_entity_id+1`. Sentinel `u32::MAX` | `rebuild_entity_cache`, `rebuild_group_index` | `entity()`, `entity_mut()`, `hot_entity()`, `cold_entity()` |
| group_index | `GroupIndex` | Contiguous per-settlement + per-party entity ranges (see below) | `rebuild_group_index` | system dispatch (runtime, exploration, supply) |
| settlement_index | `Vec<u32>` | `settlement_id → idx` into `settlements` vec | `rebuild_settlement_index` | `settlement()`, `settlement_mut()` |

Note: all three are strictly **derived** from primary state and re-buildable from `entities`/`settlements`. They exist to avoid linear scans.

#### Spatial / terrain state

| Field | Type | Persistence | Purpose |
|---|---|---|---|
| tiles | `HashMap<TilePos, Tile, ahash>` | Primary | Sparse 2D tile grid |
| fidelity_zones | `Vec<FidelityZone>` | Primary (entity_ids membership is derived, re-computed each tick) | Sim-fidelity bubbles |
| build_seeds | `Vec<BuildSeed>` | Primary | Pending room-growth seeds |
| voxel_world | `VoxelWorld` | Primary | 3D chunked voxel world (physical truth) |
| nav_grids | `Vec<NavGrid>` | Derived cache | Baked 2D walkable surfaces per settlement |
| region_plan | `Option<Arc<RegionPlan>>` (`#[serde(skip)]`) | Primary (regen from seed) | Biome/elevation plan driving terrain generation |
| structural_events | `Vec<StructuralEvent>` | Per-tick buffer | Collapse/fracture events this tick |
| chronicle | `Vec<ChronicleEntry>` | Primary (bounded ring) | Narrative log |
| world_events | `Vec<WorldEvent>` | Primary (bounded) | Recent events for system queries |

#### Cache fields (all `#[serde(skip)]`)

| Field | Type | Rebuilt by | Purpose |
|---|---|---|---|
| surface_cache | `SurfaceCache` (HashMap<u64, i32>) | `scan_voxel_resources_cached` (lazy) | Fallback analytical surface-height cache |
| surface_grid | `FlatSurfaceGrid` | `warm_surface_cache` | Dense per-settlement surface-height tiles |
| cell_census | `CellCensus` (HashMap<(i32,i32),[u32;6]>) | `scan_voxel_resources_cached` (lazy) | Per-cell target-material counts for NPC discovery |
| sim_scratch | `SimScratch` | Caller clears before each use | Pooled scratch buffers, NOT persistent |

---

### WorldState.tiles

**Type:** `HashMap<TilePos, Tile, ahash::RandomState>` (state.rs:505)
**Purpose:** Sparse 2D grid of placed tiles — floors, walls, doors, furniture, farmland, workspaces, ditches. Only modified tiles stored; unmodified positions are empty. 2.0 world units per tile.

#### Tile fields
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tile_type | TileType | What kind of tile | `construction.rs:71,125,143` (wall/door placement), `action_eval.rs` (PlaceTile action) | flood_fill, interiors, buildings, pathing, movement_cost lookups |
| placed_by | Option\<u32\> | Entity that placed it (attribution) | construction, action_eval | rarely queried |
| placed_tick | u64 | Tick placed | construction | sometimes used for decay/age checks |
| floor_level | u8 | Floor index within a multi-story building (0 = ground level, 1 = second floor, etc.). Outdoor tiles use 0. Multi-story buildings occupy one `Tile` entry per (TilePos, floor_level). | construction (per-floor placement) | interior navigation (`floor_height(pos, building)`), rendering, mask predicates (overhear: same building → any floor; planar_distance + z_separation apply across floors) |

Tiles remain 2D per-floor: the `HashMap<TilePos, Tile>` keys are still `TilePos { x, y }`, but the stored `Tile` carries `floor_level`. Multi-story buildings store one `Tile` entry per occupied (x, y, floor) triple via distinct map insertions keyed on a `(TilePos, floor_level)` tuple (implementation detail; logical schema above treats `floor_level` as a `Tile` field).

#### TileType variants (state.rs:205)
Exhaustive list:
- **Terrain:** `Dirt`, `Stone`, `Water`
- **Structural:** `Floor(TileMaterial)`, `Wall(TileMaterial)`, `Door`, `Window`
- **Infrastructure:** `Path`, `Bridge`, `Fence` (blocks monsters only)
- **Agricultural:** `Farmland`
- **Furniture (in-room):** `Workspace(WorkspaceType)`, `Bed`, `Altar`, `Bookshelf`, `StorageContainer`, `MarketStall`, `WeaponRack`, `TrainingDummy`, `Hearth`
- **Defensive:** `Moat`, `TowerBase`, `GateHouse`, `ArcherPosition`, `Trap`

Per-variant gameplay role is encoded in `movement_cost()`, `is_solid()`, `is_wall()`, `is_floor()`, `is_furniture()`, `blocks_monsters_only()` on `TileType`.

#### TileMaterial variants (state.rs:246)
`Wood`, `Stone`, `Iron`. Used to parametrise wall/floor types.

#### WorkspaceType variants (state.rs:254)
`Forge`, `Anvil`, `Loom`, `AlchemyBench`, `Kitchen`, `Sawbench`.

#### BuildingFunction enum (state.rs:316)
`Shelter`, `Production`, `Worship`, `Knowledge`, `Defense`, `Trade`, `Storage`. Used by `BuildSeed.intended_function` and to compute `minimum_interior()` requirements. Not stored in `Tile` directly — a property of planned rooms.

#### Characteristics
- **Update frequency:** incremental, sparse writes. Only when NPCs `PlaceTile`/construction runs room-growth. Rarely removed (door placement rewrites single tiles).
- **GPU-friendliness:** hostile — HashMap lookup per neighbor in flood_fill. `is_enclosed`, `has_door`, `find_door_position` all do 4/8-connected scans via HashMap.
- **Derived?** No — primary state.
- **Candidates for flat-grid conversion:** yes. Spatially dense within settlements; currently paying hash cost per neighbor lookup. A chunked flat grid keyed by TilePos would be faster.

#### TilePos helpers (state.rs:156)
- `TilePos { x: i32, y: i32 }` with hand-packed `Hash` (pack to u64, single `write_u64` — ~2× faster than derive(Hash)).
- `TilePos::from_world(wx, wy) → (wx/2).floor()`
- `TilePos::to_world()` — returns tile center
- `neighbors8()`, `neighbors4()`

---

### WorldState.voxel_world (VoxelWorld)

**Type:** `VoxelWorld` (voxel.rs:443)
**Purpose:** 3D chunked voxel world — physical source of truth. Sparse; only loaded chunks stored.

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

#### ChunkPos (voxel.rs:31)
| Field | Type | Meaning |
|---|---|---|
| x, y, z | i32 | Chunk-space coord (each chunk covers CHUNK_SIZE³ voxels) |

Derive-Hash is used (3× write_i32) — not hand-optimised like TilePos, but `ahash` mitigates.

#### Chunk (voxel.rs:385)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| voxels | `Vec<Voxel>` (len = CHUNK_VOLUME = 262,144) | Dense voxel array, row-major `(z·CS + y)·CS + x` | `Chunk::set`, VoxelWorld mutations | renderer, structural_tick, scans |
| pos | ChunkPos | Identity | Constructor | mesh gen |
| dirty | bool | Any voxel changed — regen SDF/mesh | `set`, `set_voxel`, `damage_voxel`, `mine_voxel`, filled by structural_tick to `false` after processing | mesh regen, structural_tick (picks dirty chunks) |

#### Voxel (voxel.rs:326)
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

Structural health + support are per-voxel: `effective_hp() = integrity * material.properties().hp_multiplier`. `damage_voxel` at ≤0 HP sets integrity=0 (load-bearing) or replaces with Air, then triggers `cascade_collapse` upward.

#### VoxelMaterial variants (voxel.rs:90)
~45 variants: Air, natural terrain (Dirt/Stone/Granite/Sand/Clay/Gravel/Grass), fluids (Water/Lava/Ice/Snow), ores (IronOre/CopperOre/GoldOre/Coal/Crystal), placed materials (WoodLog/WoodPlanks/StoneBlock/StoneBrick/Thatch/Iron/Glass), agricultural (Farmland/Crop), additional (Basalt/Sandstone/Marble/Bone/Brick/CutStone/Concrete/Ceramic/Steel/Bronze/Obsidian), biome surfaces (JungleMoss/MudGrass/RedSand/Peat/TallGrass/Leaves), entity markers (NpcIdle/Walking/Working/Fighting/MonsterMarker — non-solid, rendering only).

Each material has: `is_solid()`, `is_fluid()`, `is_transparent()`, `hardness() → u32`, `mine_yield() → Option<(commodity_idx,f32)>`, `properties() → MaterialProperties` (hp_multiplier, fire_resistance, load_bearing, weight, rubble_move_cost, construction_cost, blast_resistance).

#### VoxelZone (voxel.rs:307)
`None`, `Residential`, `Commercial`, `Industrial`, `Military`, `Agricultural`, `Sacred`, `Underground`. For building zone tracking.

#### Surface-height paths

Signature: `surface_height: vec2 → f32` (outdoor only). Returns world-space z of the topmost solid voxel surface at planar coordinate `(vx, vy)`. Interior navigation uses `floor_height: (vec3, building_id) → f32` instead, derived from the tile `floor_level` layer the agent is on.

Three code paths, in priority order:
1. **Analytical fbm path** (`terrain::materialize::surface_height_at(vx, vy, plan, seed)`, voxel.rs:492) — used when `region_plan` is `Some`. Pure function of (vx, vy, plan, seed); **zero chunk lookups**. Vastly cheaper than chunk-walking (flamegraph: was 72% of program time pre-optimisation).
2. **Chunk-walking fallback** (`surface_height_from_chunks`, voxel.rs:503) — used when no region_plan. Walks chunk-z-slices top-down; one HashMap lookup per chunk-z. Returns `sea_level` if no solid found.
3. **Cached paths** (`surface_grid` dense tile → `surface_cache` sparse HashMap) — used by exploration scans to avoid recomputing even the analytical path ~16K times per cell.

The ground-snap cascade rule (see `spec.md` §7) reads `surface_height(pos.xy)` for outdoor agents and `floor_height(pos, building_id)` for indoor agents, then sets `pos.z = h + creature_height/2`.

#### Characteristics
- **Update frequency:** chunks added lazily (`generate_chunk` / `ensure_loaded_around`). Voxels mutated by harvest, construction, structural_tick, damage.
- **GPU-friendliness:** chunks are inherently chunked (already the right shape). HashMap wrapper is hostile; indexed/pool-allocated chunks would be GPU-friendly. Dense `Vec<Voxel>` per chunk is GPU-upload-ready (16 bytes/voxel, 4 MB/chunk).
- **Derived?** No — primary state. `region_plan` + seed regenerates chunks, but once voxels are edited, those edits are primary.

**Role for 3D agent positions.** `Agent.pos: vec3` is authoritative. Agents with volumetric `creature_type` (Dragon, Fish, Bat) can be placed anywhere inside the voxel chunk grid — inside caverns, underwater, in a flying island, inside a dungeon chamber — without a separate indoor/outdoor state flag. Ground-locked agents resolve their `pos.z` via the snap cascade (outdoor `surface_height` or indoor `floor_height` per `inside_building_id`). Spatial queries (`query::nearby_agents`) chunk against the voxel grid in 3D by default. [OPEN] whether the spatial hash is 3D-chunked across full voxel space, or 2D-grid keyed on chunk-column with per-cell z-range buckets — pick one per `spec.md` §9.

---

### WorldState.region_plan (RegionPlan)

**Type:** `Option<RegionPlan>` in WorldState (region_plan.rs:67). Also stored by value in `VoxelWorld.region_plan`.
**Purpose:** Continental-scale biome plan — grid of `RegionCell`s classifying terrain, settlements, dungeons, plus polyline rivers and road segments. Stored so chunk generation can reference the same plan after world creation.

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

#### Characteristics
- **Update frequency:** static after init. `#[serde(skip)]` on WorldState — regenerated from seed on load.
- **GPU-friendliness:** friendly. Cells are a flat `Vec`, rivers/roads can flatten via `to_gpu_cells()`, `to_gpu_rivers()` (already implemented for `feature="app"`).
- **Derived?** Deterministic function of seed — effectively primary (source of all terrain gen), but regenerable.

---

### WorldState.nav_grids

**Type:** `Vec<NavGrid>` (state.rs:545, nav_grid.rs:15)
**Purpose:** Baked 2D walkable surfaces derived from `VoxelWorld`. One per settlement area. Pathfinding (A*, flow fields) operates on NavGrid, not on VoxelWorld.

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

#### Characteristics
- **Update frequency:** rebuilt when `VoxelWorld` has structural changes in the area. `buildings.rs:rebake_nav_grids` walks the column from max_z down for each (x,y) in the footprint. Called after construction completes and when test setups add buildings.
- **Writers:** `buildings.rs:212,213` (push new nav), `buildings.rs:960,961` (test path), `buildings.rs:197` (iterate-and-rebake).
- **GPU-friendliness:** friendly — flat `Vec<NavNode>` row-major.
- **Derived?** **Yes — pure function of `voxel_world` contents.** Can be rebuilt at any time via `NavGrid::bake(world, origin, w, h, max_z)`. Serialized for convenience.

---

### FidelityZone

**Type:** `FidelityZone` (state.rs:4009), stored in `WorldState.fidelity_zones: Vec<FidelityZone>`
**Purpose:** Proximity bubble controlling simulation fidelity around a point. Entities inside the zone run at its fidelity level (High/Medium/Low/Background).

#### Fields
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique zone ID | spawn (`tick.rs:366,384,405`, `compute_high.rs:168`, `world_sim_cmd.rs:2907`) | grid/zone lookups |
| fidelity | Fidelity | `High` / `Medium` / `Low` / `Background` | `exploration::compute_exploration_for_settlement` (EscalateFidelity delta), threat systems | tick fidelity dispatch (`runtime.rs:hot_entity_fidelity`) |
| center | vec3 | World-space center | spawn | `update_grid_membership` (proximity test; radius is 3D Euclidean) |
| radius | f32 | Zone radius | spawn | membership |
| entity_ids | `Vec<u32>` | Entity IDs currently inside | `runtime::update_grid_membership` (rewritten each tick) | `has_hostiles`, `has_friendlies` queries |

#### Characteristics
- **Update frequency:** `entity_ids` re-populated every tick in `update_grid_membership`. Zones themselves added at init or on escalation.
- **GPU-friendliness:** mixed — scalar fields are friendly; `entity_ids` is a `Vec<u32>` per zone. Acceptable.
- **Derived?** `entity_ids` is **derived** (recomputed from entity positions + zone center/radius each tick). The zone definition (id/fidelity/center/radius) is primary.

---

### BuildSeed

**Type:** `BuildSeed` (state.rs:399), stored in `WorldState.build_seeds: Vec<BuildSeed>`
**Purpose:** A placed room-growth seed. NPC sets a seed, the room-growth automaton (construction.rs) enlarges outward until enclosed.

#### Fields
| Field | Type | Meaning |
|---|---|---|
| pos | TilePos | Seed tile position |
| intended_function | BuildingFunction | What room function this targets (drives `minimum_interior`) |
| minimum_interior | u32 | Required interior tile count for completion |
| placed_by | u32 | Entity ID that set the seed |
| tick | u64 | Tick placed |
| complete | bool | Marked true when grown successfully, OR when stalled past `MAX_SEED_ATTEMPTS` |
| attempts | u16 | Room-growth attempts performed — stall detection |
| last_interior_size | u16 | Last observed interior size; if unchanged across attempts, seed is stalled |

#### Characteristics
- **Update frequency:** added by `action_eval.rs:1178` (NPC PlaceBuildSeed action). Mutated by `construction.rs` room-growth (attempts, last_interior_size, complete). Pruned when `complete=true`.
- **GPU-friendliness:** tiny — flat `Vec<BuildSeed>`, ~40 bytes each, easily GPU-shaped.
- **Derived?** Primary state — must persist.

---

### StructuralEvent

**Type:** `StructuralEvent` (state.rs:436), stored in `WorldState.structural_events: Vec<StructuralEvent>`
**Purpose:** Per-tick events emitted by voxel collapse/fracture. **Cleared at tick start** (`runtime.rs:1540`) — ephemeral.

#### Variants
- `FragmentCollapse { chunk_x, chunk_y, chunk_z, fragment_voxel_count: u32, cause: CollapseCase }`
- `StressFracture { chunk_x, chunk_y, chunk_z, cluster_mass: f32, material_strength: f32 }`

#### CollapseCase enum (state.rs:428)
- `NpcHarvest` — caused by voxel_harvest (tree felling, mining)
- `NpcConstruction` — placed a voxel that destabilised neighbours
- `Natural` — organic collapse from structural_tick

#### Characteristics
- **Update frequency:** per-tick buffer. Cleared at start of each tick (`runtime::tick`). Appended by `structural_tick.rs:71`.
- **GPU-friendliness:** small — tagged union. OK.
- **Derived?** Consumed same tick; technically a per-tick output buffer, not persistent state.

---

### SimScratch (NOT persistent state)

**Type:** `SimScratch` (state.rs:344)
**Purpose:** Pooled scratch buffers reused across tick systems. **Not persistent. Cleared + refilled within a single function call.** Avoids per-tick Vec/HashMap allocations (pre-pooling: ~55 page faults per tick, 220KB/tick allocator churn).

`Clone` returns `Default` intentionally — cloning a WorldState shouldn't duplicate scratch allocations.

#### Sub-buffer ownership
| Buffer | Owner | Purpose |
|---|---|---|
| snaps | `action_eval::evaluate_and_act` | Read-only entity snapshots for scoring |
| snap_grid | `action_eval` | Snap indices by spatial cell |
| snap_grids_typed | `action_eval` | Kind-typed spatial grids (resources/buildings/combatants) |
| deferred | `action_eval` | Deferred action decisions |
| npc_indices | `exploration::scan_all_npc_resources` | NPC indices to scan |
| npc_pos_voxel | `exploration` | NPC voxel positions cached for step 3 |
| visible_cells | `exploration` | Visible cell set |
| flood_visited | `construction::flood_fill_with_boundary` | Generational visited tag (128×128 flat grid, u16 gen) |
| flood_current_gen | `construction` | Current generation tag |
| flood_queue | `construction` | BFS queue |
| flood_interior | `construction` | BFS result interior |
| flood_boundary | `construction` | BFS result boundary |

#### Characteristics
- **Update frequency:** cleared + filled within one function call each.
- **GPU-friendliness:** N/A — CPU-side pools.
- **Derived?** Ephemeral. Never read across boundaries.

---

### GroupIndex

**Type:** `GroupIndex` (state.rs:1032)
**Purpose:** Contiguous per-settlement / per-party entity ranges. After `rebuild_group_index()`, entities are sorted by `(settlement_id, party_id)` so settlement members are adjacent. Systems iterate a slice instead of scanning all entities.

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
`settlement_entities(sid)`, `settlement_npcs(sid)`, `settlement_buildings(sid)`, `settlement_monsters(sid)`, `party_entities(pid)`, `unaffiliated_entities()` — all return `Range<usize>`.

#### Characteristics
- **Update frequency:** rebuilt by `rebuild_group_index()` after structural entity changes. Called in `rebuild_all_indices` after spawn/despawn. Runtime invokes when `entities.len() != hot.len()` (runtime.rs:1764).
- **GPU-friendliness:** friendly — flat `Vec<(u32,u32)>` arrays.
- **Derived?** **Yes — purely derived from `entities` + `settlement_id`/`party_id` membership.** `#[serde(skip)]`-able (currently serialised with Default).

---

### SurfaceCache (exploration.rs:222)

**Type:** `pub type SurfaceCache = HashMap<u64, i32, ahash::RandomState>`
**Key:** packed `(vx as u32) << 32 | vy as u32` via `pack_xy`.
**Value:** surface z-height.

- Populated lazily by `scan_voxel_resources_cached` on HashMap miss.
- Fallback for positions outside any FlatSurfaceTile in `surface_grid`.
- Persistent across ticks (world is mostly static; valid as long as region_plan + seed unchanged).
- **Derived — pure function of (vx, vy, region_plan, seed).** Regenerates on demand.
- `#[serde(skip)]`.

### FlatSurfaceGrid / FlatSurfaceTile (exploration.rs:229, 253)

#### FlatSurfaceTile
| Field | Type | Meaning |
|---|---|---|
| origin_x, origin_y | i32 | Tile origin in voxel-space |
| width, height | i32 | Tile dimensions |
| heights | `Vec<i16>` | Row-major `(dy·width + dx)` surface z-heights |

#### FlatSurfaceGrid
| Field | Type | Meaning |
|---|---|---|
| tiles | `Vec<FlatSurfaceTile>` | One tile per settlement region |

- Populated by `warm_surface_cache` per settlement (exploration.rs:497).
- `FlatSurfaceGrid::get(vx,vy)` linearly scans tiles — fine for small N (1–10 settlements).
- **30× faster than HashMap lookup** and 10× less memory; on-cache-hit path for per-cell census.
- **Derived** — rebuilt from analytical path.
- **GPU-friendly** — flat `Vec<i16>` row-major.
- `#[serde(skip)]`.

### CellCensus (exploration.rs:203)

**Type:** `pub type CellCensus = HashMap<(i32,i32), [u32; 6], ahash::RandomState>`
**Key:** `(cell_x, cell_y)` where each cell spans `RESOURCE_CELL_SIZE = 128` voxels.
**Value:** `[count_wood, count_iron, count_copper, count_gold, count_coal, count_crystal]` — 6 target materials.

- Populated lazily when any NPC can see the cell (`scan_voxel_resources_cached`, exploration.rs:705).
- Persistent for the run. Invalidation on voxel edits is a known tech-debt refinement; resources change slowly and NPCs reconfirm on harvest.
- **Derived** — pure function of chunk contents in cell's surface band. Rebuildable.
- **GPU-friendliness:** hostile (HashMap). Could migrate to a flat grid since cells align to 2×CHUNK_SIZE.
- `#[serde(skip)]`.

---

### WorldState.chronicle (narrative log)

**Type:** `Vec<ChronicleEntry>` (state.rs:571; entry at state.rs:4893)
**Purpose:** Bounded ring buffer of narrative events — what happened in the world. Human-readable.

#### ChronicleEntry
| Field | Type | Meaning |
|---|---|---|
| tick | u64 | When it happened |
| category | ChronicleCategory | Battle / Quest / Diplomacy / Economy / Death / Discovery / Crisis / Achievement / Narrative |
| text | String | Human-readable text |
| entity_ids | `Vec<u32>` | Entities involved |

#### Characteristics
- **Update frequency:** appended by ~20 systems (battles, quests, legends, family, death, warfare, settlement_founding, prophecy, oaths, outlaws, sea_travel, etc.).
- **GPU-friendliness:** hostile — contains `String`.
- **Derived?** Primary (lore state; append-only), bounded.

### WorldState.world_events

**Type:** `Vec<WorldEvent>` (state.rs:579; enum at state.rs:4919)
**Purpose:** Recent events for system queries (bounded).

#### WorldEvent variants
`Generic{category,text}`, `EntityDied{entity_id,cause}`, `QuestChanged{quest_id,new_status}`, `FactionRelationChanged`, `RegionOwnerChanged`, `BondGrief`, `SeasonChanged`, `BattleStarted`, `BattleEnded`, `QuestPosted`, `QuestAccepted`, `QuestCompleted`, `SettlementConquered`.

#### Characteristics
- **Update frequency:** appended by relevant systems, bounded.
- **GPU-friendliness:** hostile — some variants contain `String` (`Generic`, `EntityDied`).
- **Derived?** Primary (event log).

---

### Summary

- **Primary state** (irreplaceable): `tick`, `rng_state`, `next_id`, `tiles`, `voxel_world.chunks`, `voxel_world.sea_level`, `region_plan` (regenerable from seed), `build_seeds`, `chronicle`, `world_events`, `fidelity_zones` (zone definitions — id/fidelity/center/radius), `structural_events` (per-tick buffer).
- **Derived state** (rebuildable from primary): `entity_index`, `group_index`, `settlement_index`, `surface_cache`, `surface_grid`, `cell_census`, `nav_grids`, `max_entity_id`, `fidelity_zones[].entity_ids` (recomputed every tick in `update_grid_membership`).
- **Scratch state** (ephemeral, pooled, NOT persistent): `SimScratch` and all sub-buffers — cleared + filled + consumed within one function call.
- **GPU-hostile shapes:** `HashMap<TilePos, Tile>`, `HashMap<ChunkPos, Chunk>`, `HashMap<u64, i32>` (surface_cache), `HashMap<(i32,i32),[u32;6]>` (cell_census), `Vec<ChronicleEntry>`/`Vec<WorldEvent>` (contain `String`), `VecDeque<TilePos>` (flood_queue).
- **GPU-friendly shapes already in place:** `Chunk.voxels: Vec<Voxel>` (dense, 16B/voxel), `NavGrid.nodes: Vec<NavNode>` (row-major), `FlatSurfaceTile.heights: Vec<i16>` (row-major), `RegionPlan.cells: Vec<RegionCell>` (flat, with `to_gpu_cells()` already implemented), `RiverPath.points: Vec<(f32,f32)>` with `to_gpu_rivers()`, `GroupIndex` range arrays.
- **Candidates for flat-grid conversion:**
  - `tiles` (spatially dense within settlements; currently hashing per neighbor in flood_fill / is_enclosed / has_door).
  - `voxel_world.chunks` (inherently chunked — just needs indexed/slab-pooled chunk storage rather than HashMap).
  - `cell_census` (cells align to 2×CHUNK_SIZE; a flat grid keyed by cell coord would be GPU-friendly).
  - `surface_cache` is already superseded by `surface_grid` on the hot path.
