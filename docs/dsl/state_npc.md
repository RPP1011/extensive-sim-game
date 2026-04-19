# Per-Agent Internal State

## Agent (top-level)

Universal entity for any agentic actor — humans, wolves, dragons, goblins. The same struct, distinguished by `creature_type` and configuration.

### Identity & Lifecycle
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | AgentId | Unique agent ID, stable across ticks | spawn | all systems (lookups) |
| creature_type | CreatureType | Human, Elf, Dwarf, Wolf, Goblin, Dragon, ... | spawn | mask predicates (can_speak, predator_prey), observation features, narrative |
| alive | bool | Whether agent is active | death cascade | action_eval (filtering), movement |
| level | u32 | Power tier | progression (level-up), template init | combat damage scaling, action eval |

### Physical State
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| pos | vec3 | World-space position (z authoritative) | movement system (advance_movement), post-movement ground-snap cascade, spawning | action_eval (distance / planar_distance / z_separation), threat assessment, pathfinding |
| grid_id | Option<u32> | Settlement grid assignment (for nav caching) | movement entry/exit, grid rebinding | pathfinding, movement cost |
| local_pos | Option<vec3> | Relative position within room (for interiors) | interiors system (room navigation) | rendering, room-based spatial queries |
| move_target | Option<vec3> | Desired destination (set by goal system) | goal system, movement subsystem | movement system (pathfinding goal) |
| move_speed | f32 | Base units/tick | template init, item bonuses, status effects | movement speed calc, action duration |
| move_speed_mult | f32 | Multiplier (default 1.0) | status effects (stun/slow) | effective movement calc |

**Ground-locked vs volumetric creature_types.** Ground-locked types (Human, Elf, Dwarf, Wolf, Goblin) carry a post-movement constraint: a `@phase(post)` cascade rule keyed on `MovementApplied` snaps `pos.z = surface_height(pos.xy) + creature_height/2` outdoors, or `floor_height(pos, building) + creature_height/2` when `inside_building_id` is set. The constraint is applied automatically based on `creature_type` config. Volumetric types (Dragon, Fish, Bat) skip the snap and retain free z motion. `creature_height` is a derived field from `creature_type` config.

### Combat/Vitality
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

### Sub-structures
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| data | AgentData | Agent state (see AgentData section below) | agent_inner, action_eval, social/economic systems | all behavioral systems |
| inventory | Inventory | Commodity + gold storage (every agent has one) | work (production), trade, eating, looting | action_eval (resource availability), trader intent |
| memberships | Vec<Membership> | Groups this agent belongs to (faction, family, guild, religion, party, pack, settlement). Multi-membership produces emergent loyalty conflicts. | JoinGroup / LeaveGroup actions; spawn-time defaults | mask predicates (group-gated actions), observation features (memberships slot), `is_hostile` / `is_friendly` views |
| capabilities | Capabilities | Jump/climb/tunnel/fly/siege flags + can_speak / can_hear / hearing_range / can_build / can_trade flags derived from `creature_type`. `can_speak` / `can_hear` / `hearing_range` drive `Communicate` / `Announce` / overhear mask eligibility; `fly` gates z-separation combat predicates | template init (from `creature_type` config) | pathfinding obstruction, construction defense, mask predicates (`Communicate`, `Announce`, overhear), observation features |

Items the agent has equipped are referenced by ID from `data.equipped_items`. The Item entities themselves are catalogued in this doc's Item section.

---

## Membership

A single membership relating an agent to a group (faction, family, guild, religion, party, pack, settlement, etc.).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| group_id | GroupId | Which group | JoinGroup event | mask predicates, observation features |
| role | GroupRole | Member, Officer, Leader, Founder, Apprentice, Outcast | JoinGroup, PromoteEvent | mask predicates (leader-gated actions), reputation calculations |
| joined_tick | u64 | When joined | JoinGroup event | tenure calculation, observation feature |
| standing | f32 | -1..1 reputation within the group | various social events involving the group | mask predicates (Outcast cannot vote, etc.) |

`Group` itself (with leadership chain, treasury, standing relations to other groups, internal rules) is documented in `state_aggregate.md`.

---

## StatusEffect

Temporary state modifier on an entity.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| kind | StatusEffectKind | Enum: Stun, Slow{factor}, Root, Silence, Dot/Hot{rate,interval}, Buff/Debuff{stat,factor} | apply_flat (effect dispatch) | movement (Slow/Root penalty), action (Silence blocks), damage (Dot/Hot tick) |
| source_id | u32 | Entity who applied this effect | apply_flat (effect source) | credit for damage/control |
| remaining_ms | u32 | Time left (milliseconds, converted from ticks) | cooldown system (decay per tick) | effect expiry check |

---

## Needs (6-dimensional)

Maslow-inspired primary drives. Range 0–100. Higher = more satisfied.

### Need Dimensions
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

## Personality (5-dimensional)

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

## Emotions (6-dimensional, Transient)

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

## Aspiration (Phase B: Medium-term Orientation)

Personality-weighted need gap vector. Recomputed every 500 ticks. Provides 100-tick horizon behavioral bias.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| need_vector | [f32; 6] | Weighted gap (100 - need[i]) × personality[i], normalized | agent_inner (recompute every 500 ticks from needs + personality) | action_eval (action bias scoring), goal_eval (priority weighting) |
| vector_formed_at | u64 | Tick when last recomputed | agent_inner (formed tick) | recompute interval check (500 tick delta) |
| crystal | Option<Crystal> | Concrete target (Entity/Class/Building/Location/Resource) bound to a need dimension | salient events (quest, discovery, relationship), failure resets | action goals (concrete pursue path vs. need balancing) |
| crystal_progress | f32 | Completion toward crystal (0.0–1.0) | systems progressing crystal goal (work toward building, gather resource) | crystal_last_advanced tick check, UI display |
| crystal_last_advanced | u64 | Last tick crystal_progress increased | crystal progress events | stall detection (no progress → abandon crystal) |

---

## Memory

Event log + semantic beliefs formed from experience.

### Memory.events (Ring Buffer)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| events | VecDeque<MemoryEvent> | Capped at 20 entries (oldest discarded) | event recording (memory.record_event) | belief formation, narrative, relationship updates |

### MemoryEvent
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

## Source (enum)

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

### Memory.beliefs (Semantic Layer)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| beliefs | Vec<Belief> | Semantic conclusions from events (LocationDangerous/Safe, EntityTrustworthy/Dangerous, SettlementProsperous/Poor, SkillValuable, FactionFriendly/Hostile, Grudge, HeardStory) | belief_formation system (from memory events, social gossip, trade info) | action_eval (destination safety), relationship trust init, goal location bias |

**Belief.confidence** [0.0, 1.0] — decays without update (~0.95/tick absent reinforcement).

---

## Relationship (Per-pair, NPC → NPC)

Directional relationship from one NPC toward another (asymmetric).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| trust | f32 | [-1.0, 1.0]. Positive = friendly, negative = hostile. Starts 0.0 | interaction events (trade +0.1, combat -0.3, betrayal -0.5, rescue +0.3), time passage (slow decay) | action_eval (trade partner preference, combat targeting), marriage likelihood |
| familiarity | f32 | [0.0, 1.0]. Grows with proximity and interaction | co-location, trade, work, socializing (+0.02–0.05/interaction) | relationship influence weight, interaction frequency |
| last_interaction | u64 | Tick of last trade/combat/work together | interaction events | relationship weight decay, "forgotten" threshold |
| perceived_personality | PerceivedPersonality | Mental model of other's traits (see section below) | observe_action method (from witnessed NPC behavior) | compatibility check, collaboration prediction, gossip basis |
| believed_knowledge | Bitset<32> | Theory-of-mind: bits self believes the other agent knows, indexed by knowledge domain (Combat, Trade, Family, Politics, Religion, Craft, ...). Same bit layout as `self.knowledge_domain_bits`. | set when self witnesses or is told about the other's domain-tagged actions ("saw them fight → Combat bit"); slow decay without reinforcement | mask predicates (e.g. `Deceive(t) when ¬believed_knowledge(t, Fact::X)`), observation features, gossip targeting |

**Capped at 20 relationships per NPC** (evict lowest familiarity if over 20).

### PerceivedPersonality (Theory of Mind)

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| traits | [f32; 5] | Estimated [risk_tol, social, ambition, compassion, curiosity] | observe_action (action→trait signal, alpha learning) | compatibility calc with own personality, behavior prediction |
| confidence | [f32; 5] | Per-trait confidence [0.0, 1.0] | observe_action (+=0.01×attention per action) | trait update alpha (lower confidence → higher alpha), gossip weight |
| observation_count | u32 | Total action observations | observe_action (count++) | authority weight (more observations → less influence) |

**Emergent?** Derived entirely from observed actions. Reconstructible from action history.

---

## Needs-Driven Behavioral State

### Personality-Weighted Need Gap
**Emergent.** Aspiration.need_vector = personality-weighted normalization of (100 - needs[i]).

---

## Goal & Action Execution

### Goal (Priority-Sorted Stack)

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

### GoalStack (Interrupt/Resume Mechanism)

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| goals | Vec<Goal> | Priority-sorted (descending), cap 8 | push (dedup + re-sort), pop (completion), remove_kind (override) | current() accessor (top goal drives agent), interrupt check (new high-priority goal preempts) |

**Behavior:** Goals are processed by priority. Eating/fleeing can interrupt work. On interrupt completion, the previous goal resumes (hysteresis).

---

## AgentData

Container for all agent-specific state — needs, emotions, personality, relationships, classes, equipment, work state, social ties, learned biases. Carried by every Agent regardless of `creature_type`.

### Identity & Metadata
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| name | String | Generated from entity_id + seed deterministically | name generation (spawn) | UI display, speech, quest text |
| adventurer_id | u32 | Campaign Adventurer link (0 if unlinked) | party entry, class assignment | party queries, campaign persistence |
| creature_type | CreatureType | Enum: Citizen (default), PackPredator, Territorial, Raider, Megabeast | constructor (template.kind), monster spawn | personality preset, ability unlock, action gating |

### Economic & Trade
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

### Location & Settlement Affiliation
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| home_settlement_id | Option<u32> | Settlement where NPC is "based" for economics | settlement residence, movement | work assignment building location, trade route origin, relocation trigger |
| home_building_id | Option<u32> | Primary residence (where Rest goal leads) | settlement housing, family founding | rest location, shelter need satisfaction |
| work_building_id | Option<u32> | Assigned workplace (building entity ID) | work assignment system | work goal target, work_state machine |
| inside_building_id | Option<u32> | Currently inside building (entity ID) | movement (enter/exit interiors) | interior navigation, room access, shelter provision |
| current_room | Option<u8> | Room index within building.rooms | interiors system (room occupation) | room-based AI (separate grid), visual location |
| home_den | Option<vec3> | Lair position for monsters (Territorial/PackPredator) | constructor (spawn position) | monster return-home behavior, territorial defense |

### Movement & Pathfinding Cache
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| cached_path | Vec<(u16, u16)> | Waypoints on settlement grid (col, row) | pathfinding (A* result) | movement (follow path), path expiry on goal change |
| path_index | u16 | Current waypoint index | movement (advance on arrival), goal change (reset) | movement steering, arrival detection |
| goal_stack | GoalStack | Priority-sorted goal list (see GoalStack section) | goal_eval (push/pop), interrupts (prioritize) | agent_inner (execute top goal), action dispatch |

### Work & Labor State Machine
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| work_state | WorkState | Enum: Idle, Travel, Work, Carry. State machine for work loop | work system (state transition) | work action dispatch, commodity accumulation, haul completion |
| action | NpcAction | Enum: Idle, Walking, Eating, Working{activity}, Hauling, Fighting, Socializing, Resting, Building, Fleeing, Trading, Harvesting | action system (per goal/state) | UI display, animation state, action duration (ticks_remaining) |
| behavior_production | Vec<(usize, f32)> | [(commodity_index, production_rate_per_tick), ...] | work assignment (set per job) | production accumulation, yield bonus, passive income |

### Skill & Class System
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| class_tags | Vec<String> | Lowercase class names (e.g. ["miner", "farmer"]) | progression (class grant), apprenticeship | ability unlock, behavior profile bias, passive effect generation |
| classes | Vec<ClassSlot> | Acquired classes with levels and XP | progression (level-up from XP), class grant | ability unlock threshold, stat bonus (HP/armor/damage), behavior tag generation |
| behavior_profile | Vec<(u32, f32)> | Sorted tag_hash → accumulated_weight pairs | accumulate_tags per action (all systems), periodic xp_grant (class xp += behavior_profile[tag]) | passive effect calc, action bias weighting, tag lookup (binary search) |

### Relationships & Social (Phase C/D)
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

### Perception & Resource Knowledge
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| known_resources | HashMap<u32, ResourceKnowledge> | Discovered resource entities (position, type, observation_tick) | exploration (scan for resources nearby), gossip (hear of resource), stale >2000 ticks | gatherer targeting, trade route planning, known good source preference |
| known_voxel_resources | Vec<VoxelResourceKnowledge> | Ore veins, stone, etc. (voxel coordinates) | voxel exploration, mining gossip | miner targeting, mining site navigation |
| harvest_target | Option<(i32, i32, i32)> | Currently-being-mined voxel (chunk x, y, z) | harvesting system (voxel mining) | harvest progress tracking, harvest completion |

### Psychological & Emotional
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

### Campaign System Fields (Migrated)
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

### Current Action & Intention
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| current_intention | Option<(NpcAction, f32)> | Committed action + utility score at commitment time | action system (action commit with utility), goal change (clear) | action stickiness (hysteresis), action_type_id feedback |
| intention_ticks | u32 | Ticks executing current intention | action system (increment per tick), action reset (clear on change) | action switching penalty, commitment duration (prevents thrashing) |

### Equipment & Items
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| equipment | Equipment | Legacy quality levels (weapon/armor/accessory) | progression (quality grant), crafting | stat bonus calc (mostly replaced by item entities) |
| equipped_items | EquippedItems | Item entity IDs in slots (weapon/armor/accessory) | equipping system (pick up item), unequipping (drop/swap) | stat bonus lookup (effective_quality × rarity × durability) |

### Passive & Active Abilities
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| passive_effects | PassiveEffects | Simulation-level bonuses (production_mult, trade_bonus, aura_radius, etc.) | PassiveEffects::compute from behavior_profile (on class level-up) | work production (yield boost), trade (price boost), aura application |
| world_abilities | Vec<WorldAbility> | Parsed DSL abilities usable in world sim | ability parsing (from class_tags), cooldown reset | ability application (rally, fortify, reveal, etc.), cooldown check |

---

## BuildingData

Structure definitions for player-placed buildings.

### Identity & Type
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| building_type | BuildingType | Enum: Shelter, Farmhouse, Mine, Laboratory, etc. (20+ types) | constructor (building placement) | function determination, capability unlock |
| settlement_id | Option<u32> | Owning settlement | constructor (placement), settlement system | economic integration, NPC affiliation |
| name | String | Display name | constructor (type default), renaming system | UI, narrative, identity |

### Spatial & Structure
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| grid_col, grid_row | u16 | Position on settlement 128×128 grid | constructor | room/workspace layout, collision, interior navigation |
| footprint_w, footprint_h | u16 | Size in grid cells | constructor (type-dependent) | building extent, placement collision check |
| rooms | Vec<Room> | Interior room definitions (floor plan) | construction growth (room addition) | NPC room occupation, function distribution |
| tier | u8 | Upgrade level | construction (tier advancement) | capacity scaling, stat scaling |

### Occupancy & Workforce
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| resident_ids | Vec<u32> | NPC residents (for Shelter-type buildings) | housing assignment system | resident count, social interaction location |
| worker_ids | Vec<u32> | NPC workers assigned to this building | work assignment system | production efficiency, wage payment, work location |
| residential_capacity | u32 | Max residents | type default, upgrades | housing scarcity, population pressure |
| work_capacity | u32 | Max workers | type default, upgrades | labor bottleneck, staffing urgency |
| worker_class_ticks | Vec<...> | Per-class worker tick counts (for class-specific production) | work system (time logged by class) | production quality calc, class-specific yields |

### Production & Storage
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| storage | [f32; NUM_COMMODITIES] | Commodity inventory | production (accumulate), trade (buy/sell), haul (receive) | inventory cap check, supply for trade, haul source |
| storage_capacity | f32 | Total commodity cap | type default, upgrades | inventory overflow management, trade limiting |

### Construction & Status
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

## Room (Interior Floor Plan)

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

## Inventory (Portable Commodity Storage)

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| commodities | [f32; NUM_COMMODITIES] | Per-commodity quantity | work (produce), eating (consume), trade (transfer), haul (load) | trader intent (have/want calc), action_eval (resource availability) |
| gold | f32 | Currency | trade (cost transfer), wage payment, looting | purchasing power, loan eligibility |
| capacity | f32 | Total weight cap | constructor (NPC 50.0, building 500.0) | encumbrance check, overflow prevention |

**Updated by:** Nearly all economic/production systems.

---

## Derivation Graph

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

## Update Frequency Summary

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

## Notes on Determinism & Serialization

- **hot/cold split**: `Entity` is unpacked into `HotEntity` (cache-line scalars) and `ColdEntity` (heap-heavy refs) for iteration performance. Both are serialized via entity.split() / Entity::from_parts().
- **Scratch buffers (SimScratch)**: #[serde(skip)] — ephemeral per-tick, regenerated on demand. Not serialized.
- **entity_index, group_index**: #[serde(skip)] — rebuilt on load via rebuild_all_indices().
- **Determinism**: All entity randomness via entity_hash(id, tick, salt) — never thread_rng(). Same (id, tick) always gives same result.
- **Tag hashing**: Compile-time FNV-1a hash(tag_name) for tag_hash constants — used for personality trait matching, class lookup, behavior_profile binary search.

