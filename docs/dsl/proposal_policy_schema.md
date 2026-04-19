# Proposal: ECS DSL Policy Schema

Current schema for the policy/observation/action interface. Working doc — `[OPEN]` markers flag unresolved decisions. Key commitments:

- **Single neural backend** for production. Utility kept as bootstrap/baseline only. LLM is research-only.
- **Leaders aren't smarter** — role differentiation via action mask + downstream cascade impact, not a more capable policy.
- **Action vocabulary** is ~19 verbs (four universal macro mechanisms in `proposal_universal_mechanics.md` + micro primitives including `Communicate` / `Read`).
- **Spatial + non-spatial slots** — top-K-by-distance for physical encounters; `known_actors[K]` and `known_groups[K]` for non-spatial references.
- **Hierarchical action heads** (macro/micro) for training stability with rare actions.
- **GPU compilation scoped** to specific kernels (observation packing, mask, batched neural forward); cascade rules + cross-entity predicates stay CPU.
- **Rich observation** (~1600 floats per agent) covers full state richness.
- **Entity model is Agent + Item + Group** (+ optional Projectile). Buildings and resources are derived views over spatial data.
- **Inter-agent disposition is per-pair relationships + group standings.** `is_hostile(a, b)` is a derived view; multi-group membership is first-class.

---

## 1. Premises

1. **Event-sourced runtime.** State mutation replaced by typed events; current state is a fold over events + entity baseline. ~19 of 158 systems are truly essential physics; the rest are emergent agentic actions or derived views.
2. **Single backend.** All agents run the same neural model. Utility scoring exists only as bootstrap (initial trajectories), regression baseline, and fallback during training. No long-term polymorphic-backend maintenance.
3. **Role power = mask + cascade.** A leader and a commoner think the same way (same model, same observation features). Leaders get more options because their mask passes verbs commoners' masks reject, and their actions cascade to higher-impact events.
4. **Universal action vocabulary.** ~19 verbs total: 4 macro mechanisms (PostQuest, AcceptQuest, Bid, Announce) + ~15 micro primitives (including `Communicate`, `Read`). Specified in `proposal_universal_mechanics.md`.
5. **Rich observation.** ~1500–1800 floats per agent, derived from `state_npc.md` + `state_aggregate.md` + `state_world.md`. Combat-style ~30-feature observations throw away the bulk of decision-relevant signal.
6. **Observation = ML training contract.** Once weights ship, the schema is frozen. Schema versioning + migration are first-class concerns.
7. **Slot-based for spatial; named-reference for non-spatial.** Top-K-by-distance covers physical encounters. `known_actors` / `known_groups` slots cover entities the agent knows about regardless of proximity (group leaders, distant rivals, absent spouses, named adversaries).
8. **Agent + Item + Group entities.** Agents (humans, wolves, dragons — distinguished by `creature_type` tag + config + group memberships) act in the world. Items have path-dependent owner/durability/history. Groups are the universal social-collective primitive (faction, family, guild, pack, religion, party, settlement). Buildings and resources are derived views over the world's tiles + voxels + harvest event log.
9. **Disposition is per-pair + per-group.** Each agent carries `Relationship` records and `memberships` in groups; group standings (Allied / Tense / AtWar / Vassal / ...) drive collective dynamics. Multi-group membership produces emergent loyalty conflicts.

---

## 2. The schema — five components

### 2.1 Observation

Per-agent packed feature tensor. Total budget: ~1700 floats per agent.

#### 2.1.1 Self atomic features (~60)

Vitals (12): hp_pct, max_hp_log, shield_pct, armor, magic_resist, attack_damage, attack_range, move_speed, weapon_durability, armor_durability, weapon_attack_bonus, armor_def_bonus

Creature/role (14): creature_type_one_hot(~8), creature_height, can_speak, can_hear, hearing_range, can_fly, has_inventory, can_build, can_trade, is_leader_anywhere, is_outlaw, wanted_level (speak/hear/hearing_range/fly/creature_height derived from creature_type; others state)

Needs (6): hunger, safety, shelter, social, purpose, esteem

Emotions (6): joy, anger, fear, grief, pride, anxiety

Personality (5): risk, social, ambition, compassion, curiosity

Derived (3): mood (view), morale_pct, focus (view)

Body/equipment summary (~10): inventory commodity counts (8), n_equipped_items, equipment_quality_avg

#### 2.1.2 Self contextual (~155)

Aspiration: need_vector(6), intensity, has_crystal, crystal_progress
Classes: total_level, num_classes, class_tag_bits(16), top_3_classes ((id_one_hot(16), level_norm) per slot)
Behavior profile top-K=8: (tag_id_one_hot(16), weight_log, exists)
Current state: economic_intent_one_hot(4), work_state_one_hot(~6), current_goal_kind_one_hot(~10), boolean flags (has_active_party, is_adventuring, is_at_home, is_at_work_structure), intention_ticks_log
Memory beliefs summary: counts of LocationDangerous, LocationSafe, EntityTrustworthy, EntityDangerous, friend_deaths, starvations_witnessed, recent_witness_count_5tick, recent_witness_count_50tick

Information self-summary: `knowledge_domain_bits[32]` (bitset over coarse topic categories: Combat, Trade, Family, Politics, Religion, Craft, ...; bit set if any memory event of that category exists), `memory_fill_pct` (fraction of the memory ring currently populated)
Economic: gold_log, debt_log, credit_history, income_rate_log, n_active_contracts
Learned biases: action_outcome_EMA per ActionKind (~16 in collapsed vocabulary), price_beliefs (8), cultural_bias per major action_tag (~14)
Social ties (per-agent): has_spouse, n_children, has_mentor, has_apprentice, mentor_lineage_depth, n_close_friends, spouse_alive, reputation_summary, fame_log

#### 2.1.3 Group memberships (~130)

Each agent can belong to multiple groups (faction, family, guild, religion, party, hunting pack, criminal cabal). Multi-membership produces emergent loyalty conflicts.

```
memberships[K=8]             groups self belongs to, sorted by relevance
                             (rank in group × group activity)
  group_kind_one_hot(8)      Faction, Family, Guild, Religion, Party,
                             Pack, Settlement, Other
  my_role_in_group_one_hot(6) Member, Officer, Leader, Founder,
                             Apprentice, Outcast
  group_size_log
  my_tenure_log              ticks since I joined
  my_standing_in_group       (-1, 1) reputation within the group
  group_activity_log         recent events involving the group
  group_intel_velocity       EWMA of Announce/Communicate events
                             propagating through this group — a derived
                             observation metric, NOT stored state
  is_active_party            (this group is currently in a quest party)
  exists
                                                                           ~13 × 8 = 104
```

Plus self summary atoms (~24): n_total_groups, n_political_groups, n_kin_groups, n_religious_groups, n_economic_groups, n_party_groups, n_groups_in_active_quest, primary_group_kind_one_hot(8), authority_in_top_group, leadership_count (groups where I'm Leader), founder_count.

#### 2.1.4 Spatial slot arrays (~1030)

```
nearby_actors[K=12]          sorted by 3D distance           45 × 12 = 540
                             (creature_type discriminates wolves
                              from humans, etc.)
nearby_resources[K=8]        sorted by 3D distance           13 ×  8 = 104
                             derived view over voxel materials + harvest events
nearby_structures[K=6]       sorted by 3D distance           15 ×  6 =  90
                             derived view over tile/voxel regions
threats[K=8]                 sorted by time_to_impact        12 ×  8 =  96
recent_memory_events[K=12]   from latest                     22 × 12 = 264
                             (+1 for source_one_hot_bucket)
cooldowns[K=8]               for current verbs                10 ×  8 =  80

** Per-actor slot features (~40 floats):
   relative_pos: vec3 (x, y, z)                                              (3)
   z_separation_log       log-normalized |self.z - other.z|                  (1)
   creature_type_one_hot(~8)                                                 (8)
   hp_pct, level_log, role_tag_bits(8)                                      (10)
   relationship_valence (signed), relationship_familiarity                   (2)
   n_shared_groups_log, n_opposed_groups_log, closest_shared_group_kind(8)  (10)
   is_in_active_quest_with_me, is_spouse, is_kin, is_mentor, is_apprentice   (5)
   info_source_one_hot[5]  Witnessed / TalkedWith / Overheard / Rumor /
                           NeverMet — from MemoryEvent.source for the
                           most recent memory referencing this actor          (5)
   exists                                                                    (1)
   →                                                                       (~45)
```

Per-pair relationship valence + group-overlap counts carry inter-agent disposition. Hostility is derived per-pair via `is_hostile(self, other) = relationship_valence < HOSTILE_THRESH ∨ groups_at_war(self, other) ∨ predator_prey(self.creature_type, other.creature_type)`.

#### 2.1.5 Non-spatial slot arrays (~350)

Entities the agent knows about regardless of proximity — group leaders, distant rivals, absent spouses, named adversaries.

```
known_actors[K=10]           agents the agent knows about by id
  relative_or_absent_pos: vec3, hp_pct,
  z_separation_log,
  relationship_valence, relationship_familiarity,
  n_shared_groups, n_opposed_groups,
  relationship_kind_one_hot(8) (Spouse, Kin, Mentor, Apprentice,
                                 Friend, Rival, Sworn Enemy, Stranger),
  info_source_one_hot[5] (Witnessed / TalkedWith / Overheard /
                          Rumor / NeverMet),
  last_known_loc_age, in_nearby_actors_slot_idx, exists                     29 × 10 = 290

known_groups[K=6]            groups the agent knows about (own + nearby +
                             at-war + allied + suzerain + vassals)
  group_kind_one_hot(8), my_membership (member/leader/none/exiled),
  group_size_log, group_strength_log,
  standing_with_me_one_hot(4) (Allied, Neutral, Tense, AtWar),
  exists                                                                     10 ×  6 =  60
```

`known_actors` is populated from a per-agent view: `union(spouse_id, mentor_id, apprentice_id, group_leader_ids of my groups, top-K-grudges, top-K-friendships)` deduped. Each slot has a "currently in nearby_actors slot N" backref so the model can correlate.

`known_groups` covers any group kind the agent has cause to know about.

#### 2.1.6 Known resources / quests (~170)

```
known_named_resources[K=6]   from agent.known_resources                     11 × 6 = 66
known_voxel_resources[K=6]   from agent.known_voxel_resources               11 × 6 = 66
active_quests[K=4]           includes group-scoped wars where applicable    10 × 4 = 40
```

#### 2.1.7 Context blocks (~80)

Settlement context (~30): treasury_log, food_status, threat_level, population_log, infrastructure_level, n_active_quests_at_settlement, charter_one_hot(8), specialty_one_hot(8), context_tags(14), recent_chronicle_event_counts_by_category. (Settlement is itself a Group; this block is the residence-group context.)

Region context (~12): terrain_one_hot(8), sub_biome_one_hot(8), elevation, monster_density (now: hostile-creature-type density), threat_level, unrest, control_score, n_dungeons, has_river_access, distance_to_coast.

World context (~20): tick_normalized, current_year, season_one_hot(4), time_of_day, current_world_age_one_hot(8), n_legendary_agents, n_settlements, recent_world_event_counts_by_kind.

#### 2.1.8 Total

```
self atomic            ~60
self contextual       ~155   (+ knowledge_domain_bits[32] + memory_fill_pct)
group memberships     ~130   (+ group_intel_velocity per slot)
spatial slots       ~1030
non-spatial slots    ~350
known resources/quests 170
context               ~80
─────────────────────────
total              ~1975 floats per agent
```

~1975 × 4 bytes × 20K agents = 158 MB observation tensor per tick. On-GPU is free; CPU pack <5ms. The extra z-axis + info features add ~320 floats net — dominated by the per-slot vec3 + info_source_one_hot expansion in `nearby_actors` and `known_actors`.

### 2.2 Action space (after vocabulary collapse)

```
action {
  // Macro head — one of 6 (or NoOp)
  head categorical macro_kind: enum {
    NoOp, PostQuest, AcceptQuest, Bid, Announce, InviteToGroup, AcceptInvite
  }

  // Micro head (when macro_kind = NoOp, this is the actual action)
  head categorical micro_kind: enum {
    Hold,
    MoveToward, Flee,                                      // movement
    Attack, Cast, UseItem,                                 // combat
    Harvest, Eat, Drink, Rest,                             // resource
    PlaceTile, PlaceVoxel, HarvestVoxel,                   // construction
    Converse, ShareStory, Communicate, Read,               // social / info
    Remember                                               // memory
  }

  // Pointer head — select from observation slots
  head pointer target: select_from
    nearby_actors ∪ nearby_resources ∪ nearby_structures
    ∪ known_actors ∪ known_groups ∪ active_quests
    ∪ memberships ∪ recent_memory_events

  // Continuous heads
  head continuous pos_delta: vec3 ∈ [-1, 1]³
  head continuous magnitude: f32 ∈ [0, 1]

  // Macro-action structured params (only relevant when macro_kind != NoOp)
  head categorical quest_type:        enum QuestType  (~16 variants)
  head categorical party_scope:       enum PartyScope  (~6 variants)
  head categorical reward_type:       enum RewardKind  (~10 variants)
  head categorical payment_type:      enum PaymentKind  (~6 variants)
  head categorical announce_audience: enum AnnounceAudience (Group / Area / Anyone)
  head pointer     fact_ref:          select_from recent_memory_events
                                                    // FactRef = local memory event_id
}
```

`macro_kind` and `micro_kind` are decoupled. When the model emits `macro_kind=PostQuest`, the `quest_type/party_scope/reward_type` heads are read; `micro_kind` is ignored (or forced to NoOp). When `macro_kind=NoOp`, the `micro_kind` and its associated heads (target, pos_delta, magnitude) drive a per-tick physical/social act.

This **two-level head** is the hierarchical decomposition the attacker called for. Macro decisions are rare (a few per NPC per ~100 ticks), micro decisions are per-tick. Training can up-weight rare macro decisions or use separate heads with different learning rates.

### 2.3 Mask — load-bearing role gate

Per-tick boolean array of valid actions, computed declaratively:

Distance predicates come in three flavors:
- `distance(a, b)` — 3D Euclidean. Default for combat, threat, direct interaction.
- `planar_distance(a, b)` — xy-only. Default for social range, audibility, formations.
- `z_separation(a, b)` — |a.z − b.z|. For altitude-gated predicates ("can't melee-attack someone 20m above without flight").

Info-gated predicates:
- `knows_event(self, event_kind, target_pattern)` — `self.memory` contains a matching event (any source).
- `knows_agent(self, agent_id)` — `self.memory` references `agent_id` OR `agent_id ∈ self.relationships`.
- `confident_about(self, fact, threshold=0.5)` — `self.memory` contains `fact` with `confidence ≥ threshold`.
- `recent(self, event_kind, max_age_ticks)` — `self.memory` contains matching event with `(now − tick) ≤ max_age_ticks`.

```
mask {
  // Micro
  Attack(t)         when t.alive ∧ is_hostile(self, t)
                         ∧ distance(self, t) < AGGRO_RANGE
                         ∧ (z_separation(self, t) < MELEE_Z_TOLERANCE ∨ self.can_fly)
                    // is_hostile = relationship.valence < HOSTILE_THRESH
                    //              ∨ groups_at_war(self, t)
                    //              ∨ predator_prey(self.creature_type, t.creature_type)
  Eat               when self.inventory.food > MEAL_COST
  Harvest(r)        when r.remaining > 0 ∧ distance(self, r) < HARVEST_RANGE
                    // r is a derived view from voxel materials, not a Resource entity
  PlaceTile(p, _)   when distance(self, p) < REACH ∧ tile_at(p).tile_type == Empty
  Converse(t)       when t.creature_type.can_speak
                         ∧ planar_distance(self, t) < SOCIAL_RANGE
                         ∧ z_separation(self, t) < 2.0
  Communicate(r, fact_ref)
                    when r.can_hear ∧ self.can_speak
                         ∧ planar_distance(self, r) < CONVERSE_RANGE
                         ∧ z_separation(self, r) < 2.0
                         ∧ fact_ref ∈ self.memory
  Read(doc)         when doc ∈ self.inventory.documents
                         ∨ (distance(self, doc.pos) < REACH ∧ doc.tile_type == Document)

  // Macro — gated by group membership/role and info state
  PostQuest{type=Conquest, party=Group(g), ...}    when g ∈ self.leader_groups
                                                        ∧ g.kind ∈ {Faction, Pack}
                                                        ∧ g.military_strength > 0
  PostQuest{type=Marriage, party=Individual(_), target=Agent(t), ...}
                                                  when self.creature_type.can_marry
                                                        ∧ ¬married(self)
                                                        ∧ ¬married(t)
                                                        ∧ relationship(self, t).valence > MIN_TRUST
                                                        // planar_distance NOT required —
                                                        // political marriages by message
  PostQuest{type=Found, party=Group(g), ...}       when g ∈ self.leader_groups
                                                        ∧ population(home_settlement(g)) > THRESH
  PostBounty(t)           when knows_agent(self, t)
                                ∧ confident_about(self, Fact::CommittedCrime(t), 0.5)
  Announce{audience=Group(g), fact_ref=f}
                          when g ∈ self.memberships
                                ∧ f ∈ self.memory
                                ∧ self.can_speak
  Announce{audience=Area(c, r), fact_ref=f}
                          when f ∈ self.memory
                                ∧ self.can_speak
                                ∧ r ≤ MAX_ANNOUNCE_RADIUS
  Bid(auction_id, ...)    when auction.visibility ⊇ self
                                ∧ self.gold ≥ auction.min_bid
                                ∧ auction.deadline > now
  AcceptQuest(qid)        when self ∈ quest.eligible_acceptors
                                ∧ ¬conflicts_with_my_active_quests(qid)
  JoinGroup(g)            when g.recruitment_open
                                ∧ self.eligible_for(g)
                                // covers: enlisting in faction, joining religion,
                                //         apprenticeship, party membership, etc.
  LeaveGroup(g)           when g ∈ self.memberships
                                ∧ ¬self.binding_oaths_with(g)
                                // covers: defection (with reputation cascade),
                                //         excommunication petition, divorce, etc.

  // Theory-of-mind gated
  Deceive(t, fact)        when ¬believed_knowledge(t, fact.domain)
                                ∧ planar_distance(self, t) < SOCIAL_RANGE
  ...
}
```

Mask predicates compile to:
- A per-tick boolean tensor `[N × NUM_ACTIONS]` where `NUM_ACTIONS` = product over heads of valid combinations (NOT a flat enumeration; that would be combinatorial)
- For sampling efficiency, the mask is per-head: `categorical_mask[N × NUM_KINDS]`, `target_mask[N × NUM_SLOTS]`, etc. The model samples each head independently respecting its mask.

The role gating happens entirely here. Removing or adding a role's powers = changing mask predicates, no model retraining needed (the model just sees more/fewer logits passing the mask).

### 2.4 Backend (singular)

```rust
trait PolicyBackend {
    fn evaluate_batch(
        &self,
        observations: &PackedObservationBatch,  // [N, OBS_DIM]
        masks:        &PackedMaskBatch,         // per-head masks
    ) -> ActionBatch;                           // [N] of typed Action
}

// Production:
impl PolicyBackend for NeuralBackend {
    fn evaluate_batch(...) -> ActionBatch {
        let logits = self.model.forward(observations);
        let actions = sample_with_mask(logits, masks, self.temperature);
        actions
    }
}

// Training-time only:
impl PolicyBackend for UtilityBackend {
    // declarative scoring rules from DSL; argmax over masked candidates
}

// Research-time only (off the per-tick path):
impl PolicyBackend for LlmBackend {
    // serialize obs to JSON, send to LFM, parse action; usable for prototyping
    // novel behaviors before training a model on them
}
```

One `evaluate_batch` call per tick. All N alive NPCs go through one forward pass. No per-role dispatch at runtime; the role differences live in the observation features and mask, both of which are already per-NPC.

[OPEN] When LlmBackend is used for prototyping, it's not in the per-tick loop — it runs slower-than-real-time on a subset of NPCs to generate trajectories that get distilled into NeuralBackend training data.

### 2.5 Reward (for ML training)

Declarative per-policy:

```
reward {
  delta(self.needs.satisfaction_avg)         × 0.1   // need recovery
  delta(self.hp_frac)                        × 5     // damage hurts, healing rewards
  +1.0  on event(EntityDied{killer=self ∧ target.team ≠ self.team})
  -1.0  on event(EntityDied{target ∈ self.close_friends})
  +0.05 per behavior_tag accumulated this tick
  +2.0  on event(QuestCompleted{quest.party_member_ids ∋ self})
  -1.0  on event(QuestExpired{quest.party_member_ids ∋ self})
  +0.02 × delta(self.fame_log)
  -0.5  on event(self.role transition: Leader → Outlaw)
  ...
}
```

Compiler emits:
- A per-tick reward kernel that diffs pre/post-tick observations + scans events involving self
- Logging hooks for (observation, action, reward) tuples → training dataset emission
- Validation: reward components must reference declared views/events

[OPEN] Reward shaping for rare macro actions (Conquest reward arrives 2000+ ticks after the PostQuest decision). Need temporal credit assignment — GAE/lambda returns or hierarchical RL.

---

## 3. DSL surface for declaring observation

```
policy NpcDecision {
  observation {
    // Atom: a single normalized scalar
    self.hp_pct = self.hp / self.max_hp
    self.mood  = view::mood(self)

    // Block: named reusable group
    block self.psychological {
      from self.needs       as vec(6)
      from self.emotions    as vec(6)
      from self.personality as vec(5)
      from view::mood(self) as f32
    }

    // Slot array — spatial
    slots nearby_actors[K=12] from query::nearby_agents(self, radius=50)
                              sort_by distance(self, _) {
      atom relative_pos = (other.pos - self.pos) / 50
      block other_features = view::agent_features_for_observation(other, viewer=self)
      atom relationship_valence  = view::relationship(self, other).valence
      atom n_shared_groups_log   = log1p(view::shared_group_count(self, other))
      atom is_in_known_actors    = exists_in(known_actors, other.id)
    }

    // Slot array — non-spatial
    slots known_actors[K=10] from query::known_actors(self)
                             sort_by relevance(self, _) {
      atom relative_pos_or_absent = if other.pos_known { (other.pos - self.pos)/100 } else { ABSENT }
      atom relationship_kind = one_hot(view::relationship_kind(self, other), 8)
      atom relationship_valence = view::relationship(self, other).valence
      atom last_known_age    = log1p(now - other.last_seen_tick) / 10
    }

    slots known_groups[K=6] from query::known_groups(self)
                            sort_by relevance(self, _) { ... }

    // Slot array from event ring
    slots recent_events[K=12] from latest(self.memory_events) {
      atom type     = one_hot(e.event_type, EVENT_TYPE_VOCAB)
      atom age      = log1p(now - e.tick) / 10
      atom target_in_nearby = index_in_slots(e.target_id, nearby_actors)
      atom valence  = e.valence
    }

    // Bitset → fixed one-hot
    bitset settlement_culture[14] = self.home_settlement.context_tags

    // Summary fold over recent events
    summary recent_chronicle[NUM_CHRONICLE_CATEGORIES] {
      from world.chronicle filter |e| now - e.tick < 200
      group_by e.category
      output count_log
    }
  }

  action { ... }   // see §2.2
  mask   { ... }   // see §2.3
  reward { ... }   // see §2.5
  backend Neural { weights: "npc_v3.bin", h_dim: 256 }
}
```

Compiler emits:
- Packed struct with named field offsets (debug tools decode bytes back to fields)
- Packing kernel — per-NPC, gather all referenced fields/views, write to packed buffer
- JSON schema dump for ML pipeline tooling
- Schema hash + versioning metadata
- Validation: every observation field must reference declared `entity` field, `view`, or `event` source

---

## 4. Schema versioning + migration

The attacker's strongest point. Once the model trains on observation schema v1.0, adding a new feature breaks the model's input layout.

**Approach:**

1. **Schema hash** — DSL compiler emits a SHA256 over the (sorted, normalized) observation field schema. Every model checkpoint stores its training-time hash. Loading a model whose hash mismatches the current DSL is a hard error unless explicitly opted-in via migration.

2. **Migration tables** — DSL supports `@since(v=1.1)` annotations on observation fields. Compiler can emit padded-zero observations when running an old model against a new schema (the new fields read as zero), or rejected-with-explanation when the model would need fields that no longer exist.

3. **Append-only schema growth** — when adding features, append; never reorder existing slots. Removing a feature is a breaking change requiring model retraining.

4. **CI guard** — observation schema changes that touch existing slots fail CI unless accompanied by a model checkpoint bump.

[OPEN] Whether schema hash should also cover the action vocabulary. Adding a new ActionKind is also breaking (model's categorical head changes shape).

---

## 5. GPU compilation — scoped (not "compiles to CUDA")

The attacker is right that "compiles to CUDA" is aspirational. Concrete GPU-targeted kernels:

**GPU-amenable (compile to wgpu/CUDA):**
- Observation packing kernel — per-NPC gather of fields + views into packed buffer. Embarrassingly parallel.
- Per-head mask evaluation — boolean tensor computation per NPC.
- Neural forward pass — already what Burn does.
- Event-fold view materialization — reductions over event rings (sum damage events for hp, etc.).

**GPU-hostile (stay CPU):**
- Mask predicates with cross-entity references (e.g. `t ∈ quest.eligible_acceptors` requires walking quest membership). Compute on CPU into the boolean tensor; that boolean tensor is what GPU consumes.
- LLM backend (CPU + remote inference always).
- Chronicle text generation (CPU).
- Quest cascade rules involving heterogeneous events (CPU).
- World event emission (rare, cheap, CPU).

**Hybrid:**
- Spatial queries (nearby_actors, threats) — 3D-chunked spatial hash (keyed on voxel-chunk coord) is the default; CPU uniform-3D-grid fallback for small N. Observation packing writes per-slot `relative_pos: vec3` and `z_separation_log`; layout is contiguous, trivially GPU-friendly.

The DSL compiler outputs different backend code per kernel based on annotations. `@gpu_kernel` for the packers; default CPU for the rule cascades.

---

## 6. Open questions

The previous v1 listed 10 open questions. The debate + iteration resolved or reframed several:

| # | Question | Resolution |
|---|---|---|
| 1 | K=12 vs K=20 nearby? | K=12 default, K=16 for combat-role NPCs (per-role observation depth). [OPEN] |
| 2 | Slot ordering sorted/unsorted? | Sorted by distance (stable input for utility/heuristic + pos invariance for neural) |
| 3 | Eager vs lazy observation? | Eager batch (one pack kernel per tick) — lazy adds complexity for marginal saving |
| 4 | Per-policy vs per-entity backend? | **Single neural backend** — question dissolved |
| 5 | Rare actions handling? | **Macro/micro head decomposition** — see §2.2 |
| 6 | Conditional vs always-emit heads? | Always emit; mask zeroes irrelevant heads. Simpler training. |
| 7 | Observation versioning? | Schema hash + migration table, see §4 |
| 8 | Reward placement? | Per-policy DSL block, see §2.5 |
| 9 | Multi-NPC collective policies? | **Reduced to per-NPC via PostQuest/AcceptQuest** — question dissolved |
| 10 | Curriculum / role progression? | One policy with role-conditioned masking + role features in observation |

New open questions from this iteration:

| # | Question | Notes |
|---|---|---|
| 11 | `known_actors[K=10]` capacity? | 10 may be too few; or too many. Needs validation against actual decision quality. |
| 12 | Macro head firing rate | Should the model emit macro actions every tick (mostly NoOp) or run macro inference at lower cadence (every 10 ticks)? Affects training + per-tick cost. |
| 13 | Quest discovery | Push-based (event-triggered re-eval for eligible NPCs) vs pull-based (always in observation). |
| 14 | Reward credit assignment for macro actions | Wars resolve 2000+ ticks after PostQuest. GAE/n-step returns vs goal-conditioned reward shaping. |
| 15 | Mask predicate compilation | Cross-entity predicates (faction membership, quest eligibility) need careful index design to avoid O(N²). |
| 16 | When to retire utility backend | After what training milestone is utility fully redundant? Or do we keep it permanently as regression baseline? |
| 17 | Schema hash scope | Should hash cover observation only, or also action vocabulary + reward? |
| 18 | LLM backend integration with training pipeline | LLM trajectories → distilled into Neural model? Or LLM as evaluator/critic? |

---

## 7. Risks (acknowledged)

- **Observation as ML training contract.** Once frozen, schema is hard to change. Mitigation: schema hash + append-only growth + CI guard. **Accept as cost of doing business.**
- **Slot K choices may be wrong.** No way to know without empirical validation. Mitigation: support per-role K (combat NPCs see more enemies; merchants see more resources). Plan A/B testing once a baseline model exists.
- **Macro-action training is hard.** Rare events are a known RL challenge. Mitigation: hierarchical heads, GAE returns, prioritized replay buffer with rare-event up-weighting, possibly imitation pretraining from utility-backend trajectories.
- **DSL complexity vs hand-written Rust.** If we end up with one or two policy declarations, hand-rolling is simpler. Mitigation: build minimal DSL; only invest in primitives that pay off across multiple declarations (observation schema, mask language, action heads). Skip primitives that don't.
- **GPU compilation underdelivers.** "Compiles to CUDA" is hard. Mitigation: scope to specific kernels (observation pack, neural forward, event folds), keep cascade logic on CPU.
- **Auction/quest unification adds runtime complexity.** The unified design depends on a real auction system + extended quest type. Mitigation: build minimum viable versions (`HighestBid` resolution only, per-AuctionKind cadence). Defer fancy features.

---

## 8. What this version commits to

1. **Single neural backend** for production. Utility for bootstrap; LLM for research. No mixed dispatch.
2. **Observation is rich** (~1600 floats) and includes spatial slots + non-spatial named-reference slots + context blocks. Schema versioned.
3. **Action vocabulary is collapsed** to ~15 verbs via PostQuest/AcceptQuest/Bid + micro primitives, per `proposal_universal_mechanics.md`.
4. **Action space is hierarchical** (macro/micro heads) for training stability.
5. **Mask is load-bearing** for role differentiation and structural constraints. Compile predicates declaratively.
6. **GPU compilation is scoped** to specific kernels, not the whole policy.
7. **Reward is declarative** per policy, with credit-assignment open.

Next steps: synthesis pass over all 17 docs, then DSL surface design (#23). Then a prototype: implement minimum viable observation packing + utility backend + one trained neural backend on a small NPC count, measure decision quality vs current `action_eval`. Iterate from there.
