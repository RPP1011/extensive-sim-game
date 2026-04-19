# Proposal: ECS DSL Surface

Formal surface-language specification for the world-sim ECS DSL. Covers grammar, type system, runtime semantics, and compilation targets. Consumes the settled foundations in `state_npc.md`, `state_aggregate.md`, `state_world.md`, and the active proposals in `proposal_universal_mechanics.md` and `proposal_policy_schema.md`. Extends them with the authoring, observation, training, runtime, modding, and player-behavior surfaces surfaced in the story investigations (`stories_A_B.md` through `stories_H.md`).

---

## 1. Language overview

The DSL declares a closed-world simulation: typed state, typed events, declarative derivations, cascade rules, action masks, policies, and trainer plumbing. Eleven top-level declaration kinds compose into a single compiled artefact (Rust runtime + SPIR-V kernels + JSON schema + packed checkpoints).

- **`entity`** — parameterization of one of the three predefined root kinds (Agent, Item, Group). Authors cannot introduce new root kinds; they declare `creature_type`, `ItemKind`, or `GroupKind` variants with default stats, capabilities, eligibility, and starting memberships. (`stories_A_B.md` §1, §10, §11)
- **`event`** — typed, append-only records. The universal state-mutation channel. Annotations mark replayability, high-volume classification, observation visibility. (`stories_A_B.md` §2; `stories_C.md` §15; `stories_F_G_I_J.md` §60)
- **`view`** — pure or event-folded derivations. Eager (`@materialized`) or lazy; first-class spatial and non-spatial queries. (`stories_A_B.md` §3, §5; `stories_E.md` §26)
- **`physics`** cascade rule — phase-tagged transforms from events to events with compile-time cycle detection, race detection, and schema-drift guards. (`stories_A_B.md` §4; `stories_E.md` §27)
- **`mask`** — per-action predicates forming the role gate and the legality contract. Compiles to per-head boolean tensors. (`stories_A_B.md` §8; `proposal_policy_schema.md` §2.3)
- **`verb`** — composition sugar that bundles mask + cascade + reward into a named gameplay action without extending the categorical action vocabulary. (`stories_A_B.md` §7)
- **`policy`** — observation block, action heads, reward, value, advantage, training, backend, curriculum, and telemetry sub-blocks. One forward pass per tick for all alive agents. (`proposal_policy_schema.md` §2; `stories_D.md` §20, §22, §24)
- **`invariant`** — static, runtime, or debug-only predicates over state, checked at the phase boundary they describe. (`stories_A_B.md` §6)
- **`probe`** — named scenario + behavioral assertion, evaluated on a checkpoint against seeded trajectories. CI regression surface. (`stories_C.md` §18)
- **`curriculum`** — staged training with mask overrides, reward overrides, and transition criteria. (`stories_D.md` §22)
- **`telemetry`** — per-metric emission and alert declarations driving both dashboards and curriculum gating. (`stories_D.md` §24)

At runtime the tick pipeline is fixed: pre-phase rules → event emission → materialized-view updates → policy inference (observation pack + mask + forward + sample) → action application → cascade fixed-point → post-phase rules → telemetry emission. Determinism flows through a single per-world RNG; text generation sits outside the deterministic fold. (`stories_E.md` §27, §29; `stories_F_G_I_J.md` §60)

---

## 2. Top-level declarations

### 2.1 `entity` declaration

`entity` parameterizes the three predefined root kinds. There is no grammar for declaring a fourth root kind.

```
entity <Name> : Agent {
  creature_type:        CreatureType,
  capabilities:         Capabilities { ... },
  default_personality:  Personality,
  default_needs:        Needs,
  hunger_drives:        [ HungerDriveKind, ... ],
  default_memberships:  [ GroupSpec, ... ],
  predator_prey:        { prey_of: [CreatureType], preys_on: [CreatureType] },
}

entity <Name> : Item {
  kind:           ItemKind,
  rarity:         Rarity,
  base_stats:     { ... },           // fixed-shape struct per ItemKind
  slots:          [EquipSlot, ...],
}

entity <Name> : Group {
  kind:                    GroupKind,
  eligibility_predicate:   <predicate>,    // compiles to EligibilityRef
  recruitment_open:        bool,
  governance:              GovernanceKind,  // Hereditary | Elective | Council | Theocratic | AlphaContest | AdHoc
  default_standings:       [ (GroupKind, Standing), ... ],
  required_fields:         [ FieldName, ... ],
}
```

Layout hints:

- `@materialized` — the compiler generates an update path (see §2.3).
- `@spatial` — the entity carries a `pos` field participating in spatial indices (KD-tree, uniform grid, voxel chunk grid).
- `@capped(K)` — on collection fields; bounds the fixed storage layout (§5.1 bitset / sorted-Vec / ring buffer / Map types).
- `@inline` — embed the child struct in the parent's SoA row rather than reference it (cache-line tuning).

The compiler emits:

- A Rust struct populated with defaulted fields per declaration.
- An enum-variant extension for the corresponding discriminator (`CreatureType`, `ItemKind`, or `GroupKind`).
- A spawn-template that initialises all required fields.
- A schema-hash contribution for the one-hot width of that discriminator (§4).

### 2.2 `event` declaration

Typed records. Every state mutation reaches primary state through an event.

```
event <Name> {
  <field>: <type>,
  ...
}

@replayable                  // default; event replays deterministically
@non_replayable              // text-gen events, LLM prose, side-channel
@high_volume                 // routes to a separate ring-buffer storage class
@observation_visible         // participates in observation slot arrays
@gpu_amenable                // scalar fields only; triggers GPU event-fold codegen
```

`MemoryEvent` (declared in `state_npc.md`) carries a `source: Source` field alongside the usual payload:

```
enum Source {
    Witnessed,              // confidence = 1.0
    TalkedWith(AgentId),    // confidence = 0.8
    Overheard(AgentId),     // confidence = 0.6
    Rumor { hops: u8 },     // confidence = 0.8^hops
    Announced(GroupId),     // confidence = 0.8
    Testimony(ItemId),      // confidence = item.trust_score
}
```

`Source` is a standard type used both inside `MemoryEvent` payloads and projected into observations as `info_source_one_hot[5]` (see §3.1).

The compiler emits:

- A variant on the runtime event enum for that kind-bucket (`WorldEvent` / `MemoryEvent` / `StructuralEvent` — the bucket is chosen by the compiler from the event's payload references, not by the author).
- A fixed-capacity ring buffer, sized by `@high_volume` (larger) or the default.
- Serialisation hooks for replay (if `@replayable`).
- An entry in the observation event-vocabulary table (if `@observation_visible`).
- A pattern-match kernel for use in cascade, reward, and probe blocks.

String payloads are permitted only on `@non_replayable` events; the compiler rejects `String` fields on replayable events (`stories_F_G_I_J.md` §60). All load-bearing references are `AgentId`, `GroupId`, `ItemId`, `QuestId`, or `AuctionId` — no text. This forces `class_tags: Vec<String>` and `archetype: String` on legacy agent fields to migrate to `TagId` / `ArchetypeId` (`stories_F_G_I_J.md` §60 cross-cutting).

### 2.3 `view` declaration

Views are pure over their inputs. The compiler chooses between lazy evaluation at observation-pack time and eager event-fold materialization per declaration.

```
view <name>(<args>) -> <type> {
  <expression>
}

@materialized(on_event = [EventName, ...])
view mood(a: Agent) -> f32 {
  initial: 0.0,
  on DamageReceived{target: a} { self -= 0.1 * e.damage / a.max_hp }
  on FriendDied{observer: a}   { self -= 0.3 }
  on NeedsSatisfied{agent: a}  { self += 0.05 * e.delta }
  clamp: [-1.0, 1.0],
}

@lazy                           // default; pure function evaluated at read
view is_hostile(a: Agent, b: Agent) -> bool {
  relationship(a, b).valence < HOSTILE_THRESH
  || groups_at_war(a, b)
  || predator_prey(a.creature_type, b.creature_type)
}

@spatial(radius, kind = [Agent|Resource|Structure])
query nearby_agents(self: Agent, radius: f32) -> [Agent]
sort_by distance(self, _) limit K { ... }

@indexed_on(field)             // forces a sorted or hashed index on `field`
@top_k(K)                      // bounded output size
@backend(cpu | gpu)            // override auto-selection
@fidelity(>= Medium)           // skip evaluation at lower fidelity (§2.3 of this doc; stories_E.md §31)
```

The compiler emits:

- For `@materialized`: a field on the corresponding entity + an event-dispatch table mapping each `on_event` to the update body. GPU-amenable materializations sort events by target before commutative reduction to preserve determinism (`stories_E.md` §29 GPU-determinism traps).
- For `@lazy`: an inline function referenced from observation packing and mask predicates.
- For `@spatial`: routing to the appropriate spatial index — `voxel_engine::ai::spatial` on GPU, or a CPU uniform-grid fallback.
- For `@top_k`: a fixed-cap partial-sort that writes into a `SimScratch` buffer.

Queries are first-class and carry their output cap. `query::nearby_agents(self, r)` returns at most K records; `query::known_actors(self)` returns the union `{spouse, mentor, apprentice, group_leaders(self), top_grudges, top_friendships}` deduped (`proposal_policy_schema.md` §2.1.5).

### 2.4 `physics` cascade rule

Cascade rules respond to events and emit further events. They never mutate state directly — all writes go through `emit`. Phase-tagged ordering prevents races.

```
physics <name> @phase(pre | event | post) {
  on <EventPattern> where <predicate> {
    emit <EventName> { <field>: <expr>, ... }
    ...
  }
}

physics collapse_chain @phase(event) {
  on VoxelDamaged{pos: p, new_integrity: i} where i <= 0.0 {
    emit FragmentCollapse { pos: p, cause: NpcHarvest }
    for q in voxel_neighbors_above(p) {
      emit VoxelDamaged { pos: q, amount: COLLAPSE_DAMAGE }
    }
  }
}

physics auction_item_resolved @phase(event) {
  on AuctionResolved{kind: Item, winner: W, payment: P, auction_id: id} {
    let a = auction(id);
    emit TransferGold { from: W, to: a.seller, amount: P.gold_amount() }
    emit TransferItem { from: a.seller, to: W, item: a.item.item_id() }
  }
}
```

Annotations:

- `@phase(pre | event | post)` — fixed three-phase ordering (`stories_E.md` §27). `pre` runs before event emission (read-only state access). `event` runs during the fixed-point event fold. `post` runs after all events quiesce for the tick.
- `@before(OtherRule)` / `@after(OtherRule)` — explicit ordering between rules in the same phase.
- `@terminating_in(N)` — asserts the cascade converges within N hops when self-emission is possible (`stories_A_B.md` §4).

Compile-time validation:

1. Event-type existence on every `on` pattern.
2. Pattern-field mismatch against the event's declared fields.
3. Field-write race: two unordered rules in the same phase both emitting events that materialize into the same entity field fail unless one uses a commutative update (`+=`, `min=`, `max=`).
4. Cycle detection in the event-type emission DAG; self-loops require `@terminating_in`.
5. Schema-hash drift on event or field references.
6. All referenced views, events, and fields resolve.

### 2.5 `mask` predicates

Per-action validity predicates. Cross-entity references are allowed; the compiler routes them through CPU when they require entity-table walks.

```
mask <ActionHead>(<bindings>) when <predicate>

mask Attack(t) when t.alive
                ∧ is_hostile(self, t)
                ∧ distance(self, t) < AGGRO_RANGE

mask PostQuest{type: Conquest, party: Group(g), target: Group(t)}
  when g in self.leader_groups
     ∧ g.kind in {Faction, Pack}
     ∧ g.military_strength > 0
     ∧ g.standings[t] != AtWar
     ∧ eligible_conquest_target(self, t)

mask Bid(auction_id, payment, _) when auction(auction_id).visibility contains self
                                    ∧ self.gold >= auction(auction_id).reserve_price.gold_amount()
                                    ∧ auction(auction_id).deadline > now

mask InviteToGroup{kind: Family, target: Agent(t)}
  when self.creature_type.can_marry
     ∧ ¬married(self)
     ∧ ¬married(t)
     ∧ relationship(self, t).valence > MIN_TRUST
     ∧ ¬self.in_active_invite_to(t, Family)
```

Supported operators: set membership (`contains`, `in`), quantifiers (`forall`, `exists`), bounded folds (`count`, `sum`, `max`, `min`), arithmetic comparison, and view calls. No user-defined functions inside masks (they cannot be safely compiled to GPU boolean kernels).

Compilation:

- Per-head boolean tensors `categorical_mask[N × NUM_KINDS]`, `target_mask[N × NUM_SLOTS]`, etc.
- Mask predicates that reference only intrinsic scalar fields compile to SPIR-V compute shaders; cross-entity predicates (`t in quest.eligible_acceptors`, `at_war(self.faction, f)`) are CPU-patched into the same boolean buffer before sampling.
- Every predicate node has a stable AST ID; an explanation kernel reruns the predicate against a captured observation snapshot for `trace_mask(agent, action, tick)` (`stories_F_G_I_J.md` §34).

### 2.6 `verb` (composition sugar)

`verb` declares a named gameplay action that composes an existing micro primitive with additional mask predicates, cascades, and reward hooks. It does NOT add to the closed categorical action vocabulary. (`stories_A_B.md` §7)

```
verb Pray(self, shrine: Structure) =
  action Converse(target: shrine.patron_agent_id)
  when  self.memberships contains Group{kind: Religion}
      ∧ distance(self, shrine) < REACH
  emit  PrayCompleted { prayer: self, shrine: shrine, faith_delta: 1.0 }
  reward +0.5
```

The compiler expands a `verb` into:

1. A mask entry narrowing an existing `Converse` / `Attack` / other micro primitive.
2. A cascade handler that emits the declared event on successful action application.
3. A reward hook added to the policy's reward block.

Adding a `verb` does not bump the schema hash. Adding a new micro primitive does (see §4).

### 2.7 `policy` declaration

```
policy <Name> {
  observation { ... }        // §3.1
  action { ... }             // §3.2
  mask { ... }               // §2.5
  reward { ... }             // §3.4
  value { ... }              // §3.4 — actor-critic head
  advantage { ... }          // §3.4
  training { ... }           // §3.4
  curriculum { ... }         // §2.10
  telemetry { ... }          // §2.11
  backend <Kind> { ... }     // §3.5
}
```

Exactly one `policy` block covers all agents. Role differentiation is entirely through mask and observation features — not per-role backends (`proposal_policy_schema.md` §1). (§2.4 remains the singular `PolicyBackend::evaluate_batch` call point.)

### 2.8 `invariant` declaration

```
invariant <name>(<scope>) @<mode> { <predicate> }

@static         // enforced at compile time via type system / dataflow analysis
@runtime        // checked after every cascade phase that writes the predicate's support
@debug_only     // runtime check, panic only in debug builds

invariant no_bigamy(a: Agent) @runtime {
  count(g in a.memberships where g.kind == Family && g.is_marriage) <= 1
}

invariant append_only_observation() @static {
  forall (old_offset, new_offset) in schema_diff:
    new_offset >= old_offset || is_slot_internal_append(old_offset, new_offset)
}

invariant party_member_agents_alive(q: Quest) @debug_only {
  forall m in q.party_member_ids: entity(m).alive
}
```

Cascades that write a field in the invariant's support must annotate `@must_preserve(<invariant>)`; the compiler rejects cascades that do not declare it.

Schema invariants are separate from state invariants: the append-only-observation invariant is checked by the compiler at CI time, not at runtime.

### 2.9 `probe` declaration

Named behavioral tests evaluated against a checkpoint's trajectories. Probes live in `probes/` alongside their seed scenarios. (`stories_C.md` §18)

```
probe <Name> {
  scenario   "probes/<name>.toml"
  seed       <u64>
  ticks      <u32>
  backend    neural:<path> | utility | llm
  tolerance  <f32>

  assert {
    <assert_expr>
  }
}

assert_expr := count_expr | prob_expr | mean_expr

count_expr  := "count" "[" filter "]" <comparator> <scalar>
prob_expr   := "pr"    "[" <action_filter> "|" <obs_filter> "]" <comparator> <prob>
mean_expr   := "mean"  "[" <scalar_expr> "|" <filter> "]" <comparator> <scalar>
```

Example:

```
probe LowHpFlees {
  scenario "probes/low_hp_1v1.toml"
  seed 42
  ticks 200
  backend neural:generated/npc_v3.bin
  tolerance 0.02

  assert {
    pr[ action.micro_kind in {Flee, MoveToward_away_from_threat}
      | self.hp_pct < 0.3 ]
    >= 0.80
  }
}

probe NoSpouseAttacks {
  scenario "probes/random_family_5agents.toml"
  assert {
    pr[ action.micro_kind == Attack | action.target.is_spouse == 1 ] == 0.0
  }
}
```

Probes compile to trajectory queries over the replay buffer format in §3.5.4. Schema-hash mismatch between a probe's reference fields and the current DSL is a hard error (fail-loud, §4).

### 2.10 `curriculum` declaration

```
curriculum {
  stage <Name> {
    inherits <PreviousStage>        // optional
    mask_override { ... }
    training_weights { ... }
    reward_override { ... }
    transition_when {
      metric <name> <comparator> <value>
      min_steps <u64>
    }
  }
}

curriculum {
  stage Foraging {
    mask_override {
      micro_kind allow [Hunt, Eat, Drink, Rest, MoveToward, Hold]
      macro_kind allow [NoOp]
    }
    training_weights { micro_kind { Hunt: 5.0, Eat: 3.0 } }
    transition_when {
      metric action_entropy(micro_kind) >= 1.2
      metric mean_episode_reward        >= 2.0
      min_steps 50_000
    }
  }
  stage Combat {
    inherits Foraging
    mask_override { micro_kind allow_additional [Attack, Cast, Flee] }
    transition_when { metric win_rate_vs_baseline >= 0.4; min_steps 100_000 }
  }
  stage Macro {
    inherits Combat
    mask_override {
      macro_kind allow_additional [PostQuest, AcceptQuest, Bid]
      quest_type allow [Hunt, Escort]
    }
  }
  stage Full {
    inherits Macro
    mask_override { macro_kind allow_all; micro_kind allow_all; quest_type allow_all }
  }
}
```

Mechanics:

- `stage_mask[i]` is composed with the runtime mask via bitwise AND: `final_mask = runtime_mask & stage_mask_i`. Runtime rules ("can't attack a non-hostile") still hold; the stage further restricts.
- `transition_when` criteria AND together. Once advanced, stages do not regress.
- `reward_override` replaces the base reward block during the stage. Critic weights are re-warmed after each transition by freezing the policy for M steps.
- Stage pointer lives in the checkpoint metadata alongside weights and schema hash (§3.5.5).

### 2.11 `telemetry` declaration

```
telemetry {
  metric <name> = <expr>
    [ window <ticks> ]
    [ emit_every <ticks> ]
    [ conditioned_on <expr> ]
    [ alert when <value_comparator> ]
}

telemetry {
  metric entropy_macro_kind = entropy_of(action.macro_kind)
    window 1000 emit_every 100 alert when value < 0.3

  metric entropy_quest_type = entropy_of(action.quest_type)
    conditioned_on action.macro_kind == PostQuest
    window 10000 alert when value < 0.8

  metric freq_micro = histogram(action.micro_kind)
    window 1000 alert when max_bin > 0.85

  metric value_error = mse(value.v_pred, montecarlo_return)
    window 1000 alert when value > 10.0
}
```

Alerts emit structured log records and participate in curriculum `transition_when` clauses. Alert suppression for N ticks after a stage transition prevents spurious entropy-low alerts when the mask widens.

---

## 3. Policy / observation / action grammar

### 3.1 Observation declaration

```
observation {
  // Atom — single normalized scalar
  self.hp_pct = self.hp / self.max_hp
  self.mood   = view::mood(self)

  // Information self-summary (see proposal_universal_mechanics.md)
  bitset self.knowledge_domain_bits[32] = view::knowledge_domains(self)
    // Combat / Trade / Family / Politics / Religion / Craft / ... —
    // bit set if any memory event of that category exists
  self.memory_fill_pct = self.memory.events.len() / self.memory.events.cap()

  // Block — named reusable group
  block self.psychological {
    from self.needs       as vec(6)
    from self.emotions    as vec(6)
    from self.personality as vec(5)
    from view::mood(self) as f32
  }

  // Spatial slot array
  slots nearby_actors[K=12] from query::nearby_agents(self, radius=50)
                            sort_by distance(self, _) {
    atom relative_pos: vec3   = (other.pos - self.pos) / 50
    atom z_separation_log     = log1p(abs(other.pos.z - self.pos.z))
    block other_features = view::agent_features_for_observation(other, viewer=self)
    atom relationship_valence  = view::relationship(self, other).valence
    atom n_shared_groups_log   = log1p(view::shared_group_count(self, other))
    atom is_in_known_actors    = exists_in(known_actors, other.id)
    atom is_trespasser         = view::is_trespasser(self, other)
    atom other_reputation_visible = view::reputation_visible(self, other)
    atom info_source_one_hot: [f32; 5] = one_hot(view::last_memory_source(self, other), 5)
                                          // Witnessed / TalkedWith / Overheard /
                                          // Rumor / NeverMet
  }

  // Non-spatial slot array
  slots known_actors[K=10] from query::known_actors(self)
                           sort_by relevance(self, _) { ... }

  slots known_groups[K=6] from query::known_groups(self)
                          sort_by relevance(self, _) {
    atom group_kind_one_hot       = one_hot(other.kind, 9)
    atom my_membership_one_hot    = one_hot(view::my_role(self, other), 4)
    atom group_size_log           = log1p(other.members_count)
    atom group_strength_log       = log1p(other.military_strength.unwrap_or(0.0))
    atom standing_with_me_one_hot = one_hot(view::standing(self.primary_group, other), 5)
    atom military_strength_ratio  = log1p(other.military_strength) - log1p(self.primary_group.military_strength)
    atom is_adjacent_territory    = view::is_adjacent(self.home_region, other.territory)
    atom controls_scarce_resource = view::controls_scarce_of(self.home_settlement, other)
    atom is_at_war_with_my_enemies = view::shared_enemies_count(self.primary_group, other) > 0
  }

  slots memberships[K=8] from self.memberships
                         sort_by (rank_in_group * group_activity) {
    atom group_kind_one_hot        = one_hot(m.group.kind, 9)
    atom my_role_in_group_one_hot  = one_hot(m.role, 6)
    atom group_leader_vacant       = m.group.leader_id.is_none()
    atom my_tenure_log             = log1p(now - m.joined_tick)
    atom my_standing_in_group      = m.standing
    atom group_activity_log        = log1p(m.group.recent_activity.len())
    atom group_intel_velocity      = view::group_intel_velocity(m.group)
                                     // EWMA of Announce/Communicate events
                                     // propagating through this group —
                                     // derived observation metric
    atom is_active_party           = m.group.active_quests.len() > 0
    atom my_tenure_relative_to_other_seniors = view::relative_tenure(self, m.group)
  }

  // Invite inbox — story 63 extension
  slots incoming_invites[K=4] from view::pending_invites(self) {
    atom invite_id          = i.invite_id as u32
    atom kind_one_hot       = one_hot(i.kind, 9)
    atom inviter_id         = i.inviter
    atom time_remaining_log = log1p(i.expires_tick - now)
    atom inviter_relationship_summary = view::relationship(self, i.inviter).valence
  }

  // Active quests / auctions
  slots active_quests[K=4]   from view::active_quests(self) { ... }
  slots active_auctions[K=4] from view::observable_auctions(self) {
    atom auction_id   = a.id as u32
    atom kind_one_hot = one_hot(a.kind, 7)
    atom seller_id    = a.seller.agent_or_group_id()
    atom best_bid_log = log1p(a.best_bid_value())
    atom deadline_log = log1p(a.deadline_tick - now)
  }

  // Known resources, memberships summaries, context blocks — same pattern
  summary recent_chronicle[NUM_CHRONICLE_CATEGORIES] {
    from world.chronicle filter |e| now - e.tick < 200
    group_by e.category
    output count_log
  }

  bitset settlement_culture[14] = self.home_settlement.context_tags

  // Atom for reputation
  self.reputation_log = log1p(view::reputation(self))
  self.war_exhaustion = view::war_exhaustion(self.primary_group)
  self.n_conflicting_group_pairs = view::group_standing_conflicts(self)
}
```

Every field has a declared `norm` kind: `identity`, `log1p`, `scale(d)`, `clamp(lo, hi)`, `one_hot(K)`, `bitset(N)`. The normalization taxonomy is fixed; new kinds require DSL changes (`stories_C.md` §13).

The compiler emits a packed struct with named field offsets, per-block contiguous regions, and a JSON schema dump consumed by ML tooling (§4). Slot-internal appends (a new field inside `nearby_actors[]`) grow every slot by `new_bytes`; total footprint grows by `K × new_bytes`.

### 3.2 Action heads

Closed action vocabulary. Adding to this vocabulary bumps the schema hash; adding a `verb` does not.

```
action {
  head categorical macro_kind: enum {
    NoOp, PostQuest, AcceptQuest, Bid, Announce, InviteToGroup, AcceptInvite,
    WithdrawQuest,                  // taker-only: retract acceptance (§9 #9)
    SetStanding                     // unify DeclareWar / MarkHostile / MarkFriendly (§9 #19)
  }

  head categorical micro_kind: enum {
    Hold,
    MoveToward, Flee,
    Attack, Cast, UseItem,
    Harvest, Eat, Drink, Rest,
    PlaceTile, PlaceVoxel, HarvestVoxel,
    Converse, ShareStory,
    Communicate, Read,
    Remember
  }

  head pointer target: select_from
    nearby_actors ∪ nearby_resources ∪ nearby_structures
    ∪ known_actors ∪ known_groups ∪ active_quests
    ∪ active_auctions ∪ incoming_invites ∪ memberships
    ∪ recent_memory_events

  head pointer fact_ref: select_from recent_memory_events
                         // FactRef = local memory event_id; used by
                         // Communicate and Announce

  head continuous pos_delta: vec3 ∈ [-1, 1]³
  head continuous magnitude: f32  ∈ [0, 1]

  // Structured parameter heads for macros
  head categorical quest_type:        enum QuestType
  head categorical party_scope:       enum PartyScope
  head categorical quest_target:      enum QuestTarget
  head categorical reward_kind:       enum RewardKind
  head categorical payment_kind:      enum PaymentKind
  head categorical group_kind:        enum GroupKind            // for InviteToGroup
  head categorical announce_audience: enum AnnounceAudience     // Group / Area / Anyone
  head categorical standing_kind:     enum StandingKind         // for SetStanding (Hostile | Neutral | Friendly | Vassal | Suzerain)
  head categorical resolution:        enum Resolution           // auction resolution
                                                                // HighestBid | FirstAcceptable | MutualAgreement
                                                                // | Coalition{min_parties: u8} | Majority (§9 #1)
}
```

`macro_kind` and `micro_kind` are decoupled. When the model emits `macro_kind != NoOp`, the macro parameter heads drive the structured action and `micro_kind` is ignored. When `macro_kind == NoOp`, the `micro_kind` + its heads drive a per-tick physical/social primitive.

### 3.3 Action vocabulary (consolidated)

Four macro mechanisms + thirteen micro primitives.

| Head           | Variants                                                                                                                  |
|----------------|---------------------------------------------------------------------------------------------------------------------------|
| `macro_kind`   | `NoOp`, `PostQuest{...}`, `AcceptQuest{quest_id, role_in_party}`, `WithdrawQuest{quest_id}` (taker retracts), `Bid{auction_id, payment, conditions}`, `Announce{audience, fact_ref}`, `InviteToGroup{kind, target, terms}`, `AcceptInvite{invite_id}`, `SetStanding{target_group, kind}` (Hostile/Friendly/Vassal/Neutral/Suzerain — universal backbone for war/peace/alliance, §9 #19) |
| `micro_kind`   | `Hold`, `MoveToward(pos)`, `Flee(from)`, `Attack(t)`, `Cast(ability, target)`, `UseItem(slot, target)`, `Harvest(node)`, `Eat(food)`, `Drink(water)`, `Rest(loc)`, `PlaceTile(pos, type)`, `PlaceVoxel(pos, mat)`, `HarvestVoxel(pos)`, `Converse(t)`, `ShareStory(audience, topic)`, `Communicate(recipient, fact_ref)`, `Read(doc)`, `Remember(entity, valence, kind)` |

Enums carried as parameter heads:

- `QuestType` — Hunt, Escort, Deliver, Explore, Defend, Gather, Rescue, Assassinate, Custom, Conquest, MutualDefense, Submit, Found, Charter, Diplomacy, Marriage, Pilgrimage, Service, Heist, Trade, FulfillProphecy, **Claim**, **Peace**, **Raid**, **HaveChild**. (Claim/Peace/Raid extensions from `stories_H.md` §46, §51; HaveChild from `stories_H.md` §52.)
- `PartyScope` — `Individual(AgentId)`, `Group(GroupId)`, `Settlement(SettlementId)`, `Anyone`, `Role(RoleTag)`.
- `QuestTarget` — `Agent`, `Group`, `Location`, `Structure`, `CreatureType`, `Pair(Target, Target)`, `Predicate`, `Item`, `Region`, **`Role(RoleTag)`** (for Claim quests).
- `RewardKind` — `Gold`, `Xp`, `Items`, `Reputation`, `Faith`, `Spoils`, `Charter`, `Union`, `Reciprocal`, `Protection`, `Glory`, `Promise`, `Combination`.
- `PaymentKind` — `Gold`, `Commodity`, `Item`, `Service`, `Reputation`, `Combination`.
- `GroupKind` — for `InviteToGroup`: Faction, Family, Guild, Religion, Pack, Party, Settlement, Court, Cabal, Alliance, Coven, Other.

Invite and auction coverage (`stories_F_G_I_J.md` §63, `stories_H.md` §54):

| Invite / auction usage       | `macro_kind`        | Parameters                                                         |
|------------------------------|---------------------|--------------------------------------------------------------------|
| Marriage proposal            | `InviteToGroup`     | `kind=Family` (spouse role), `target=AgentId`                     |
| Alliance petition            | `PostQuest`+auction | `AuctionKind::Diplomatic`, `AuctionItem::AllianceTerms`            |
| Vassal petition              | `InviteToGroup`     | `kind=Vassal` (hierarchical standing)                              |
| Party recruitment (ad-hoc)   | `InviteToGroup`     | `kind=Party`                                                       |
| Apprenticeship               | `InviteToGroup`     | `kind=Family` (mentor/apprentice sub-role)                         |
| Job offer                    | `InviteToGroup`     | `kind=Guild`                                                       |
| Guild admission              | `InviteToGroup`     | `kind=Guild`                                                       |
| Peace treaty                 | `PostQuest`         | `type=Peace`, target=rival Group                                   |
| Mercenary hiring             | `PostQuest`+auction | `AuctionKind::Service`                                             |
| Market commodity auction     | emergent            | agents emit `Bid` on posted auctions                               |
| Charter petition             | `PostQuest`+auction | `AuctionKind::Charter`, `Payment::Reputation` permitted            |
| Coup                         | `PostQuest`         | `type=Conquest`, party=cabal of loyalists, target=own Group        |

Gameplay verbs like Pray, DefendDen, Raid-by-opportunity are `verb` declarations over these primitives.

### 3.4 Reward, value, advantage, training

Reward is per-tick scalar. The value, advantage, and training sub-blocks select the algorithm variant without touching reward semantics. (`stories_D.md` §20)

```
reward {
  delta(self.needs.satisfaction_avg)               × 0.1
  delta(self.hp_frac)                              × 5
  +1.0 on event(EntityDied{killer: self, target.team != self.team})
  -1.0 on event(EntityDied{target in self.close_friends})
  +0.05 per behavior_tag accumulated this tick
  +2.0 on event(QuestCompleted{quest.party_member_ids contains self})
  -1.0 on event(QuestExpired{quest.party_member_ids contains self})
  +0.02 × delta(self.reputation_log)
  -0.5  on event(self.role transition: Leader -> Outlaw)
  -0.5  on event(LeaveGroup{agent: self, role_in_group: Leader})
}

value {
  head scalar v_pred
  trunk    shared                    // shared | separate
  loss     mse                       // mse | huber
  clip_range 10.0
}

advantage {
  kind       gae                     // gae | nstep | montecarlo
  gamma      0.99
  lambda     0.95
  normalize  per_batch               // per_batch | per_agent | none

  // Per-head gamma overrides for macro credit assignment (stories_D.md §20 gaps)
  macro      { gamma: 0.999 }
  micro      { gamma: 0.99  }
}

training {
  algorithm ppo {
    clip_epsilon   0.2
    vf_coef        0.5
    entropy_coef   0.01
    n_epochs       4
    minibatch_size 4096
    target_kl      0.02
  }
  // or:
  // algorithm reinforce { baseline: v_pred }
  // algorithm bc        { loss: cross_entropy }

  optimizer adamw  { lr: 3e-4, beta2: 0.98, weight_decay: 1.0 }
  grokfast  ema    { alpha: 0.98, lamb: 2.0 }
}

action {
  head categorical macro_kind: enum {
    NoOp                              @training_weight(1.0),
    PostQuest                         @training_weight(50.0) @rare,
    AcceptQuest                       @training_weight(20.0),
    Bid                               @training_weight(10.0),
    InviteToGroup                     @training_weight(20.0),
    AcceptInvite                      @training_weight(15.0)
  }
  head categorical quest_type: enum QuestType {
    Hunt, Escort, ...                    // defaults to 1.0
    Conquest      @training_weight(100.0),
    Marriage      @training_weight(50.0),
    Found         @training_weight(100.0),
    Claim         @training_weight(80.0)
  }
}
```

Per-action `@training_weight` drives both prioritised replay sampling and per-head loss scaling. Weights can be overridden per curriculum stage (§2.10).

### 3.5 Backend

A single trait. Three implementations (`proposal_policy_schema.md` §2.4).

```
trait PolicyBackend {
    fn evaluate_batch(
        &self,
        observations: &PackedObservationBatch,   // [N, OBS_DIM]
        masks:        &PackedMaskBatch,
    ) -> ActionBatch;                             // [N] typed Action
}
```

- **`backend Neural`** — production. Loads weights from `path`, runs one GPU or CPU forward pass per tick, samples per-head with mask.
- **`backend Utility`** — bootstrap and regression baseline. Declarative scoring rules, argmax over masked candidates, softmax-with-temperature for calibrated log-probs when emitting trajectories for BC (`stories_D.md` §19).
- **`backend Llm`** — research. Serialises observation to JSON, sends to an external model, parses the action. Off the per-tick path; used to seed BC datasets.

Backend assignment is per-policy. Assigning backends per-role is allowed but discouraged — role differentiation lives in mask and observation, not backends.

Checkpoint layout (`stories_D.md` §23):

```
<policy>.bin
├── header (128 bytes):
│   ├── magic "NPCPOL\0\0"          (8)
│   ├── format_version  u32          (4)
│   ├── schema_hash     [u8; 32]     (32)  // SHA256 over obs + action + event taxonomy + reward
│   ├── policy_name     [u8; 32]     (32)
│   ├── training_step   u64          (8)
│   ├── stage_name      [u8; 16]     (16)
│   ├── reserved        [u8; 20]     (20)
│   ├── weights_offset  u64          (8)
├── weights section (named tensor blobs, safetensors-style)
└── footer: CRC32 over weights
```

Hot-swap semantics: file watcher triggers reload at the next tick boundary, schema-hash check fails loud on mismatch, atomic pointer swap on success (`stories_D.md` §23).

---

## 4. Schema versioning

The schema hash is a content-addressed fingerprint over:

1. Observation layout — field order, offsets, types, normalization constants, one-hot vocabularies.
2. Action vocabulary — head enums, variant lists, parameter-head combinations.
3. Event taxonomy — declared event names and field shapes.
4. Reward declaration — components and weights.

The compiler emits four sub-hashes and one combined hash:

```
schema.observation_hash = sha256(canonicalize(observation_schema))
schema.action_hash      = sha256(canonicalize(action_vocabulary))
schema.event_hash       = sha256(canonicalize(event_taxonomy))
schema.reward_hash      = sha256(canonicalize(reward_block))
schema.combined_hash    = sha256(observation_hash || action_hash || event_hash || reward_hash)
```

Loading a checkpoint whose `combined_hash` differs from the current DSL is a hard error. (`stories_C.md` §15, `stories_D.md` §23, `stories_F_G_I_J.md` §64.) The error prints a diff of the four sub-hashes, a textual diff of which fields/variants changed, and a git-remediation hint:

```
error: policy checkpoint schema mismatch
  checkpoint: generated/npc_v3.bin (trained 2026-04-10, step 1_400_000)
  checkpoint schema_hash: sha256:a1b2c3...7890
  current DSL schema_hash: sha256:e4f5g6...2345
  diff:
    + appended observation: self.war_exhaustion (offset 1655, size 1, norm identity)
    + appended action variant: macro_kind::InviteToGroup (slot 4)
    + appended event: InvitePosted
  action: retrain from current DSL, or git-checkout the commit whose
          schema_hash matches the checkpoint.
```

There are no `@since` annotations. There are no padded-zero migration tables. There are no v1/v2/v3 schemas in the codebase — git holds history. Two branches with different schemas produce mutually incompatible checkpoints, which is correct.

CI guard: a commit that modifies observation, action, event, or reward declarations computes the pre- and post-change hashes; non-append changes (remove, reorder, type change, norm change) block merge without an explicit checkpoint bump.

This rewrites `proposal_policy_schema.md` §4.

---

## 5. Type system

### 5.1 Primitive and structural types

```
Scalar:    f32, i32, u32, u64, bool
Vector:    vec2, vec3, vec4
Time:      Tick (u64)
ID:        AgentId, ItemId, GroupId, QuestId, AuctionId, InviteId, StructureRef,
           EventId, PredicateId, TagId, ArchetypeId, RoleTag
FactRef:   { owner: AgentId, event_id: u32 }  // ephemeral handle into live memory
FactPayload: { tick, kind, params: [u32; 4], author_id: AgentId }
                                                 // durable materialisation used by
                                                 // Quest.fact_refs and Document.facts (§9 #29)
Bounded:   Bitset<N>,              // fixed-size bit vector
           SortedVec<T, K>,        // sorted, cap K, partial-sort eviction
           RingBuffer<T, K>,       // fixed ring, oldest-evict
           Map<K, V, Cap>,         // fixed-capacity ordered map
           OneOf<K, V>             // sum-type union with fixed discriminant
Struct:    Membership, Relationship, MemoryEvent, AgentData, Group, Item,
           Quest, Auction, Invite, Document, ChronicleEntry (structural-only)
Enum:      Source (Witnessed / TalkedWith / Overheard / Rumor / Announced /
                   Testimony),
           AnnounceAudience (Group / Area / Anyone),
           ...plus §3.3 parameter heads + CreatureType, ItemKind (incl.
           Document), GroupKind, AuctionKind, AuctionResolution,
           GovernanceKind, RoleTag, ChronicleCategory, DropReason, ...
```

`@spatial` types expose `pos: vec3` and participate in 3D spatial indices. Every bounded collection declares its capacity; capacity is part of the schema hash.

### 5.2 Forbidden types

`String` is forbidden on `@replayable` events, on `@primary` state fields, and on any field referenced by a mask or reward predicate. Strings are permitted only on `@non_replayable` events, the `chronicle_prose` side channel, and at display time. All load-bearing references are IDs (`stories_F_G_I_J.md` §60). `class_tags` and `archetype` agent fields carry `TagId` / `ArchetypeId` with a compile-time string table.

Unbounded `Vec<T>` is forbidden on `entity`, `event`, and in-world struct declarations. Only `SimScratch` pools and world-level ring buffers may hold `Vec`. The compiler rejects:

```
entity BadAgent : Agent { history: Vec<Event> }    // ERROR: unbounded collection
```

### 5.3 Structural type relationships

- `Agent.pos: vec3` — world-space 3D position. Ground-locked creature_types have `pos.z` snapped by the post-phase cascade in §7.
- `Agent.movement_mode: enum { Walk, Climb, Fly, Swim, Fall }` — cascade-updated; drives spatial-index sidecar (§9 #25). Slopes are `Walk`.
- `Agent.spouse_ids: SortedVec<AgentId, 4>` — polygamous-capable; cross-species gated by `can_marry(a, b)` view (§9 #17).
- `Relationship.believed_knowledge: Bitset<32>` — theory-of-mind projection of `knowledge_domain_bits`.
- `Relationship.believed_knowledge_refreshed: [u32; 32]` — per-bit last-reinforced tick; decay computed at read time using per-bit volatility (§9 #28).
- `KnowledgeDomain.volatility: enum { Short=500, Medium=20_000, Long=1_000_000 }` — compile-time per-variant tag; sets half-life in ticks.
- `Agent.memberships: SortedVec<Membership, 8>` — capped at K=8 simultaneous groups.
- `Agent.known_actors: SortedVec<Relationship, 32>` — LRU by `last_known_age`; pinned (spouse, mentor) entries survive eviction.
- `Agent.memory_events: RingBuffer<MemoryEvent, 64>` — with up to 5 "indelible" slots reserved for `emotional_impact > 0.8`. `MemoryEvent` carries `source: Source` and `confidence: f32`.
- `Agent.behavior_profile: SortedVec<(TagId, f32), 16>` — capped; lowest-weight eviction.
- `Document: Item{ kind: Document, author_id: AgentId, tick_written: Tick, seal: Option<SealId>, facts: SortedVec<FactPayload, 16> }` — no stored `trust_score`; reader derives confidence on `Read(doc)` from `(relationship_to_author, seal_validity, known_author_biases)` (§9 #27). Forgery is an emergent skill-gated quest.
- `Quest.fact_refs: SortedVec<FactPayload, 8>` — materialised at `QuestPosted` from the poster's memory (§9 #29).
- `Quest.child_quest_ids: SortedVec<QuestId, 8>` — sub-quest tracking; populated by `QuestPosted{parent_id: Some(_)}` cascade; drives recursive cancellation and spoils distribution (§9 #15).
- `Group.members` — derived view over agent `memberships`, not stored.
- `Group.standings: SortedVec<(GroupId, StandingKind, f32), 16>` — updated by `SetStanding` macro; "alliance" / "war" are emergent thresholds over `standing_value` (§9 #19).
- `Quest.party_member_ids` — resolved at read time from `party_scope` against current group membership; no stored list.
- `Quest.reward_evaluation: enum { CommitAtPosting, ComputeAtCompletion }` — every reward variant evaluates at completion (§9 #8). `CommitAtPosting` reserved for rare designer-opt-in cases; not the default.
- `ChronicleEntry` — `{ tick, category: ChronicleCategory, entity_ids: [AgentId; 4], template_id: u32 }` with text via side-channel `chronicle_prose: Map<ChronicleEntryId, String>`.

---

## 6. Compilation targets

### 6.1 Native Rust (CPU)

- SoA buffers per entity kind, with `@hot` / `@cold` field partitioning (§6.3).
- Rayon-parallel iteration on `[N, OBS_DIM]` observations, masks, and action application.
- Per-agent kernels as `fn` with `#[inline(never)]` in profiling builds for flamegraph attribution (`stories_F_G_I_J.md` §40).
- `SimScratch` pools carry all per-tick scratch — zero steady-state allocation. Agent slot pool sized at init; ring buffers fixed-cap; event buffers use `SmallVec<[T; N]>` with CI-enforced worst-case bounds (`stories_E.md` §30).
- **Spatial index is 2D-grid + per-column sorted z-list + movement-mode sidecar** (§9 #25). Primary structure keys `(cx, cy) → SortedVec<(z, AgentId)>` with 16m cells matching voxel-chunk edges. Planar queries walk 9 columns (3×3) and take all. Volumetric queries walk 9 columns and binary-search the z-range. Agents with `movement_mode != Walk` (Fly / Swim / Climb / Fall) live in a separate `in_transit: Vec<AgentId>` that every spatial query scans linearly (expected |in_transit| ≪ N). Slope-walkers stay in the column index — the structure exploits floor-clustering, not flat-ground assumptions.
- RNG: a single `rng_state: u64` per world, consumed in a fixed order (`stories_E.md` §29). Per-agent RNG streams seeded from `hash(world_seed, agent_id, tick, purpose)` for parallel sampling.

### 6.2 GPU (`voxel_engine::compute::GpuHarness`)

Target is voxel-engine's Vulkan/ash + gpu-allocator stack via `GpuHarness`, not wgpu and not raw CUDA. (`stories_E.md` §28.) Shader codegen emits SPIR-V via `shaderc` (already in voxel-engine's `build-dependencies`), loaded through `GpuHarness::load_kernel`. Precedents: `terrain_compute.rs` (1024-slot LRU chunk pool), `ai/spatial.rs` (spatial indexing).

The DSL compiler emits a `PolicyRuntime`:

```rust
pub struct PolicyRuntime {
    harness:       voxel_engine::compute::GpuHarness,
    obs_field:     FieldHandle,      // [N, OBS_DIM] f32 / f16
    mask_field:    FieldHandle,      // per-head boolean buffers
    logits_field:  FieldHandle,      // [N, NUM_LOGITS] f32
    action_field:  FieldHandle,      // [N] packed action rows
    weights_field: FieldHandle,      // safetensors-style
    event_ring:    FieldHandle,      // GPU-resident event buffer (replayable subset)
}

impl PolicyRuntime {
    pub fn tick(&mut self, world: &WorldState) -> &[Action] {
        upload_event_ring_delta(&mut self.harness, world);
        self.harness.dispatch("pack_observations", &[...], [n_groups, 1, 1])?;
        self.harness.dispatch("eval_mask_micro",    &[...], [n_groups, 1, 1])?;
        cpu_patch_mask_for_cross_entity(&mut self.harness, world);
        self.harness.dispatch("mlp_forward",        &[...], [n_groups, 1, 1])?;
        self.harness.dispatch("sample_with_mask",   &[...], [n_groups, 1, 1])?;
        self.harness.download(&ctx, &self.action_field)
    }
}
```

GPU-amenable kernels:

- Observation packing (structural gather over SoA agent fields). Per-slot `relative_pos: vec3` + `z_separation_log` pack contiguously — no layout change beyond width.
- Mask evaluation for intrinsic scalar predicates, including `distance` / `planar_distance` / `z_separation`.
- Neural forward (hand-emitted fused GEMM + activation shaders specialised per network topology, matching existing Grokking transformer pattern).
- Event-fold materialization for commutative scalar views (sort events by target before reduction to preserve determinism).
- 3D spatial hash (voxel-chunk-keyed) for `query::nearby_agents` — reuses `voxel_engine::ai::spatial` infrastructure.

CPU-only:

- Cascade rules with cross-entity walks (`t in quest.eligible_acceptors`, `at_war(self, f)`).
- LLM backend.
- Chronicle prose rendering.
- Quest-eligibility and auction-eligibility indices.
- Mixed CPU/GPU mask patching: GPU writes initial mask, CPU patches cross-entity bits, GPU sampler reads final mask (one fence per tick).

GPU determinism constraints (`stories_E.md` §29):

- Reductions feeding policy decisions use integer fixed-point or sorted-key accumulation to avoid float-associativity drift.
- Materialized views sort events by `target_id` before atomic accumulation.
- Reduction shader workgroup size is pinned via specialization constants.
- Policy sampling seeds from `hash(world_seed, agent_id, tick, "sample")` so parallel sampling is deterministic.

### 6.3 Hot/cold storage split

Mandatory at 200K scale (`stories_E.md` §31). Authors annotate Agent fields with `@hot` or `@cold`; the compiler emits two SoA layouts and a per-tick sync schedule.

```
entity Agent {
  // Hot — resident, packed into observation buffer
  @hot pos:              vec3,
  @hot hp:               f32,
  @hot max_hp:           f32,
  @hot shield_hp:        f32,
  @hot needs:            [f32; 6],
  @hot emotions:         [f32; 6],
  @hot personality:      [f32; 5],
  @hot memberships:      SortedVec<Membership, 8>,

  // Cold — paged, loaded on policy-tick for High fidelity only
  @cold memory_events:   RingBuffer<MemoryEvent, 64>,
  @cold behavior_profile: SortedVec<(TagId, f32), 16>,
  @cold class_definitions: [ClassSlot; 4],
  @cold creditor_ledger: [Creditor; 16],
  @cold mentor_lineage:  [AgentId; 8],
}
```

Fidelity gating: `@fidelity(>= Medium)` on a view or cascade skips evaluation for Background-fidelity agents. Background agents skip policy inference and cold-field access.

Target: hot ≤ 4 KB/agent, 200K × 4 KB = 800 MB; cold paged to SSD with LRU. Cold fields for non-High agents are swapped out.

---

## 7. Runtime semantics

### 7.1 Tick pipeline

```
pre phase    (@phase(pre) rules; read-only state access)
       ↓
event emission (policy actions + world events emitted this tick)
       ↓
materialized-view updates (event-fold per @materialized view, sorted-key reduction)
       ↓
observation pack (GPU dispatch; one row per alive agent)
       ↓
mask evaluation (GPU + CPU patch for cross-entity)
       ↓
policy forward + sample (one evaluate_batch call)
       ↓
action application (shuffled-but-seeded order via rng_state)
       ↓
cascade fixed-point (@phase(event) rules; up to terminating_in(N) hops)
       ↓
post phase   (@phase(post) rules; ground-snap, overhear scan,
              chronicle / telemetry emission)
       ↓
telemetry metrics updated; alerts emitted
```

**Ground-snap phase** — a post-MovementApplied cascade rule for ground-locked `creature_type`s:

```
physics ground_snap @phase(post) {
  on MovementApplied{agent: a} where creature_type(a).ground_locked {
    let h = match a.inside_building_id {
      Some(b) => floor_height(a.pos, b),
      None    => surface_height(a.pos.xy),
    };
    emit SetPos { agent: a, pos: vec3(a.pos.x, a.pos.y, h + creature_height(a) / 2) }
  }
}
```

Volumetric `creature_type`s (Dragon, Fish, Bat) have no matching rule; their `pos.z` persists as emitted.

**Announce propagation** — a cascade rule triggered by `Announce`:

```
physics announce_broadcast @phase(event) {
  on AnnounceAction{speaker: s, audience: aud, fact_ref: f} {
    let recipients = match aud {
      Group(g)    => members_of(g).filter(|r| hearing_eligible(s, r))
                                   .take(MAX_ANNOUNCE_RECIPIENTS),
      Area(c, r)  => query::nearby_agents_3d(c, r.min(MAX_ANNOUNCE_RADIUS))
                                   .filter(|r| r != s && r.can_hear)
                                   .take(MAX_ANNOUNCE_RECIPIENTS),
      Anyone      => query::nearby_agents_3d(s.pos, MAX_ANNOUNCE_RADIUS)
                                   .filter(|r| r != s && r.can_hear)
                                   .take(MAX_ANNOUNCE_RECIPIENTS),
    };
    for r in recipients {
      emit RecordMemory { observer: r, payload: copy_fact(f),
                          source: match aud { Group(g) => Announced(g), _ => Overheard(s) },
                          confidence: 0.8 }
    }
    // Overhear scan — bystanders not in the audience (§9 #26)
    for b in query::nearby_agents_3d(s.pos, OVERHEAR_RANGE)
                 .filter(|b| b != s && !in_audience(b, aud) && overhear_eligible(s, b)) {
      let cat = overhear_category(s, b);  // SameFloor | DiffFloor | Outdoor
      let base = match cat { SameFloor => 0.75, DiffFloor => 0.55, Outdoor => 0.50 };
      let conf = base * exp(-planar_distance(s, b) / OVERHEAR_RANGE);
      emit RecordMemory { observer: b, payload: copy_fact(f),
                          source: Overheard(s), confidence: conf }
    }
  }
}
```

Runtime constants: `MAX_ANNOUNCE_RECIPIENTS` (default 64) bounds the per-emission cascade; `MAX_ANNOUNCE_RADIUS` (default 100m) caps `Anyone` / `Area` reach; `OVERHEAR_RANGE` (default 10m) drives bystander scan; `CONVERSE_RANGE` (default 5m) for `Communicate`. Overhear eligibility: `same_building(s, b)` (any floor) OR `planar_distance(s, b) < OVERHEAR_RANGE ∧ z_separation(s, b) < 3.0`.

### 7.2 Determinism contract

- All randomness flows through `rng_state: u64` (PCG-style) plus per-agent derived streams (§9 #12). Per-agent RNG seeded from `hash(world_seed, agent_id, tick, purpose_tag)`; no stored per-agent RNG state. Never `thread_rng` or external RNG inside the simulation.
- Agent processing order is deterministically shuffled each tick via an RNG-seeded permutation.
- Events emitted within a phase are collected in append order; handlers process them deterministically.
- HashMap iteration seeded from `hash(world_seed ^ tick)` at world init; never re-seeded.
- Wall-clock reads (`SystemTime`, `Instant`) are forbidden in DSL code. Only the tick counter is available.
- GPU reductions over floats use sorted-key or fixed-point accumulation.
- Text generation (`ChronicleEntry.text`, LLM prose) is `@non_replayable` (§9 #21): templates render eagerly at event emission, an async LLM pass may rewrite entries in flagged categories (`Legendary`, `Founding`, `Death`, `Prophecy`). Saved prose is canonical — template-library changes only affect future entries. Bug-report replay artefacts bundle the template library; casual replay uses the current library.

Tests in `src/ai/core/tests/determinism.rs` (runtime) and the compiler's invariant checker verify that:

- `sim(seed).step(N).save() + load() + step(M) ≡ sim(seed).step(N + M)` for all primary state.
- Any policy code reading `event.text` fields is rejected at compile time.

### 7.3 Replay scope

Practical replay window is the bug-report scope (~1000 ticks, ~25 GB uncompressed, 2–5 GB compressed via zstd) (`stories_E.md` §33). Full-run replay at 200K agents (~5 TB/hour raw event volume) is not a goal.

Event-log storage policy (§9 #14): (a) filter to the replayable event-type subset only, (c) snapshot cadence **N=500 ticks** for rollback granularity, (d) zstd compression on both snapshot and event-log segments. Dev-only per-save rollback lets time-travel debugging jump to any 500-tick boundary inside the current bug-report window.

Replay artefacts per recorded segment:

1. Initial snapshot (zstd-compressed safetensors).
2. Event log with `{ tick, kind: u16, params: [u32; 4], source_agent: u32 }` — numeric IDs only, zstd-framed per 500-tick segment.
3. Tick-boundary RNG checkpoints every 100 ticks.
4. Policy checkpoint hash (forces weight-compatible replay).
5. (Bug-report artefacts only, §9 #21) Chronicle template library snapshot for prose-exact reproduction.

Non-replayable text is regenerated from templates on playback.

### 7.4 Save/load

Snapshot contents:

- `tick`, `rng_state`, `next_id`, `max_entity_id`.
- Primary fields of all entities (hot + cold).
- Group definitions (memberships, standings, treasuries, active_quests).
- Quest / auction / invite state.
- `tiles`, `build_seeds`, `voxel_world.chunks`, `voxel_world.sea_level`, `region_plan.seed`.
- Bounded rings: chronicle, world_events, per-agent memory_events.
- Materialized views (serialized) with schema-hash guard. Mismatch forces rebuild from baseline.
- Event log ring (fixed-cap).
- GPU-resident buffers downloaded and serialized; re-uploaded on load.
- Policy weight hash.

Snapshot format: safetensors-compatible, length-prefixed, deserialize-in-place. Zero-malloc load uses pre-allocated agent slot pool.

---

## 8. Worked example

Three agents A, B, C in a settlement. A proposes marriage to B; cascade forms a Family group; B's membership updates; C witnesses and gains a positive memory.

**State before** (relevant fields only):

```
Agent A { id=1, pos=vec3(10, 5, 42.1), creature_type=Human, can_marry=true,
          married=false, personality.social_drive=0.7,
          memberships=[Settlement{s=1, role=Member}] }
Agent B { id=2, pos=vec3(11, 5, 42.1), creature_type=Human, can_marry=true,
          married=false, memberships=[Settlement{s=1, role=Member}] }
Agent C { id=3, pos=vec3(12, 6, 42.0), creature_type=Human,
          memberships=[Settlement{s=1, role=Member}] }

// surface_height((10, 5)) = 41.3; creature_height(Human) = 1.6; ground-snap keeps
// pos.z = 42.1 for all three after prior movement.

view relationship(1, 2) = { valence=0.72, familiarity=0.60, last_interaction=tick-5 }
```

**Tick T — A acts.** Observation packing runs; A's policy sees B in `known_actors[0]` and emits:

```
action {
  macro_kind  = InviteToGroup
  group_kind  = Family
  target      = known_actors[0]      // resolves to agent_id = 2
}
```

**Mask evaluation (A):**

```
mask InviteToGroup{kind: Family, target: Agent(2)}
  when A.can_marry            = true
     ∧ ¬married(A)            = true
     ∧ ¬married(B)            = true
     ∧ relationship(A, 2).valence > MIN_TRUST (= 0.5)   = 0.72 > 0.5  ✓
     ∧ ¬A.in_active_invite_to(2, Family)              = true
  → mask passes
```

**Cascade fires (@phase(event)):**

```
physics invite_posted @phase(event) {
  on InviteToGroup{inviter: u, target: t, kind: k, terms: tm} {
    emit InvitePosted { invite_id: next_invite_id(), inviter: u, target: t,
                        kind: k, terms: tm, expires_tick: now + 500 }
  }
}

→ emit InvitePosted { invite_id: 42, inviter: 1, target: 2,
                      kind: Family, expires_tick: T+500 }
```

**Tick T+1 — B observes the invite.** B's `incoming_invites[0]` surfaces invite 42. B's policy reads the inviter's relationship summary (valence=0.72), the kind (Family), and emits:

```
action {
  macro_kind = AcceptInvite
  target     = incoming_invites[0]    // resolves to invite_id = 42
}
```

**Mask evaluation (B):**

```
mask AcceptInvite{invite_id: 42}
  when 42 in B.incoming_invites             = true
     ∧ invite(42).kind_acceptable_to(B)      = true  (B is eligible for Family)
     ∧ B.eligible_for(invite(42).kind)       = true
     ∧ invite(42).expires_tick > now         = true
  → mask passes
```

**Cascade fires (@phase(event)):**

```
physics invite_accepted @phase(event) {
  on AcceptInvite{invite_id: id} {
    let inv = invite(id);
    match inv.kind {
      Family => {
        emit MarriageFormed { a: inv.inviter, b: inv.target, tick: now }
      }
      Alliance => { ... }
      ...
    }
    emit InviteResolved { invite_id: id, accepted: true }
  }
}

physics marriage_formed @phase(event) {
  on MarriageFormed{a: x, b: y, tick: t} {
    emit FoundGroup { kind: Family, founder_ids: [x, y], name_seed: (x ^ y) }
    emit RecordMemory { observer: x, event_type: MarriedB, entity_ids: [y],
                        emotional_impact: 0.7 }
    emit RecordMemory { observer: y, event_type: MarriedA, entity_ids: [x],
                        emotional_impact: 0.7 }
  }
}

physics witness_cascade @phase(post) {
  on MarriageFormed{a: x, b: y} {
    for w in query::nearby_agents_3d(pos(x), WITNESS_RADIUS)
             where w != x && w != y
                && z_separation(w, x) < 3.0 {
      emit Witnessed { observer: w, subject: x, kind: MadeNewFriend,
                       location: pos(x), entity_ids: [x, y],
                       emotional_impact: 0.15, source: Witnessed,
                       confidence: 1.0 }
    }
  }
}
```

**Agent C (nearby, same altitude) witnesses.** Cascade emits a `Witnessed` event; C's `memory_events` ring ingests it with `source=Witnessed, confidence=1.0`; C's `joy` view bumps next observation.

**Tick T+2 — A announces to the settlement.** A's policy, observing the just-completed marriage as a fresh memory event, emits:

```
action {
  macro_kind         = Announce
  announce_audience  = Group(settlement_s1)
  fact_ref           = recent_memory_events[0]   // resolves to A's MarriageFormed memory
}
```

Cascade enumerates up to `MAX_ANNOUNCE_RECIPIENTS` settlement members within hearing range, emits `RecordMemory { observer: r, source: Announced(s1), confidence: 0.8 }` for each. Bystanders outside the settlement group but within `OVERHEAR_RANGE` of A receive `source: Overheard(A), confidence: 0.6`. This seeds gossip propagation: a recipient's later `Communicate` rebroadcasts the fact as `source: TalkedWith`, decrementing confidence per the rules in `proposal_universal_mechanics.md`.

**State after (tick T+2):**

```
Agent A { id=1, memberships=[Settlement{s=1}, Family{g=99, role=Founder}] }
Agent B { id=2, memberships=[Settlement{s=1}, Family{g=99, role=Founder}] }
Agent C { id=3, memory_events.push(Witnessed{MadeNewFriend, [A, B], 0.15}) }
Group  99 { kind=Family, founder_ids=[1, 2], members=[1, 2], founded_tick=T+1 }

view mood(A) += 0.05 (from NeedsSatisfied via social_drive fulfillment)
view relationship(C, A).familiarity += 0.02
```

The entire cascade is 4 events (`InvitePosted`, `AcceptInvite` trigger, `MarriageFormed`, `FoundGroup` + memory + witness), one mask per action, two policy decisions (A's emit + B's emit), and one post-phase spatial query for witness cascade. No special "marriage system" exists; the outcome falls out of `verb` declarations, cascade rules, and views.

---

## 9. Settled decisions

All 29 open questions from the prior revision have been resolved through design interviews. Entries record the decision and one-line rationale; authoritative detail lives in the cross-referenced schema sections.

### 9.1 Action / quest mechanics

1. **Auction state machine** — `Resolution` enum = `{HighestBid, FirstAcceptable, MutualAgreement, Coalition{min_parties: u8}, Majority}`. `Coalition` for multi-party diplomatic pacts (stories_H §54); `Majority` for contested succession (story 46). `PostAuction` is an alias of `PostQuest{kind: Diplomacy|Charter|Service}` — no separate macro head. Cadence is per-world config, not compiled-in.
2. **Macro head firing rate** — Macro head runs **every tick**. Most emissions are `NoOp`; the network budgets cost via small macro sub-network + large micro pipeline. Avoids stale-context issues from gated macro inference.
3. **Macro credit assignment** — **GAE(γ=0.99, λ=0.95)** per head with per-head γ override (`γ_macro = 0.999` for 2000+-tick Conquest). Reward deltas mark credit windows; no hand-scheduled per-scenario γ.
4. **Quest discovery push/pull hybrid** — **Hybrid**: posting emits `Announce(fact_ref=quest_id, audience)` cascade → recipients get the quest into their `known_quests` slot. Physical proximity + `GatherInformation` intent produces pull-style discovery for quests outside the announce radius. No always-on public broadcast; quest visibility is behaviour-driven.
5. **Slot K tuning** — **K=12 across the board** for spatial slots; K for non-spatial slots scales with role (leaders get larger `known_actors`, `known_groups`). Emergent behaviour and reputation dynamics are option-A ("agent must actively seek information") — no auto-populated global views.
6. **Cross-entity mask index design** — **Hybrid**: eager index materialization for `standing(group_a, group_b)`, `quest.eligible_acceptors`, `same_building`; lazy scan for low-cardinality predicates. Indices rebuild on the owning event (`GroupStandingChanged`, `QuestPosted`, `ChunkEnteredBuilding`).
7. **Concurrent quest membership** — **Multi-quest list** (up to K=4 active). Information primitives support multi-step plans (scout → report → assault chain). Mutual exclusion is policy-level against internal state, not schema-enforced; the policy can choose to decline a second quest that conflicts.
8. **Reward delivery on long quests** — **Compute at completion from current state**. No escrow. Poster reliability becomes emergent reputation: `QuestDefault` event fires if poster can't pay; accepters price the risk via `believed_intrinsic_value(poster_reputation, reward)`. Behavior-driven quest economics.
9. **Cancellation / amendment** — **`WithdrawQuest` macro head for taker only** (accepter changes mind, pays reputation cost). Poster amendment is forbidden; a poster who wants different terms must post a new quest. Simpler semantics; forces honest initial terms.
10. **Bid currency parity** — **`Payment::Combination{material, immaterial}` with per-agent valuation**. `intrinsic_value(observer, payment)` view computes subjective worth; `believed_intrinsic_value(self, other, payment)` gives theory-of-mind for bid crafting. Material and immaterial contributions are both required; ratio is per-buyer preference.
15. **Nested quest cancellation** — **C with B as acceleration structure**: parent Quest has `child_quest_ids: SortedVec<QuestId, 8>` materialised by the `QuestPosted{parent_id: Some(...)}` cascade. Parent cancel emits a cascade that cancels children; reward cascades traverse child_quest_ids for spoils distribution. Child tracking is the acceleration index; cancellation is first-class.
17. **Polygamous / cross-species / multi-parent family** — **C for all three**: `Agent.spouse_ids: SortedVec<AgentId, 4>` (polygamy), `can_marry(a, b) → bool` view folds in `creature_type` compatibility (cross-species at designer opt-in), `ChildBorn{parents: [AgentId; 4], inheritance_blend}` supports multi-parent lineages.
18. **Mercenary / service payment direction** — **C**: `AuctionKind::Service` inverts roles via `AuctionParty::{Buyer=Patron, Seller=Labourer}` with `Payment` flowing Buyer→Seller on completion. Service commitments use `Payment::ServicePledge{duration_ticks, scope}` evaluated against `intrinsic_value` for comparison.
19. **Alliance obligation enforcement** — **C / C**: "alliance" is not a first-class concept; it is an emergent standing derived from `standing(group_a, group_b)`. `SetStanding(target, kind)` is the universal macro; "declare war" is `SetStanding(target, kind=Hostile)`; "alliance" is `standing ≥ Friendly` ∧ reciprocal. Default response to ally-under-attack is policy-governed, not mechanism-forced.
20. **Group-level invites vs agent-level invites** — **Agent-level only**. All invitations are agent-to-agent. Group mergers / coalitions use `PostQuest{kind: Diplomacy, resolution: Coalition{min_parties: K}}`. Removes the edge case where party-to-party agreement would tear factions apart.

### 9.2 Runtime / infrastructure

11. **LlmBackend distillation pipeline** — **B, part of the DSL runtime**. `backend "llm" { ... }` is a first-class DSL backend; trajectories are an opt-in export for Python training. No ML-algorithm details in DSL.
12. **Per-agent RNG streams** — **C**: per-agent RNG seeded from `hash(world_seed, agent_id, tick, purpose)`. Enables fully-parallel sampling without save/load complexity (no extra stored state; seeds are derived).
13. **Materialized-view restoration on load** — **A primary / C rollback**: views serialize with schema-hash guard; on mismatch, rebuild from event log if available, otherwise refuse load. Rollback (dev-only, per-save) enables time-travel debugging from snapshot boundaries.
14. **Event log storage compression** — **(a) + (c) + (d) combined**: event-type filtering to replayable subset, fixed snapshot cadence **N=500 ticks**, zstd compression codec. ~2–5 GB per ~1000-tick bug report window.
21. **Chronicle prose side-channel lifecycle** — **C / X / P**: eager template rendering at event emission; async LLM rewrite pass for flagged categories (`Legendary`, `Founding`, `Death`, `Prophecy`); saved prose is canonical across template changes (player-facing history doesn't retcon); replay artefacts bundle the template library for exact reproduction in bug reports.
22. **Probe default episode count** — **Config-definable** per world + per-probe override via `seeds [42, 43, ...]` syntax. No hard-coded default.
23. **Off-policy vs on-policy training dispatch** — **Out of DSL scope**. DSL emits a pytorch-compatible trajectory format (safetensors, flat tick-rows, `episode_end` flag, per-agent grouping). Training-script concerns (importance sampling, V-trace, Retrace, BC-vs-PPO dispatch) live in Python, not DSL.
24. **Utility backend retirement milestone** — **A**: Utility backend never retires. Remains a regression-baseline + untrained-world bootstrap path. Maintenance cost is bounded (~1 KLoC); removal optimises a number that doesn't matter.
25. **3D spatial hash structure** — **D**: 2D grid (cell=16m, voxel-chunk edge) with per-column sorted z-list, plus a `movement_mode ≠ Walk` sidecar (`Climb`, `Fly`, `Swim`, `Fall`). Slopes are Walk (they don't violate column-clustering). `movement_mode` is a primary field updated by the cascade.
26. **Overhear confidence decay** — **B + D hybrid**: category-based base + exponential distance decay. `base[SameFloor/DiffFloor/Outdoor] = {0.75, 0.55, 0.50}`; `confidence = base * exp(-planar_distance / OVERHEAR_RANGE)`. Walls are not raycast; wall structure is captured by category.
28. **`believed_knowledge` decay rate** — **3-tier volatility model**: `KnowledgeDomain` enum carries `volatility: {Short=500, Medium=20_000, Long=1_000_000}` ticks half-life. Reinforcement via observation-of-use, Communicate, and negative-evidence clearing. `Relationship.believed_knowledge_refreshed: [u32; 32]` stores per-bit last-refresh tick (~1 GB at 200k agents, acceptable).

### 9.3 Schema / memory

27. **Document trust_score authoring** — **C**: `trust_score` field removed. `Document: Item{kind=Document, author_id: AgentId, tick_written: u32, seal: Option<SealId>, facts: SortedVec<FactPayload, 16>}`. Reader computes confidence on `Read(doc)` from `(relationship_to_author, seal_validity, known_author_biases)`. Forgery is an emergent skill-gated quest.
29. **FactRef ownership after memory eviction** — **A**: materialize at creation. Quests and Documents store `FactPayload { tick, kind, params, author_id }` inline (16 × 20 B ≈ 320 B per doc). Memory rings evict freely; no refcount machinery; stale numeric IDs are an identity-layer concern, not a fact-layer concern.

### 9.4 Modding

16. **Mod event-handler conflict resolution** — **C (named lanes)**: handlers declare a lane `on_event(EventKind) in lane(Validation | Effect | Reaction | Audit)`. Lanes run in order; within a lane, handlers run in lexicographic mod-id (not install order). Multiple handlers per lane coexist (additive). Destructive overrides happen via forking the DSL source, not via a replace keyword.

---

A standing decision log (`docs/dsl/decisions.md`) carries rationale and reversal criteria in more detail; §9 summarises the current-state view.

---

## 10. What's NOT in scope

This DSL does not handle:

- Text generation, LLM prose, dialogue authoring — `@non_replayable` side channel; out of deterministic sim.
- Asset pipeline — mesh, texture, shader assets for the voxel renderer live elsewhere.
- Rendering — voxel meshing, marching-cubes, SDF rendering are voxel-engine concerns.
- Networking / multiplayer synchronization — snapshot format exists but replication protocol is not specified.
- Save-file format with asset references — snapshots carry simulation state only; asset binding is a separate layer.
- Audio, particle effects, UI — display layers outside the sim loop.
- Build-system integration beyond `shaderc` SPIR-V compilation — the DSL compiler is a cargo xtask, not a full build tool.
- Online learning / federated training — all training is offline over serialized replay buffers.
- Human-in-the-loop labelling — probes are the authored-assertion surface; no runtime labelling.
- Procedural content generation beyond `region_plan` (terrain) and cascade-driven naming — narrative generation, dungeon layout, quest chains procedurally derived from cascades are not DSL-authored.
