# Social Systems — Strict Reframing (v2)

## Methodology

Re-classify the 35 "social / agent-inner / personality" systems against a strict rubric that assumes:

1. **NPCs are agents.** Anything an NPC could plausibly *choose to do* (form a bond, swear an oath, accept a mentor, propose marriage, avenge a friend, hide a secret, demand a bounty) should be an **agentic action** entered into the `action_eval` decision pipeline via a DSL-defined action vocabulary, not a global system that scans the entity list and mutates state.
2. **State is reconstructable from events.** If a field is a pure function of `(memory.events, entity state, time since last update)`, it is not state — it's a **derived view**. The system that writes it can be deleted; the read sites switch to the query.
3. **Physics is the residual.** What remains is the minimum that has to directly mutate NPC fields: memory event emission, the decision engine that picks actions, and a small number of path-dependent structural writes (title-name mutation, bounty gold transfers) that cannot be rewound.

Four outcomes:

- **ESSENTIAL** — direct mutation that can't be derived from events and isn't itself an NPC choice.
- **EMERGENT** — replaceable by an NPC agentic action plus its event. The action goes into the DSL action space; the event goes into `memory.events` / `world_events`; downstream systems read the event log.
- **DERIVABLE-VIEW** — pure query over existing state/events. No storage, no per-tick system; evaluate on demand.
- **DEAD/STUB** — empty body, not dispatched, or superseded. Delete.

v1 classifications are carried forward as "Old"; the new label may disagree.

All paths are absolute under `/home/ricky/Projects/game/.worktrees/world-sim-bench/`.

---

## Per-system reanalysis

### agent_inner
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (reduced) + DERIVABLE-VIEW (majority)
- **Reframe:** `agent_inner.rs` (1563 lines) is the *one* place that emits `MemoryEvent`s into `npc.memory.events` via `record_npc_event` (line 1537). That emission IS the event-sourcing primitive — it's the bookkeeping write that makes everything else reconstructable. Note how `record_npc_event` itself (lines 1553-1562) already *couples* event recording with emotion writes — `joy+=impact×0.5, pride+=impact×0.3` on positive events and `fear/grief/anger` spikes on negative ones. Under the strict rubric even that coupling is wrong: the function should record the event and return; the emotion field should be a view (`emotions(npc) = fold_over_recent_events`). What's **not** essential is the 1400+ lines of derived-state math piled on top: emotion spikes on needs (`spike_emotions_from_needs`), belief decay (0.95/tick, line 28), personality drift (`drift_personality_from_memory` at line 862), aspiration recomputation every 500 ticks, coworker social graph maintenance (lines 650–688), price-belief decay, context-tag accumulation from emotions across the 9 emotion-tag blocks. All of these are `f(memory.events, needs, time_since_last_tick)` and should be evaluated lazily as views, not churned every 10 ticks over every NPC. The only things that must remain imperative: (1) `record_npc_event` on externally-observed world events (deaths, attacks, conquests, trades) — but as a pure appender, (2) `npc.needs.*` drift (physics: hunger/shelter depletion is actual time-integrated entropy, not a function of events), (3) perception-driven `npc.known_resources` insertion via the resource_grid lookup (observation itself is an event and the insertion IS the event-handler side effect).
- **Required NPC actions:** none — this *is* the event bus, not an action.
- **Required derived views:** `emotions(npc) = f(recent memory.events)`, `mood(npc) = f(needs, emotions)`, `beliefs(npc) = fold(memory.events → belief deltas, decay=time_since)`, `personality_drift(npc) = integrate(memory.events × trait_kernels)`, `aspiration(npc) = latest unmet-need projection`, `coworker_familiarity(a,b) = co_shifts_count / decay`, `relationship_trust(a,b)`, `action_outcomes_ema(npc, action_kind)`.
- **Required event types:** `Witnessed { observer, event_id, impact, tick }` (central emission), `NeedsTicked { npc, Δhunger, Δshelter, Δsocial, tick }`, `ResourceObserved { observer, resource_id, kind, pos, tick }`, `FriendDied { observer, friend_id, tick }`, `WasAttacked { observer, attacker, tick }` — most of these already exist as `MemEventType`, so the rewrite is really a renaming + view extraction.
- **Summary:** Keep only the event-recorder core + raw physics needs; migrate all derived fields (emotions, beliefs, personality drift, aspiration, relationship trust) to lazy-evaluated views over `memory.events`. Estimated reduction: 1563 → ~300 lines of essential mutation.

### action_eval
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (the agentic decision engine) + DERIVABLE-VIEW (its scoring inputs)
- **Reframe:** `action_eval.rs` (1215 lines, ~7.9% of tick — heaviest per-tick system in this batch). `evaluate_and_act` (line 202) runs every 5 ticks; the two-phase pattern (score → defer → execute) with typed snap-grids (line 40: `TypedSnapGrids { resources, buildings, combatants }`) is the argmax-over-candidates implementation. This is *the* function that chooses which DSL action each NPC performs — it's irreducibly essential, it's the top of the agentic pipeline, and every EMERGENT reclassification in this document assumes this engine will grow additional candidate actions for the social/political verbs listed below. However, the internal scoring is pure derivation: `utility = f(need_urgency × personality × aspiration.need_vector × cultural_bias × action_outcomes_EMA × distance)`. The utility function is a query, not a stateful system. In the strict rubric, *decision dispatch = ESSENTIAL; scoring rules = DERIVABLE*. Candidate actions and scoring kernels should be declared in the DSL (one declaration per action type with conditions, targets, and utility expression); the engine is just `argmax(enumerated_candidates)`. A secondary observation: `npc.action_outcomes` (EMA state currently persisted on NpcData) is also derivable from the `ActionChosen` event log — but for hot-path reasons it's probably worth memoizing. The cleanup is orthogonal to the rubric.
- **Required NPC actions:** the entire emergent action vocabulary (see Required Action Vocabulary section). Currently ~13 `CandidateAction` variants (Eat, Harvest, Attack, Flee, Build, Work, Move, Idle, …); needs expansion to cover social/political actions (`SwearOath`, `Train`, `MarkWanted`, `ProposeMarriage`, `SignDemonicPact`, `Court`, `BegPardon`, `PostBounty`, `JoinReligion`, `BeginPlot`, `ConsumePotion`, `PracticeHobby`, …). Each new action is one DSL declaration, not a new Rust file.
- **Required derived views:** `utility_score(npc, candidate) = f(needs, personality, aspiration, emotions, cultural_bias, action_outcomes, distance)`, `candidate_targets(npc, action_type, grid)`, `action_outcomes_ema(npc, action_kind) = fold(ActionChosen events where npc matches, decay)`.
- **Required event types:** `ActionChosen { npc, action, target, score, tick }` — so the scoring of previous choices becomes reconstructable for the EMA without storing `action_outcomes` on the NPC. `ActionFailed { npc, action, reason, tick }` — for negative-outcome EMA updates.
- **Summary:** Decision dispatch is irreducible; the utility math and action-outcome EMA are derivable from the `ActionChosen` event history; candidate action set grows to absorb the 12 EMERGENT systems below.

### action_sync
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `action_sync.rs` (113 lines) writes `npc.action` as a view over `(work_state, goal_stack.current, move_target, current_intention)`. This is a literal "compute a field from fields" system with no external triggers and no event side-effects. Delete it, replace with `fn npc_action(&npc, &entity) -> NpcAction` called at the read sites. Persisting it was a rendering convenience, not simulation state.
- **Required derived views:** `npc_action(npc, entity) -> NpcAction`.
- **Required event types:** none.
- **Summary:** Pure projection; inline as a method.

### moods
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** DERIVABLE-VIEW
- **Reframe:** `moods.rs` (232 lines). Mood as a categorical state is never stored on NpcData. Every invocation re-derives a pseudo-mood from `hp_ratio` + `fidelity_zone.fidelity` and emits `Morale ±0.5..3.0` deltas across three branches (low-HP-in-combat, high-HP-in-combat, idle-high-HP). That is textbook "view with side effects." The morale nudges are redundant with agent_inner's emotion spikes (which already run every 10 ticks). The contagion phase at lines 141–153 is marked "deterministic roll" in comments and is unimplemented. Strict rubric: the entire compute body deletes. What's valuable in this file is the infrastructure: the `Mood` enum, the `MoodCause` enum, `MoodSnapshot`, and the ~7 query helpers (`mood_combat_multiplier`, `is_reckless`, etc.) — all of which *consume* a derived mood. Those stay, but the view they consume becomes `fn mood(npc) -> Mood` defined in terms of `emotions(npc)`, `needs(npc)`, and `hp_ratio`. Morale itself becomes a view: `morale(npc) = baseline + Σ recent_event_morale_impacts × decay`.
- **Required NPC actions:** none.
- **Required derived views:** `mood(npc) -> Mood`, `morale(npc) -> f32`, `mood_combat_multiplier(mood)`, `is_reckless(mood)`, `mood_snapshot(npc) -> MoodSnapshot`.
- **Required event types:** none (consumes existing memory.events + needs).
- **Summary:** Pure function of NPC state; delete the compute body, keep the enum + query helpers as the view API.

### fears
- **Old:** STUB/DEAD + EMERGENT-CANDIDATE
- **New:** DEAD
- **Reframe:** `fears.rs` (140 lines). Phase 1 (acquisition) body is empty; Phase 4 (mentorship) is `let _ = npc_ids`; Phase 5 (contagion) is empty. Phase 2 is a 2-branch HP-ratio morale drain that exactly duplicates agent_inner's fear spike. NpcData has a `fears: Vec<u8>` field that this system never touches. Delete. If phobia acquisition ever gets implemented, it belongs as a reactive `memory.events → BeliefType::Phobia(source)` derivation, not a scanning system.
- **Required NPC actions:** `FleeFromPhobicSource` (already covered by `Flee`).
- **Required derived views:** `phobias(npc) = f(memory.events matching traumatic_attack)`.
- **Required event types:** none (reuses `WasAttacked`).
- **Summary:** Delete; phobia acquisition is a belief derivation from memory.

### personal_goals
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `personal_goals.rs` (184 lines). Fires a chronicle + morale bump when `behavior_profile[tag]` crosses a narrow threshold window. The *goal itself* is an NPC's choice of aspiration — that's agentic. In the new model, NPC picks `PursueGoal(kind)` as an action (or `action_eval` enumerates goal-pursuit candidates from `aspiration.need_vector`); reaching the threshold emits a `GoalAchieved { npc, kind, tier }` event; the chronicle and title systems derive pride/fame from the event. No per-tick scan needed.
- **Required NPC actions:** `PursueGoal(kind)`, `DeclareMastery(domain)`.
- **Required derived views:** `goal_progress(npc, kind) = npc.behavior_value(kind.tag) / kind.threshold`.
- **Required event types:** `GoalAchieved { npc, kind, tier, tick }`.
- **Summary:** Goal pursuit is agentic, completion is a threshold event; scan-and-fire body deletable.

### hobbies
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `hobbies.rs` (180 lines). Idle NPCs drift toward adjacent skills via 0.1–0.3 behavior-tag nudges every 30 ticks. "Pursuing a hobby" is *definitionally* an NPC choice: when morale > 30 and economic_intent is Produce/Idle, the NPC selects a `PracticeHobby(domain)` action. The domain is `argmax(mining,trade,combat,faith)` — a derived pick, fine. The effect (tag accumulation) becomes a normal action-outcome event.
- **Required NPC actions:** `PracticeHobby(domain)`.
- **Required derived views:** `hobby_domain(npc) = argmax_domain(behavior_profile)`.
- **Required event types:** `HobbyPracticed { npc, domain, tag_gain, tick }`.
- **Summary:** NPCs *do* hobbies; becomes a low-priority candidate action in the utility scorer.

### addiction
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT (via `ConsumePotion` action) + DERIVABLE-VIEW (withdrawal rule)
- **Reframe:** `addiction.rs` (94 lines) uses `status_effects` debuff count as a proxy for addiction state and applies `Damage{0.5|1.5} + Slow(0.7, 3s)`. The addictive *act* (consuming something that leaves a Debuff) is an NPC choice — model it as `ConsumePotion(kind)` in the DSL action space. The withdrawal damage is a status-effect-driven rule, not a standalone system: "if entity.has_debuff(Withdrawal) and hp_ratio<0.5 → Damage(1.5)" lives in the status-effect evaluator alongside all other DoTs.
- **Required NPC actions:** `ConsumePotion(kind)`, `ConsumeIntoxicant(kind)`.
- **Required derived views:** `is_addicted(npc) = count_recent(ConsumePotion) > N within window`.
- **Required event types:** `PotionConsumed { npc, kind, tick }`, `WithdrawalTick { npc, tick }`.
- **Summary:** Consumption is agentic; withdrawal is a status-effect rule; delete system.

### npc_relationships
- **Old:** DUPLICATIVE
- **New:** DERIVABLE-VIEW
- **Reframe:** `npc_relationships.rs` (241 lines). Emits `UpdateEntityField{Morale,+0.5}` (friends) and `AddBehaviorTags{COMBAT+0.2}` (rivals) based on behavior-profile cosine similarity. Pure secondary heuristic, never writes to `npc.relationships`. Agent_inner already maintains the primary relationship graph via event-driven updates (coworker ticks, trade partners, theory-of-mind `observe_action`). Replace with: `relationship(a,b) = f(shared_events_in_memory between a and b, decay)` and `rivalry(a,b) = behavior_tag_overlap(a,b) × combat_ratio`. Delete compute; keep zero cost.
- **Required NPC actions:** `BefriendNpc(target)`, `DeclareRival(target)` — but these are rare; most relationships are emergent from co-experience, which is already `memory.events`.
- **Required derived views:** `relationship(a,b)`, `rivalry(a,b)`, `top_friend(npc)`.
- **Required event types:** none (reuses `MadeNewFriend`, `TradedWith` in memory.events).
- **Summary:** Relationships are a view over shared memory events.

### npc_reputation
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** DERIVABLE-VIEW (for reputation) + DEAD (for the heal pulse)
- **Reframe:** `npc_reputation.rs` (74 lines) doesn't compute reputation at all — it emits a `Heal{5}` for every injured NPC at a settlement that has a healer (by `class_tags contains "healer"|"cleric"|"priest"`). That's a passive-aura effect that belongs to the healer's `PassiveEffects.aura_radius` (already on NpcData) or to a building-sourced Heal. Actual reputation should be a view: `reputation(npc) = Σ witnessed_memory_events involving npc, weighted by settlement observers`. Delete heal pulse; keep reputation as a view.
- **Required NPC actions:** none (healer's "tend wounded" is their active action).
- **Required derived views:** `reputation(npc) = Σ_witnessed_events_in_settlement weighted`, `has_healer(settlement)`.
- **Required event types:** `Witnessed { observer, subject, event_kind, tick }`.
- **Summary:** Reputation is a derived view; the heal pulse folds into the passive-aura evaluator.

### romance
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `romance.rs` (169 lines). Pair-wise within settlement (capped at 32 NPCs near Inn/Temple/Market/GuildHall), 15% pair roll + cosine-similarity gate (`AFFINITY_THRESHOLD=0.5`), emits `DIPLOMACY+0.3, RESILIENCE+0.2` tag nudges and a chronicle entry on strong bonds. No persistent romance state — no `romantic_partner_id`, no "in love" flag. The effect is the behavior-tag nudge, identical in pattern to `hobbies`. Replace with agentic actions: an NPC with high compatibility + proximity + morale>30 selects `Court(target)` as a candidate action when loitering at a social building; repeated successful `Court` actions emit `Courted { a, b, tick }` events; the derived view `romantic_interest(a,b) = decayed_count(Courted events between a and b)` drives further escalation (into `ProposeMarriage`). The pair-cap at 32 and the social-building proximity check both fall out naturally as gates on `Court`'s utility score. Marriage is a separate pair of actions (`ProposeMarriage` + `AcceptMarriage`). This also gives the system a missing mechanism: one-sided pursuit, rejection, courtship rivalry — all naturally emergent from two NPCs making independent decisions.
- **Required NPC actions:** `Court(target)`, `FlirtAt(target)` (lighter, non-committal variant), `RebuffAdvances(suitor)` (explicit rejection — raises suitor's grief/anger).
- **Required derived views:** `compatibility(a,b) = cosine(behavior_profile_a, behavior_profile_b)`, `romantic_interest(a,b) = decayed_count(Courted events)`, `social_building_nearby(npc) = any Inn|Temple|Market|GuildHall within R`.
- **Required event types:** `Courted { a, b, tick }`, `CourtshipRebuffed { suitor, target, tick }`.
- **Summary:** Courting is a choice; emergent drift becomes a candidate action at social buildings; pair iteration disappears.

### rivalries
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** `rivalries.rs` (107 lines). All phases commented out. Query helpers return hardcoded zeros. Rivalry is fully derivable from `relationship(a,b) = f(memory.events where both present, weighted negative by combat/conflict events)`. Delete the compute body; if a `DeclareRival` action is ever added, it emits `RivalryDeclared { a, b, reason }`.
- **Required NPC actions:** `DeclareRival(target)` (optional — mostly emergent).
- **Required derived views:** `rivalry(a,b)`.
- **Required event types:** none (already covered by memory).
- **Summary:** Delete; rivalries are a view.

### companions
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** DERIVABLE-VIEW
- **Reframe:** `companions.rs` (115 lines). Emits `+0..+1.5 morale` scaled by `min(3, friendly_neighbors) × 0.5`. No companion state. This is a transparent ambient-morale function: `morale += ambient_social_boost(grid_friendly_count)`. Fold into the `mood(npc)` derivation; no system needed.
- **Required NPC actions:** none.
- **Required derived views:** `ambient_social_boost(grid, npc) = min(3, friendly_count) × 0.5`.
- **Required event types:** none.
- **Summary:** Pure grid-count lookup; moves into `mood()` view.

### party_chemistry
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `party_chemistry.rs` (75 lines). `pop≥2 → always-on +5% buff`. Zero chemistry tracked. This is a settlement property, not an NPC one: `settlement.ambient_buff = if pop>=2 { 1.05 } else { 1.0 }`. Evaluate at damage/stat read sites.
- **Required NPC actions:** none.
- **Required derived views:** `settlement_ambient_buff(settlement)`.
- **Required event types:** none.
- **Summary:** Replace with a settlement-level passive multiplier read at the combat math.

### nicknames
- **Old:** STUB/DEAD
- **New:** DEAD (replaced by DERIVABLE-VIEW)
- **Reframe:** `nicknames.rs` (93 lines). All granting code commented; no `GrantNickname` delta exists. Replace with `nickname(npc) = top_tag_name(npc.behavior_profile)` as a query. If nicknames need to persist (titles-style), bake into `titles` as a separate tier.
- **Required NPC actions:** none.
- **Required derived views:** `nickname(npc) = top_tag_name(behavior_profile)`.
- **Required event types:** none.
- **Summary:** Delete system; nickname is a view.

### legendary_deeds
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `legendary_deeds.rs` (103 lines). Fires chronicle + morale bump on 15 behavior-profile thresholds × 2 tiers. No state persists; `deeds: Vec<u8>` on NpcData is never written. Collapse to: `is_legendary(npc) = any(npc.behavior_value(tag) >= deed_table[tag].threshold)`, and `recent_legendary_crossings(npc, window) = deed_table × behavior_history in window`. Chronicle generation becomes a subscriber to the general `BehaviorThresholdCrossed` event emitted once when `behavior_value` crosses a band.
- **Required NPC actions:** none (thresholds crossed as side-effect of action outcomes).
- **Required derived views:** `is_legendary(npc)`, `legendary_tier(npc, tag)`, `chronicle_mentions(npc)`.
- **Required event types:** `BehaviorThresholdCrossed { npc, tag, tier, tick }`.
- **Summary:** Legendary status is a view over the behavior profile; scan-and-fire deletes.

### folk_hero
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** `folk_hero.rs` (101 lines). Body empty. Fame/folk-hero status is cleanly derivable: `fame(npc) = Σ chronicle_mentions(npc) + Σ witnessed_positive_events`. If settlements should ever *choose* to elevate a local hero, it's an agentic action (`CrownFolkHero(target)` taken by a settlement leader NPC). Delete compute body.
- **Required NPC actions:** `CrownFolkHero(target)` (leader-only).
- **Required derived views:** `fame(npc)`, `folk_hero_of(settlement) = argmax_fame(residents)`.
- **Required event types:** `FolkHeroCrowned { npc, settlement, tick }`.
- **Summary:** Delete; fame is a view, crowning is an action.

### trophies
- **Old:** STUB/DEAD
- **New:** DEAD (or EMERGENT if revived)
- **Reframe:** `trophies.rs` (84 lines). All bodies empty. If trophies are ever implemented, the acquisition is agentic: NPC loots a `Trophy { source_kill_id }` as part of an `ClaimTrophy(corpse)` action; the passive bonuses are a status-effect rule keyed off inventory contents. Delete compute.
- **Required NPC actions:** `ClaimTrophy(corpse)`.
- **Required derived views:** `trophy_bonus(npc) = Σ trophy_effects(inventory)`.
- **Required event types:** `TrophyClaimed { npc, kind, source, tick }`.
- **Summary:** Delete; trophy claim is an action, passive bonus is a view.

### mentorship
- **Old:** ESSENTIAL (partial)
- **New:** EMERGENT
- **Reframe:** `mentorship.rs` (176 lines). Currently `compute_mentorship` scans per-settlement, finds Library/Workshop/GuildHall at `construction_progress≥1.0` (lines 44–57), pre-collects up to 128 NPCs near one (within 15 units squared = 225, lines 77–81), skips NPCs in High-fidelity grid (combat), insertion-sorts by level descending (lines 94–100), and pairs mentors (`mentor.level ≥ apprentice.level + 3`) with up to 2 apprentices each. Emits: `Heal{xp×0.05}` to apprentice, `AttackDamage+0.1` base stat gain, staggered `Hp+10` level-up, `AddBehaviorTags{TEACHING+2,LEADERSHIP+1}` to mentor and `DISCIPLINE+1` to apprentice. These stat gains *feel* essential only because a system has to issue them — but the natural model is: apprentice chooses `Train(teacher=mentor_id, building=X)` as an action; mentor chooses `Teach(student=app_id)`; the successful pairing emits `TrainingSession { mentor, student, skill_gain, stat_deltas, tick }`; stat deltas are *the effect of the action* (same way harvesting is the effect of `Harvest` — the inventory write is essential, but it happens as the action's consequence, not from a global scan). No scan-all-pairs system needed — both sides opt in. Bonus: this removes the arbitrary `mentor_used ≤ 2` quota (replaced by natural contention for the mentor's time slot) and the fidelity-grid skip (apprentice-initiated: if you're in combat, you can't choose Train).
- **Required NPC actions:** `Train(teacher, building)` (apprentice), `Teach(student)` (mentor — can be implicit matching), `OfferApprenticeship(target)` (mentor-initiated recruitment).
- **Required derived views:** `eligible_mentors(npc, settlement) = residents.level >= npc.level + 3 near knowledge_building`, `xp_gain(mentor_level, apprentice_level) = BASE_MENTOR_XP × (1 + mentor_level/20)`, `stat_gain_per_session(apprentice_level)`.
- **Required event types:** `TrainingSession { mentor, student, xp_gain, stat_deltas, tick }`, `LevelUp { npc, tier, tick }`, `ApprenticeshipFormed { mentor, student, tick }`.
- **Summary:** Training is a mutual action at a knowledge building; delete the scanning compute; the stat-delta writes still happen but as the action's handler, not a pass.

### marriages
- **Old:** STUB/DEAD
- **New:** DEAD — replaced by two EMERGENT actions
- **Reframe:** `marriages.rs` (82 lines). Entire body commented. The actual `spouse_id`/`children` link is set by `family.rs` (not in this batch). In the strict model, marriage formation is unambiguously agentic: `ProposeMarriage(target)` + `AcceptMarriage(suitor)`; the formal link write (`spouse_id = partner_id`) happens on accept and emits `MarriageFormed { a, b, tick }`. Children are a separate biological process. Delete this file entirely.
- **Required NPC actions:** `ProposeMarriage(target)`, `AcceptMarriage(suitor)`, `RefuseMarriage(suitor)`, `AnnulMarriage(spouse)`.
- **Required derived views:** `marriage_eligibility(a,b)`, `is_married(npc)`.
- **Required event types:** `MarriageProposed { a, b, tick }`, `MarriageFormed { a, b, tick }`, `MarriageAnnulled { a, b, tick }`.
- **Summary:** Delete; replaced by two agentic actions with a formal link write on accept.

### bonds
- **Old:** STUB/DEAD + DUPLICATIVE
- **New:** DEAD (compute) + DERIVABLE-VIEW (helpers)
- **Reframe:** `bonds.rs` (166 lines). Not dispatched. The compute body iterates O(n²) pairs and does nothing (`let _ = (a,b);`). The query helpers (`bond_strength`, `morale_bonus`, `combat_power_multiplier`, `has_battle_brothers`, `average_group_bond`) over `state.adventurer_bonds: HashMap<(u32,u32), f32>` are read by other systems — keep them, but redefine `bond_strength` as a view over shared `memory.events` (shared-battles count, co-survived count) and drop the `adventurer_bonds` HashMap. Bond growth is then automatically event-sourced.
- **Required NPC actions:** none (bonds form from shared combat experience, which is already memory events).
- **Required derived views:** `bond_strength(a,b) = f(shared_battle_events, shared_survival_events, decay)`, `has_battle_brothers(npc)`.
- **Required event types:** none (reuses `WasInBattle`, `WonFight` memory events).
- **Summary:** Delete compute; rewrite helpers as views over memory.events.

### memorials
- **Old:** DUPLICATIVE (not dispatched)
- **New:** DEAD
- **Reframe:** `memorials.rs` (118 lines). Not dispatched (already commented out in mod.rs: "redundant (grief morale handled by agent_inner)"). Grief on death is precisely `f(memory.events, FriendDied records)`. Delete. If commemorative monuments ever exist, they're buildings constructed via a `BuildMemorial(target_corpse)` action.
- **Required NPC actions:** `BuildMemorial(target)` (optional).
- **Required derived views:** `grief(npc)` already in the emotion view.
- **Required event types:** `MemorialBuilt { builder, target, tick }` (optional).
- **Summary:** Delete; grief is a view, memorials are buildings.

### journals
- **Old:** STUB/DEAD (not dispatched)
- **New:** DEAD
- **Reframe:** `journals.rs` (86 lines). Not dispatched. Body is empty conditionals. Comment in mod.rs says it all: "journal state not on entities, all logic in agent_inner memory." `memory.events` IS the journal. Delete.
- **Required NPC actions:** `WriteJournal(content)` if exportable narrative is ever wanted (purely cosmetic side-effect of reflection).
- **Required derived views:** `journal(npc) = format_memory_events(npc)`.
- **Required event types:** none (memory.events already serve).
- **Summary:** Delete; memory.events IS the journal.

### cultural_identity
- **Old:** ESSENTIAL (context_tags) + EMERGENT (reinforcement)
- **New:** DERIVABLE-VIEW + EMERGENT (adoption)
- **Reframe:** `cultural_identity.rs` (162 lines). `advance_cultural_identity` (line 53) aggregates resident behavior profiles every 500 ticks and mutates `settlement.context_tags` to hold the dominant culture tag out of 8 candidate clusters (Warrior/Scholar/Merchant/Artisan/Farming/Survivor/Seafaring/Storytelling — lines 90–98). Then reinforces: `+0.5` bonus tag to each resident (lines 121–131) and pushes a chronicle entry on first solidification (lines 152–159). The aggregation itself is a pure function: `settlement_culture(sid) = argmax_culture(Σ behavior_profiles of residents / npc_count)` with gates score>1.0 and residents≥5 (lines 80, 105). Compute lazily at read time — no need to write `context_tags` ahead of consumers. The reinforcement step is conceptually *adoption* — an NPC aligning with their local culture because they keep acting in culture-aligned ways. That's not a mandatory broadcast from a ruler; it's the NPC's own action outcomes reinforcing their profile. If an NPC chooses `Train` at a Library in a Scholar settlement, their RESEARCH tag naturally grows — you don't need a separate "culture gives you +0.5" pass. Delete the reinforcement loop; accept that culture is a view over what NPCs have actually been doing. Emit `CultureEmerged { settlement, culture, first_tick }` once per culture-transition (handled by a subscriber to behavior-aggregate changes), and `CultureAdopted { npc, culture, tick }` when an NPC's cultural_alignment view crosses a threshold — for chronicle/narrative purposes, not for mechanical effects.
- **Required NPC actions:** `InternalizeCulturalNorm(culture)` — implicit, emerges from other action outcomes in the culture's tag bundle.
- **Required derived views:** `settlement_culture(settlement)`, `cultural_alignment(npc) = cosine(npc.behavior_profile, settlement_culture(npc.home))`, `culture_score(settlement, culture_type)`.
- **Required event types:** `CultureEmerged { settlement, culture, first_tick }`, `CultureAdopted { npc, culture, tick }`, `CultureShifted { settlement, from, to, tick }`.
- **Summary:** Culture detection is a view over aggregate resident behavior; reinforcement deletes (it was double-counting action effects); two events for chronicle transitions.

### titles
- **Old:** ESSENTIAL (name mutation)
- **New:** DERIVABLE-VIEW (with rare EMERGENT formalization)
- **Reframe:** `titles.rs` (105 lines) mutates `npc.name` permanently via `format!("{} {}", old_name, title)` at line 43 — which felt essential because once prepended, `" the Oathkeeper"` is stuck. But: the *decision* in `determine_title` (lines 48–105) is a pure function of `(fulfilled_oaths, spouse_alive, friend_deaths, attacks_survived, hp_ratio, gold, trade_tag_value, starvation_count, children.len)`. The priority order (Oathkeeper → Avenger → Bereaved → Unbroken → Merchant Prince → Enduring → Patriarch) is fixed logic. Evaluate `title(npc)` lazily. Render `"{npc.base_name} {title(npc)}"` at display time — no persistent mutation. The only reason to *persist* a title is if it was formally bestowed (coronation, knighthood), which is unambiguously an agentic act by a ruler: `BestowTitle(target, title)` emits `TitleBestowed { grantor, grantee, title, tick }` and the title becomes immutable afterward. Informal titles (Bereaved, Unbroken, Patriarch) are derivations; formal titles are events.
- **Required NPC actions:** `BestowTitle(target, title)` (ruler only), `RevokeBestowal(target)` (ruler retracts a formal title — rare).
- **Required derived views:** `title(npc)`, `has_formal_title(npc)`, `display_name(npc) = "{base_name} {title(npc)}"`.
- **Required event types:** `TitleBestowed { grantor, grantee, title, tick }`, `TitleRevoked { grantor, grantee, title, tick }`.
- **Summary:** Informal titles are a view over state + oath log; formal titles are an agentic bestowal event; `npc.name` mutation is deleted in favor of `display_name(npc)`.

### oaths
- **Old:** ESSENTIAL
- **New:** EMERGENT
- **Reframe:** `oaths.rs` (159 lines). Currently `advance_oaths` (line 35) runs three phases every 200 ticks: Phase 1 (lines 42–83) rolls a 5% swear chance per NPC across 3 kinds (Loyalty/Vengeance/Protection), Phase 2 (lines 86–130) checks fulfillment via kind-specific conditions (vengeance = target dead; protection = 1000 ticks elapsed; loyalty = 1500 ticks AND still at home settlement), Phase 3 (lines 133–153) marks hostile Loyalty-oath holders as broken and bumps DECEPTION tags. Swearing an oath is the canonical agentic act: a random 5% roll is a placeholder for "NPC chose to swear." Replace: `SwearOath(kind, target)` is an NPC action chosen when `memory.events` + emotion state warrant it (grudge-belief present → Vengeance; high Faith+Discipline + home settlement → Loyalty; settlement under threat → Protection). Fulfillment and breaking are pure derivations: `oath_fulfilled(oath, state) = condition_met(oath.kind, state)`, `oath_broken(oath, state) = team_changed_after_swearing || abandoned_home_settlement`. The `oaths: Vec<Oath>` field becomes an append-only log (structural commitment persists); fulfillment/broken *flags* become views over `(oath, current state)` — no need to mutate `.fulfilled=true` on a polling pass; just evaluate whenever someone asks. Chronicle "3+ fulfilled oaths → Oathkeeper" is handled by `titles.rs`.
- **Required NPC actions:** `SwearOath(kind, target_id)`, `BreakOath(oath_idx)` (explicit betrayal — distinct from derived "broken by team-switch"), `AttemptFulfillOath(oath_idx)` (for Vengeance: actively pursue target).
- **Required derived views:** `fulfilled_oaths(npc)`, `active_oaths(npc)`, `broken_oaths(npc)`, `vengeance_target(npc) = first BeliefType::Grudge`, `oath_pride_bonus(npc) = 0.6 × fulfilled_count`.
- **Required event types:** `OathSworn { swearer, kind, target, terms, tick }`, `OathFulfilled { oath_id, tick }` (fired once when the fulfillment-view transitions to true), `OathBroken { oath_id, tick }`.
- **Summary:** Swearing is an agentic action; the Oath struct is immutable after swearing; fulfillment/break are views that emit one-shot events on transition.

### grudges
- **Old:** STUB/DEAD
- **New:** DEAD (compute) + DERIVABLE-VIEW (status)
- **Reframe:** `grudges.rs` (105 lines). Body all commented; no grudge struct in state. Query helpers (`grudge_combat_bonus`, `acts_recklessly`) currently return hardcoded defaults. Grudges already live as `BeliefType::Grudge(target_id)` in `npc.memory.beliefs`, which is populated by agent_inner's belief-formation logic from memory events (a sequence of `WasAttacked` or `FriendDied` entries mentioning the same entity forms a Grudge belief). oaths.rs reads this belief for Vengeance target selection. A grudge is literally "I remember entity X harmed my friend/me" — exactly what `memory.beliefs.filter(BeliefType::Grudge)` returns. The grudge strength is `count(negative memory events mentioning target) × decay`. There is no separate grudge state to maintain. Delete the compute body; rewrite the two query helpers as views over beliefs/memory. `Forgive(target)` is the only agentic action that needs to exist — an NPC explicitly retiring a grudge belief after reconciliation — but this is optional; most grudges naturally fade via the existing 0.95/tick belief decay.
- **Required NPC actions:** `Forgive(target)` (optional — explicitly retires a grudge belief).
- **Required derived views:** `grudges(npc) = memory.beliefs.filter(Grudge)`, `grudge_intensity(npc, target) = count_negative_memory_events(npc, target) × decay`, `grudge_combat_bonus(npc, target) = grudge_intensity × scale`, `acts_recklessly(npc) = max_grudge_intensity(npc) > threshold`.
- **Required event types:** `GrudgeFormed { holder, target, cause_event_id, tick }` (derived — fires when belief crystallizes from accumulated memory events), `GrudgeResolved { holder, target, tick }` (fires on Forgive action or natural belief decay below threshold).
- **Summary:** Grudges are beliefs are memory derivations; delete system; keep helpers as views.

### secrets
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT + DERIVABLE-VIEW
- **Reframe:** `secrets.rs` (217 lines). Six secret types × 2 no-ops, keyed by `entity.id % 8` (deterministic at spawn), firing once per threshold crossing within a narrow window. Writes: `Morale ±5..±15` on reveal, `AddBehaviorTags` related-skill boost, `UpdateTreasury{-stolen}` for spy-type reveals, and a chronicle entry. Two separate concerns need unbundling: (1) the *existence* of a secret is fixed at birth — better to emit `SecretAssigned { npc, kind, tick }` at character creation (via birth-time subscriber to `NpcSpawned`) and evaluate `has_secret(npc, kind)` by querying that event log, rather than recomputing `id % 8` on every pass. This retains determinism while making the secret first-class. (2) The *reveal condition* (behavior-tag sum crossing a threshold window) is a pure derivation from action history — subscribe to `BehaviorThresholdCrossed` and emit `SecretRevealed { npc, kind, observer, tick }`; the morale shift and tag boost happen as the event handler's side-effects, not from a scanning pass. (3) The spy-type treasury drain IS a real conserved economic transfer — that fires a `TransferGold(settlement → 0, stolen_amount)` in the reveal handler. The system compute body deletes; the reveal handler is a handful of lines in the event subscriber layer. Alternative model: NPCs could *choose* to reveal a secret via an explicit `ConfessSecret` action (e.g. under stress or in confession to a priest), giving the system dual pathways — forced reveal via threshold, voluntary reveal via action.
- **Required NPC actions:** `RevealSecret(target)` (deliberate reveal to a specific observer), `HideSecret` (active concealment — raises stress), `ConfessSecret` (voluntary unburdening, usually to priest or friend).
- **Required derived views:** `has_secret(npc, kind) = any SecretAssigned events for npc`, `reveal_risk(npc) = f(stress, behavior_crossings_near_threshold)`, `known_secrets(observer) = SecretRevealed events where observer matches`.
- **Required event types:** `SecretAssigned { npc, kind, tick }` (birth-time), `SecretRevealed { npc, kind, observer, tick }`, `SecretKept { npc, kind, stress_delta, tick }` (fires on near-miss threshold for stress feedback).
- **Summary:** Secret assignment is a birth-time event; reveal is either an action or a derived threshold event; compute body deletes.

### intrigue
- **Old:** STUB/DEAD
- **New:** DEAD (to be rebuilt as EMERGENT)
- **Reframe:** `intrigue.rs` (63 lines). Body commented. Real intrigue — plots, alliances, betrayals — is *entirely* agentic: `BeginPlot(goal, targets)`, `RecruitConspirator(target)`, `BetrayFriend(target)`, `InformOn(conspirator)`. No scanning system can capture it; the whole mechanic is NPC decisions. Delete skeleton; when rebuilt, it's a tree of agentic actions with `Plot` as an aggregate entity.
- **Required NPC actions:** `BeginPlot(goal, targets)`, `RecruitConspirator(target)`, `ExecutePlot(plot_id)`, `InformOn(conspirator)`, `BetrayFriend(target)`.
- **Required derived views:** `active_plots(region)`, `conspirator_count(plot)`, `plot_readiness(plot)`.
- **Required event types:** `PlotBegun`, `PlotJoined`, `PlotExecuted`, `PlotBetrayed`, `ConspiratorInformed`.
- **Summary:** Delete; intrigue is a constellation of agentic actions, not a system.

### religion
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** EMERGENT (joining a faith) + DERIVABLE-VIEW (heal aura)
- **Reframe:** `religion.rs` (69 lines). Heals injured NPCs and adds `FAITH+1, RITUAL+0.5` at settlements with treasury≥50 (proxying for "temple active"). The *faith-follower* status is agentic: `JoinReligion(faith)`, `PerformRitual(faith)`, `LeaveReligion`. Heal-aura should come from Temple building's `PassiveEffects.aura_radius`, not a treasury proxy. Delete the system; attach effects to the Temple entity.
- **Required NPC actions:** `JoinReligion(faith)`, `PerformRitual(faith, target)`, `LeaveReligion`.
- **Required derived views:** `religion_of(npc)`, `faithful_count(faith, settlement)`.
- **Required event types:** `ReligionJoined { npc, faith, tick }`, `RitualPerformed { priest, faith, target, tick }`.
- **Summary:** Religion adherence is an action; heal aura is a building passive; delete the treasury-proxy compute.

### demonic_pacts
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `demonic_pacts.rs` (114 lines). Uses `status_effects` debuff count as a proxy for pact tier. Three tiers of consequence: 1+ debuff → `Damage{1.0}`, 2+ debuffs → `TransferGold{10.0}` pact-holder-to-home-settlement (tithe to a demonic patron via a regular-settlement account — economically coherent because the demon is presumably a settlement-aligned cultist), 3+ debuffs → 15% roll for `Damage{15.0}` to a random ally (possession). The proxy-via-status-count pattern is the tell: the system lacks a `Pact` entity, so it's reading whatever's handy. Fix properly: `SignDemonicPact(demon, terms)` is an NPC action chosen when desperate (low-hp, starving, grieving — high need urgency + negative emotions); emits `PactSigned { npc, demon, clauses, tick }` which writes a `Pact` log entry; each clause then runs as its own declared status-effect rule in the status-rule evaluator (e.g. "if PactTier ≥ 2, emit TransferGold(npc → demon_settlement, 10) every 7 ticks"). Breaking the pact (`BreakPact`) is an action that emits `PactBroken` and strips the clauses. The system compute body deletes; the clauses become declarative status rules; all three mechanical effects happen as rule consequences, not from a scan.
- **Required NPC actions:** `SignDemonicPact(demon, terms)`, `BreakPact(demon)`, `PerformBlackRitual(demon)` (maintenance / power-up).
- **Required derived views:** `pact_tier(npc) = count(active_pact_clauses(npc))`, `is_cursed(npc) = pact_tier(npc) > 0`, `pact_patrons(npc) = distinct demons in active pacts`.
- **Required event types:** `PactSigned { npc, demon, clauses, tick }`, `PactBroken { npc, demon, tick }`, `PactTithe { npc, amount, recipient, tick }`, `PossessionStrike { npc, victim, tick }`.
- **Summary:** Signing is an action; clauses are declarative status rules; the scan deletes.

### divine_favor
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** DERIVABLE-VIEW
- **Reframe:** `divine_favor.rs` (99 lines). Uses `settlement.treasury` buckets (>100 heal+treasury, <10 damage) — a "prosperity feedback." No divine_favor state. Collapse with `religion`, `npc_reputation`, and `reputation_stories` into a single settlement-level derivation: `settlement_divine_blessing(s) = f(treasury, faith_count, recent_rituals)` → small heal/damage/treasury deltas at read sites. No system needed.
- **Required NPC actions:** none.
- **Required derived views:** `settlement_divine_blessing(settlement)`, `settlement_divine_wrath(settlement)`.
- **Required event types:** `DivineFavorTicked { settlement, delta, tick }` (optional chronicle).
- **Summary:** Treasury proxy collapses into a shared settlement derivation; delete.

### biography
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW (explicit)
- **Reframe:** `biography.rs` (569 lines) is *already* a pure function returning a `String` — it's never called by the simulation, only by UI/CLI. It's not a system. Move out of `systems/` entirely into a `views/` or `queries/` module. Zero per-tick cost today; should have been zero DSL surface too.
- **Required NPC actions:** none.
- **Required derived views:** `biography(npc) -> String` — exactly what the file does.
- **Required event types:** none.
- **Summary:** Not a system; relocate to a query module.

### reputation_stories
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `reputation_stories.rs` (47 lines). 12 lines of logic: rich settlements get `treasury+=3`, poor get `treasury-=2`. That's a derived economic rule — settlement's treasury drift as a function of its own treasury. Not "stories," not reputation. Fold into the settlement economy step or delete as trivial.
- **Required NPC actions:** none.
- **Required derived views:** `treasury_momentum(settlement)`.
- **Required event types:** none.
- **Summary:** Two-branch drift rule; fold into economy or delete.

### wanted
- **Old:** ESSENTIAL (partial — bounty payouts + threat reduction)
- **New:** EMERGENT
- **Reframe:** `wanted.rs` (260 lines, `compute_wanted` at line 44). Three phases: (1) settlements with `threat_level ≥ MIN_THREAT_FOR_WANTED=0.15` (line 64) post `WorldEvent::Generic` bounty posters for nearby hostile monsters/faction NPCs within `WANTED_RANGE_SQ=900` (line 81); (2) when hostiles die near a grid, share bounty among killers via `TransferGold(settlement → collector)`; (3) reduce settlement threat by `dead_hostile_count × 0.02`. Under the strict rubric: posting a bounty is unambiguously the settlement leader's agentic choice — a global scan is a proxy for "every frustrated ruler auto-posts." Replace with `PostBounty(target, reward)` taken by a ruler NPC (or by a mayor/council member via council.rs); it emits `BountyPosted { target, reward, funder, tick }` which writes a row to a `WorldState.bounties` log. Paying out is a derivation from the `EntityDied` event matched against outstanding bounties — runs as an event subscriber on death, not a scanning pass. The `TransferGold` IS a real physics write (gold is conserved), so that stays — but it fires as a one-shot reducer inside the death-event handler, triggered by the event, not by a pass over entities. Threat reduction (`threat -= 0.02 × recent_hostile_deaths_near(settlement)`) is a trivial derivation — either lazily computed at read time or updated incrementally by the same death-event handler. Result: zero per-tick cost when there are no deaths; essential writes still happen, but triggered by events not cadence.
- **Required NPC actions:** `PostBounty(target, reward)` (ruler), `ClaimBounty(corpse, poster)` (killer — implicit from kill event but explicit for contested corpses).
- **Required derived views:** `active_bounties(settlement)`, `bounty_on(target) = max posted bounty`, `bounty_collectors_near(corpse, radius)`.
- **Required event types:** `BountyPosted { target, reward, funder, tick }`, `BountyClaimed { claimant, target, funder, reward, tick }`, `BountyExpired { target, tick }`, `ThreatReduced { settlement, amount, cause, tick }`.
- **Summary:** Posting is agentic; payouts and threat-reduction are event-driven reducers on `EntityDied`; compute body deletes.

---

## Reduction summary

| # | System | Old (v1) | New (v2) | Replacement |
|---|---|---|---|---|
| 1 | agent_inner | ESSENTIAL | ESSENTIAL (reduced) | Keep event emitter + needs physics; move derived fields to views |
| 2 | action_eval | ESSENTIAL | ESSENTIAL | Keep decision dispatch; scoring rules become DSL-declared views |
| 3 | action_sync | EMERGENT | DERIVABLE-VIEW | `npc_action()` method |
| 4 | moods | EMERGENT/DUP | DERIVABLE-VIEW | `mood(npc)` view + keep helper queries |
| 5 | fears | STUB/EMERGENT | DEAD | Phobias = belief derivation; phase 2 handled by agent_inner |
| 6 | personal_goals | EMERGENT | EMERGENT | `PursueGoal` action + `GoalAchieved` event |
| 7 | hobbies | EMERGENT | EMERGENT | `PracticeHobby(domain)` action |
| 8 | addiction | EMERGENT | EMERGENT + VIEW | `ConsumePotion` action + status-rule withdrawal |
| 9 | npc_relationships | DUPLICATIVE | DERIVABLE-VIEW | `relationship(a,b)` view over memory events |
| 10 | npc_reputation | EMERGENT/DUP | DERIVABLE-VIEW + DEAD | Reputation as view; heal aura → building passive |
| 11 | romance | EMERGENT | EMERGENT | `Court(target)` action + `Courted` event |
| 12 | rivalries | STUB | DEAD | View over shared memory events |
| 13 | companions | EMERGENT/DUP | DERIVABLE-VIEW | Folded into `mood()` |
| 14 | party_chemistry | EMERGENT | DERIVABLE-VIEW | `settlement_ambient_buff()` |
| 15 | nicknames | STUB | DEAD | `nickname(npc)` view |
| 16 | legendary_deeds | EMERGENT | DERIVABLE-VIEW | `is_legendary()` view + `BehaviorThresholdCrossed` event |
| 17 | folk_hero | STUB | DEAD | `fame()` view; optional `CrownFolkHero` action |
| 18 | trophies | STUB | DEAD | `ClaimTrophy` action if revived |
| 19 | mentorship | ESSENTIAL | EMERGENT | `Train(teacher)` + `Teach(student)` mutual actions |
| 20 | marriages | STUB | DEAD | `ProposeMarriage`/`AcceptMarriage` actions |
| 21 | bonds | STUB/DUP | DEAD + VIEW | Delete compute; helpers become views over memory |
| 22 | memorials | DUP (dead) | DEAD | `BuildMemorial` action if revived |
| 23 | journals | STUB | DEAD | `memory.events` IS the journal |
| 24 | cultural_identity | ESSENTIAL+EMERGENT | DERIVABLE-VIEW + EMERGENT | `settlement_culture()` view + `CultureEmerged` event |
| 25 | titles | ESSENTIAL | DERIVABLE-VIEW + EMERGENT | `title(npc)` view; `BestowTitle` formalizes |
| 26 | oaths | ESSENTIAL | EMERGENT | `SwearOath/BreakOath/AttemptFulfill` actions |
| 27 | grudges | STUB | DEAD + VIEW | Delete compute; grudges are belief-memory views |
| 28 | secrets | EMERGENT | EMERGENT + VIEW | `SecretAssigned` birth event + reveal events |
| 29 | intrigue | STUB | DEAD | Rebuild as tree of agentic plot actions |
| 30 | religion | EMERGENT/DUP | EMERGENT + VIEW | `JoinReligion`/`PerformRitual` actions; aura → building |
| 31 | demonic_pacts | EMERGENT | EMERGENT | `SignDemonicPact` action + status-rule clauses |
| 32 | divine_favor | EMERGENT/DUP | DERIVABLE-VIEW | Shared `settlement_divine_*` derivations |
| 33 | biography | EMERGENT | DERIVABLE-VIEW | Move to `views/` module |
| 34 | reputation_stories | EMERGENT | DERIVABLE-VIEW | `treasury_momentum()` or delete |
| 35 | wanted | ESSENTIAL (partial) | EMERGENT | `PostBounty`/`ClaimBounty` actions + event-driven payout |

### Bucket counts

| Bucket | v1 | v2 |
|---|---|---|
| ESSENTIAL | 7 | 2 (agent_inner reduced + action_eval) |
| EMERGENT | 15 | 12 (agentic actions) |
| DERIVABLE-VIEW | 0 | 13 |
| DEAD/STUB | 10 | 10 (all stubs stay dead; 8 ex-live systems join them as "delete") |

v1's "EMERGENT-CANDIDATE" label collapsed nearly everything that was "small drift + tag accumulation" into a single bucket. v2 splits them: if the field is a pure function, it's a VIEW; if the write represents an NPC *choosing* to do something, it's EMERGENT and becomes an action. The essential set shrinks from 7 to 2 — with `agent_inner` further reduced to its event-emitter + needs-physics core.

---

## Required action vocabulary

The DSL action space must grow from ~13 mechanical actions (Eat, Harvest, Attack, Flee, Build, Work, Move, Idle, …) to cover the social/cultural/political surface now done by scanning systems. Verbs required:

### Relationships & family
- `Court(target)` — flirt/romance progression
- `ProposeMarriage(target)`
- `AcceptMarriage(suitor)`
- `RefuseMarriage(suitor)`
- `AnnulMarriage(spouse)`
- `BefriendNpc(target)` (optional — mostly emergent from shared events)
- `DeclareRival(target)` (optional)
- `BetrayFriend(target)`
- `Forgive(target)`

### Oaths, titles, faith
- `SwearOath(kind, target_id)`
- `BreakOath(oath_idx)`
- `AttemptFulfillOath(oath_idx)`
- `BestowTitle(target, title)` — ruler-only
- `JoinReligion(faith)`
- `LeaveReligion`
- `PerformRitual(faith, target)`
- `SignDemonicPact(demon, terms)`
- `BreakPact(demon)`

### Skill & self-improvement
- `Train(teacher, building)` — apprentice-initiated
- `Teach(student)` — mentor-initiated
- `PracticeHobby(domain)`
- `PursueGoal(kind)` / `DeclareMastery(domain)`
- `ConsumePotion(kind)` / `ConsumeIntoxicant(kind)`
- `WriteJournal(content)` (optional cosmetic)

### Bounties & law
- `PostBounty(target, reward)` — settlement leader
- `ClaimBounty(corpse, poster)`
- `MarkWanted(target)`

### Secrets & intrigue
- `RevealSecret(target)` / `HideSecret` / `ConfessSecret`
- `BeginPlot(goal, targets)` / `RecruitConspirator` / `ExecutePlot(plot_id)` / `InformOn(conspirator)`

### Commemoration
- `BuildMemorial(target)` (optional)
- `ClaimTrophy(corpse)` (optional)
- `CrownFolkHero(target)` (settlement leader, optional)

---

## Required event types

Event-sourcing is the backbone of the new model. Every derived view needs an event history to fold over. Required events:

### Central observation event
- `Witnessed { observer, subject, event_kind, tick, impact }` — the primary emission channel for reputation, relationship, and mood derivations. Drops into `npc.memory.events` of the observer.

### Needs & physics
- `NeedsTicked { npc, Δhunger, Δshelter, Δsocial, tick }`
- `ResourceObserved { observer, resource_id, kind, pos, tick }`

### Agent decisions
- `ActionChosen { npc, action, target, score, tick }`
- `GoalAchieved { npc, kind, tier, tick }`
- `HobbyPracticed { npc, domain, tag_gain, tick }`
- `BehaviorThresholdCrossed { npc, tag, tier, tick }` — umbrella for legendary-deed/secret/title firing

### Relationships & family
- `Courted { a, b, tick }`
- `MarriageProposed / MarriageFormed / MarriageAnnulled { a, b, tick }`
- `RivalryDeclared { a, b, reason, tick }`
- `MadeNewFriend { a, b, tick }` (exists — keep)
- `TradedWith { a, b, tick }` (exists — keep)

### Oaths, titles, faith
- `OathSworn { swearer, kind, target, terms, tick }`
- `OathFulfilled { oath_id, tick }`
- `OathBroken { oath_id, tick }`
- `TitleBestowed { grantor, grantee, title, tick }`
- `ReligionJoined { npc, faith, tick }`
- `RitualPerformed { priest, faith, target, tick }`
- `PactSigned { npc, demon, clauses, tick }`
- `PactBroken { npc, demon, tick }`
- `PactTithe { npc, amount, tick }`

### Training & progression
- `TrainingSession { mentor, student, xp_gain, stat_deltas, tick }`
- `LevelUp { npc, tier, tick }`

### Consumption & status
- `PotionConsumed { npc, kind, tick }`
- `WithdrawalTick { npc, tick }` (or fold into status-effect eval log)

### Bounties & law
- `BountyPosted { target, reward, funder, tick }`
- `BountyClaimed { claimant, target, funder, reward, tick }`
- `BountyExpired { target, tick }`

### Secrets & intrigue
- `SecretAssigned { npc, kind, tick }` (birth-time)
- `SecretRevealed { npc, kind, observer, tick }`
- `PlotBegun / PlotJoined / PlotExecuted / PlotBetrayed / ConspiratorInformed`

### Culture
- `CultureEmerged { settlement, culture, first_tick }`
- `CultureAdopted { npc, culture, tick }`

### Commemoration
- `FolkHeroCrowned { npc, settlement, tick }` (optional)
- `MemorialBuilt { builder, target, tick }` (optional)
- `TrophyClaimed { npc, kind, source, tick }` (optional)

### Grudges
- `GrudgeFormed { holder, target, cause_event_id, tick }` (derived from memory events)
- `GrudgeResolved { holder, target, tick }`

---

## Required derived views

These are all pure functions — no storage, no per-tick writes. Computed on demand at read sites, optionally memoized at tick boundaries for hot paths.

### Per-NPC state
- `emotions(npc) = fold(recent memory.events × emotion_kernels, time_decay)`
- `mood(npc) = classify(emotions(npc), needs(npc), recent_events)`
- `morale(npc) = baseline + Σ recent_event_morale_impacts × decay`
- `personality_drift(npc) = integrate(memory.events × trait_kernels)`
- `beliefs(npc) = fold(memory.events → belief_deltas, decay=0.95/tick)`
- `aspiration(npc) = latest unmet_need_projection(needs, personality)`
- `grudges(npc) = memory.beliefs.filter(Grudge)`
- `phobias(npc) = f(memory.events matching traumatic_attack)`
- `grief(npc) = recent FriendDied events × decay` (part of emotions)
- `is_addicted(npc) = count_recent(PotionConsumed) > N within window`
- `religion_of(npc) = latest ReligionJoined − LeaveReligion`
- `pact_tier(npc) = active_pact_clauses_count(npc)`

### Social / reputation
- `reputation(npc) = Σ Witnessed events involving npc, weighted by observer settlement`
- `fame(npc) = Σ chronicle_mentions(npc) + Σ witnessed_positive_events`
- `nickname(npc) = top_tag_name(behavior_profile)`
- `title(npc) = determine_title(fulfilled_oaths, spouse_state, friend_deaths, attacks, gold, trade_tag, starvations, children)`
- `has_formal_title(npc) = any TitleBestowed where grantee==npc`
- `is_legendary(npc) = any behavior_value(tag) >= deed_table[tag].threshold`
- `legendary_tier(npc, tag) = thresholds_crossed(npc.behavior_value(tag))`
- `folk_hero_of(settlement) = argmax_fame(residents)`

### Pairwise
- `relationship(a,b) = f(shared memory.events between a and b, recency, positive/negative)`
- `rivalry(a,b) = behavior_tag_overlap(a,b) × combat_ratio`
- `bond_strength(a,b) = shared_battle_events + shared_survival_events, decayed`
- `has_battle_brothers(npc) = any bond_strength(npc, _) > threshold`
- `compatibility(a,b) = cosine(behavior_profile_a, behavior_profile_b)`
- `romantic_interest(a,b) = recent Courted events / decay`
- `marriage_eligibility(a,b)`

### Oaths
- `fulfilled_oaths(npc) = npc.oaths.filter(fulfilled)`
- `active_oaths(npc) = npc.oaths.filter(!fulfilled && !broken)`
- `broken_oaths(npc) = npc.oaths.filter(broken)`
- `oath_fulfilled(oath, state) = condition_met(oath.kind, state)`
- `oath_broken(oath, state) = team_changed || abandoned_home`
- `vengeance_target(npc) = first BeliefType::Grudge in beliefs`

### Settlement / world
- `settlement_culture(sid) = argmax_culture(Σ behavior_profiles of residents / npc_count)` when npc_count≥5 and score>1.0
- `cultural_alignment(npc) = cosine(behavior_profile, settlement_culture(home_settlement))`
- `settlement_ambient_buff(sid) = if pop>=2 { 1.05 } else { 1.0 }`
- `ambient_social_boost(grid, npc) = min(3, friendly_count) × 0.5`
- `settlement_divine_blessing(sid) = f(treasury, faith_count, recent_rituals)`
- `settlement_divine_wrath(sid) = f(low_treasury, missed_rituals)`
- `treasury_momentum(sid) = sign(treasury - threshold) × ε`
- `has_healer(sid) = any resident.class_tags contains healer|cleric|priest`
- `faithful_count(faith, sid)`
- `active_bounties(sid)`, `bounty_on(target)`

### Action scoring
- `utility_score(npc, candidate) = f(needs, personality, aspiration, emotions, cultural_bias, action_outcomes_EMA, distance)` — the inside of `action_eval`, now declared in DSL
- `eligible_mentors(npc, sid) = residents.level >= npc.level + 3 near knowledge_building`
- `xp_gain(mentor_level, apprentice_level) = BASE × (1 + mentor_level/20)`
- `goal_progress(npc, kind) = npc.behavior_value(kind.tag) / kind.threshold`
- `reveal_risk(npc) = f(stress, behavior_crossings)`

### Biography
- `biography(npc) -> String` (already pure in v1; just move out of `systems/`)
- `journal(npc) -> FormattedHistory` = format_memory_events(npc)

---

## Truly essential (irreducible) set in this batch

After strict re-classification, only **two** systems in this batch must remain as direct-mutation code:

### 1. `agent_inner.rs` — reduced to event emitter + needs physics

Keep only:
- `record_npc_event` (line 1537) — the canonical memory-event writer. Called from everywhere that observes something. This IS the event-sourcing primitive.
- `drift_needs` (hunger/shelter/social/esteem depletion with time) — this is physics, not derivation.
- `perceive_resources` (line 619, 3×3 spatial grid) — perception is an observation event; the discovery write to `known_resources` is the event-handler.

Delete or move to views:
- Emotion spikes (`spike_emotions_from_needs`) — view: `emotions(npc)`.
- Belief decay (line 28, 0.95/tick) — view: apply decay lazily at read time based on `tick - last_read_tick`.
- Personality drift (`drift_personality_from_memory` at line 862) — view: integrate memory events on demand.
- Aspiration recomputation — view: `aspiration(npc)`.
- Context-tag accumulation from emotions — view.
- Coworker social graph maintenance — view over coworker proximity events.
- Price-belief decay — view.
- Morale recovery — view over recent morale events.

Estimated reduction: ~1563 lines → ~300 lines of essential mutation + ~1200 lines moved into the view layer.

### 2. `action_eval.rs` — the agentic decision engine

Keep all ~1215 lines of it — but recognize that the *utility scoring rules* inside are pure derivations that should be DSL-declared, not hand-coded in Rust. The engine's essential work is: enumerate candidates, score each via `utility_score(npc, c)`, argmax, dispatch.

**Everything else deletes or becomes a view.** The per-tick cost for social systems drops from agent_inner 3.4% + ~12 mini-systems (estimated 1–2% combined) to just agent_inner's reduced core plus a handful of event-handler subscribers that fire only when their trigger event is emitted. Hot paths (`mood`, `emotions`, `relationship`, `reputation`) are memoized per tick at read sites.

### Key insight

The v1 taxonomy ("ESSENTIAL = 7") was still counting any system that performed a state write as essential, even when that write was a pure function of readable state plus time. Under the strict rubric, the test is: *can the written field be reconstructed by replaying the event log from t=0 against the entity's birth state?* For every social-layer field except `npc.memory.events` itself (and needs physics), the answer is yes — so the system is either an event subscriber (EMERGENT) or a lazy view (DERIVABLE-VIEW), never an imperative per-tick writer.

### Secondary observations

- **The narrow-window threshold-fire pattern is the same thing as a threshold-transition event.** v1 identified a unified `ThresholdFire` kernel merging personal_goals / legendary_deeds / secrets / titles / nicknames. Under event sourcing, this becomes even simpler: emit `BehaviorThresholdCrossed { npc, tag, tier }` whenever `behavior_value` crosses a band (detected as the side-effect of the `accumulate_tags` call), and subscribers generate chronicle entries + title flags. No kernel, no scan, no window-management.

- **Every scanning-based relationship system (npc_relationships, rivalries, bonds, romance, memorials) is reconstructable from `shared_memory_events(a, b)`.** If NPC A and NPC B are present in each other's memory.events with positive impact, they're friends. If negative, rivals. If both survived a battle (`WasInBattle` event with overlapping participants), they're battle brothers. The graph doesn't need maintenance — it's a projection over each NPC's memory.

- **Proxy-based systems (addiction via debuff count, religion/divine_favor via treasury, demonic_pacts via debuff count, moods/fears via hp_ratio) are the opposite of event sourcing: they invent a proxy state variable because they lack a proper event.** Adding the proper event (`PotionConsumed`, `PactSigned`, `RitualPerformed`, `WasAttacked`) makes the proxy unnecessary and the system deletable.

- **Stubs are stubs because the *action* was never implemented.** marriages, rivalries, nicknames, folk_hero, trophies, journals, grudges, intrigue were all stubbed because their primary verb ("get married," "become rivals," "earn a nickname," "become a folk hero," "keep a trophy," "write a journal," "hold a grudge," "plot a betrayal") is an NPC *choice* and the team correctly sensed that a scanning system couldn't capture it. The scanning body was never written because it wouldn't have been right. Under the new model, these entries become DSL action declarations, not Rust files.

- **`chronicle.push` is scattered across 6 systems in this batch.** Replace with a single `on_chronicle_event(event)` subscriber that formats event descriptions for the chronicle. Events like `OathFulfilled`, `TitleBestowed`, `CultureEmerged`, `BountyClaimed`, `MarriageFormed`, `LevelUp` all produce chronicle entries via one centralized formatter, not per-system string-building.

- **The dispatch model changes.** v1 had systems running on a fixed cadence (every 10, 17, 50, 100, 200, 500 ticks). The new model has only event subscribers (fire when event emitted) plus the two always-on essentials (`action_eval` every 5 ticks, `agent_inner` reduced core every 10 ticks). The aggregate per-tick cost for social systems drops to near-zero in quiet periods and rises naturally when things are actually happening (deaths, kills, level-ups) — which matches what you want for scalability.

## Migration shape

A plausible shape for the rewrite, drawn from the reclassifications above:

1. **Define the full action vocabulary** in DSL. Each action has `{ name, preconditions, utility_expr, targets_query, on_execute: emit_event_set }`. The 30+ new verbs listed in *Required Action Vocabulary* slot in alongside the existing mechanical actions.
2. **Define the view catalog.** Each view is `{ name, inputs: [state_fields, event_stream_queries], expression }`. ~50 views listed in *Required Derived Views*. Views are lazy by default; hot ones (mood, emotions, relationship) may be memoized per-tick at first read.
3. **Define the event schema.** The central event is `Witnessed { observer, subject, kind, impact, tick }` which lands in each observer's `memory.events`. Existing `MemEventType` variants (FriendDied, WasAttacked, WonFight, LearnedSkill, TradedWith, MadeNewFriend, Starved) stay; new variants added per *Required Event Types*. World-level events (`BountyPosted`, `CultureEmerged`, `MarriageFormed`, `PlotExecuted`) land in `state.world_events` and/or trigger broadcast to witnesses.
4. **Keep** `agent_inner` reduced-core, `action_eval`, `record_npc_event` helper.
5. **Delete** 10 stub bodies (fears, rivalries, nicknames, folk_hero, trophies, marriages, journals, grudges, intrigue, memorials compute) and 11 active-but-replaceable bodies (moods, personal_goals, hobbies, addiction, npc_relationships, npc_reputation, romance, companions, party_chemistry, legendary_deeds, mentorship, oaths, secrets, religion, demonic_pacts, divine_favor, reputation_stories, wanted, cultural_identity, titles). Only `action_sync`, `biography`, and the bonds/moods query helpers move into a `views/` module.
6. **Wire event subscribers** for the six chronicle-producing events (OathFulfilled, TitleBestowed, CultureEmerged, BountyClaimed, MarriageFormed, LevelUp). One formatter, one chronicle.push call site.
7. **Expand** `action_eval`'s candidate enumerator to cover the 30+ new actions. Each action is a small DSL declaration, not a new Rust module.

Under this shape, the social-layer code surface shrinks from ~35 files / ~5000 lines to approximately:

- `agent_inner.rs` — ~300 lines (event emitter + needs physics)
- `action_eval.rs` — ~1215 lines (decision engine, unchanged structurally)
- `views/social.rs` — ~400 lines (50 view functions, pure)
- `actions/social.dsl` — ~600 lines (30+ action declarations)
- `subscribers/chronicle.rs` — ~100 lines

Total: ~2615 lines, down from ~5000 — and the tick cost drops to just `action_eval` + reduced `agent_inner` plus event-driven subscriber work proportional to event volume rather than entity count × cadence. No more `O(S × E)` every-N-ticks scans.
