# Combat / Quest / Movement Systems — Strict Reframing

## Methodology

The v1 document classified 17 of 31 systems as ESSENTIAL. That is too generous:
most of those "essential" systems are decision-making or bookkeeping that could
be reframed as either NPC-chosen actions or read-time derived views.

This v2 pass applies a stricter rubric:

- **ESSENTIAL** — physics / bookkeeping that can't be reframed. The mutation
  is the only way the state changes. Damage application (on an attack event,
  `hp -= amount`), movement integration (`pos += force * dt`), and despawn
  (alive flip at `hp <= 0`) are canonical examples.
- **EMERGENT** — anything an NPC could *choose* as an action. Fighting,
  fleeing, picking up loot, posting or accepting quests, joining a party,
  entering a dungeon, breeding, scouting, carrying a message. The DSL
  provides the action vocabulary; the system body moves into the agent's
  action-eval / action-execute policy.
- **DERIVABLE-VIEW** — state that is a pure function of log-events or time.
  Threat level (density × attacks − patrols), monster name (f of kill count),
  nemesis (kill_count ≥ K), recovery (function of time-since-damage),
  cooldown remaining (`max(0, end_tick − now)`), tile explored (∃ visit event).
- **DEAD / STUB** — not dispatched or empty.

Each system is reclassified below. Citations are `file:line` into
`src/world_sim/systems/*.rs`.

---

## Per-system reanalysis

### battles
- **Old:** ESSENTIAL (sole source of damage deltas)
- **New:** split — ESSENTIAL (damage application) + EMERGENT (engagement choice)
- **Reframe:** The **damage delta** (`WorldDelta::Damage { target, amount, source }`
  at `battles.rs:96,132`) is the irreducible physics — when a damage event
  lands, `hp -= amount`. But the decision that an engagement is happening
  (`fidelity == High`, friendlies and hostiles co-located, `atk/fcount`
  split) is agentic: each NPC or monster chooses to attack a chosen target.
  In the new model, each combatant emits a per-tick `Attack(target_id)`
  action; the sim integrates by applying one damage event per action; the
  "everyone attacks everyone" aggregation at `battles.rs:88-165` goes away.
  The chronicle entries and `AddBehaviorTags` at lines 98-122 are observer
  effects tacked onto the damage event and belong in a derivable chronicle
  view.
- **Required NPC actions:** `Attack(target_id)`, `Focus(target_id)`
- **Required derived views:** `in_combat(entity) = ∃ recent Attack event involving entity`
- **Required event types:** `DamageDealt { source, target, amount, kind }`,
  `AttackEmitted { source, target }`
- **One-line summary:** ESSENTIAL damage mutation triggered by EMERGENT
  `Attack` actions; the aggregate battle-step dies.

---

### loot
- **Old:** ESSENTIAL (spawns items, transfers gold)
- **New:** EMERGENT (pickup choice) + ESSENTIAL (item-entity spawn + gold transfer)
- **Reframe:** Right now `loot.rs:79-137` automatically distributes bounty
  gold and drops items on every monster death. Both operations should split.
  The drop itself (`SpawnItem` at `loot.rs:117-135`) is a physics fact: a
  dying entity drops a container entity at its pos — ESSENTIAL spawn, but
  triggered by the existing `EntityDied` event, not by a periodic scanner.
  NPCs then **choose** to walk over and `PickUp(item_id)`, which transfers
  ownership. The current behavior of auto-paying gold to every alive
  friendly on the grid (`loot.rs:86-95`) is an EMERGENT "claim bounty"
  action, not a physics step.
- **Required NPC actions:** `PickUp(item_id)`, `ClaimBounty(monster_id)`,
  `EquipItem(item_id, slot)`
- **Required derived views:** `dropped_items_nearby(entity) = filter items by pos`
- **Required event types:** `ItemDropped { from_entity, item_id, pos }`,
  `ItemPickedUp { entity, item_id }`, `BountyClaimed { hunter, target, payout }`
- **One-line summary:** EMERGENT pickup/claim actions; spawn of item entity
  on death remains ESSENTIAL bookkeeping.

---

### last_stand
- **Old:** ESSENTIAL (damage/heal/shield deltas)
- **New:** EMERGENT (triggered ability) + ESSENTIAL (damage/heal mutations)
- **Reframe:** `last_stand.rs:76-137` fires a burst-damage + self-heal +
  shield + morale rally whenever a friendly's `hp/max_hp ≤ 0.15`. This is
  the classic shape of a **triggered ability**: condition + effect sequence.
  In the DSL it becomes an `.ability` file with a `when hp_pct < 0.15`
  trigger and a composite effect (Damage AoE + SelfHeal + Shield +
  MoraleAoE). The decision to *use* the ability is EMERGENT (NPCs with
  the ability choose to fire it based on urgency). Underlying damage and
  heal mutations are ESSENTIAL as always.
- **Required NPC actions:** `CastAbility(last_stand)`
- **Required derived views:** `ability_available(npc, ability) = cooldown_remaining == 0 ∧ trigger_conds`
- **Required event types:** `AbilityCast { caster, ability, targets }`
  (which fans out into DamageDealt / HealApplied / ShieldApplied /
  MoraleAdjusted)
- **One-line summary:** EMERGENT ability usage; all state mutation goes
  through the existing essential damage/heal pipeline.

---

### skill_challenges
- **Old:** ESSENTIAL (reward + damage flow)
- **New:** EMERGENT
- **Reframe:** `skill_challenges.rs:61-153` is a periodic cron that rolls
  skill checks for every friendly NPC on a High-fidelity grid and pays
  gold / deals damage based on outcome. This is an NPC action:
  `AttemptChallenge(difficulty)`. The NPC decides whether to attempt (risk
  vs reward), the sim resolves the roll, and the gold/damage mutations
  flow through standard `TransferGold` / damage events. The cron itself
  disappears.
- **Required NPC actions:** `AttemptChallenge(challenge_kind, difficulty)`
- **Required derived views:** `challenges_available(region) = f(grid_fidelity, hostile_count)`
- **Required event types:** `ChallengeAttempted { entity, kind, outcome }`
  (outcome drives existing DamageDealt / GoldTransferred)
- **One-line summary:** EMERGENT — an NPC-selected attempt action; reward
  and damage ride existing event types.

---

### dungeons
- **Old:** ESSENTIAL (state machine + exploration rewards)
- **New:** mostly EMERGENT + DERIVABLE-VIEW
- **Reframe:** `dungeons.rs` has three phases. (1) Monster regen/threat
  pressure `dungeons.rs:64-89` is a DERIVABLE-VIEW of monster density —
  fidelity escalation is a function of `region.monster_density > 40`, not
  a state to maintain. (2) Exploration rewards `dungeons.rs:91-149` are
  the payout side of an EMERGENT `EnterDungeon(dungeon_id)` action that
  an NPC (or party) chose. (3) Hazard damage on friendlies
  (`dungeons.rs:143-148`) is per-tick attrition inside a dungeon — model
  this as a `DungeonHazardEvent` fired by a tile/region zone rather than
  a cron. Dungeon "state" (explored_depth, is_cleared) is updated by
  NPC explore actions, not by this system.
- **Required NPC actions:** `EnterDungeon(dungeon_id)`, `ExploreDepth()`,
  `ExitDungeon()`
- **Required derived views:** `dungeon_fidelity(d) = f(monster_density, party_present)`,
  `dungeon_cleared(d) = explored_depth ≥ max_depth`
- **Required event types:** `DungeonEntered { party, dungeon }`,
  `DungeonDepthExplored { dungeon, depth, party }`,
  `DungeonHazardApplied { target, amount }`
- **One-line summary:** EMERGENT exploration actions + DERIVABLE-VIEW
  fidelity; the state machine collapses.

---

### escalation_protocol
- **Old:** EMERGENT-CANDIDATE (alongside threat)
- **New:** DERIVABLE-VIEW
- **Reframe:** `escalation_protocol.rs:74-117` derives grid fidelity from
  region threat ≥ 60 and dead/alive hostile counts. The war-exhaustion
  treasury drain at `escalation_protocol.rs:99-104` is incidental; it is
  only firing because the cron is running. Fidelity is a pure function
  of `(threat_level, recent_kills, patrols)`; compute it on read.
- **Required derived views:**
  `grid_fidelity(grid) = max_desired(threat_level, hostile_count, recent_kill_ratio)`
- **Required event types:** none; reads existing threat + kill events
- **One-line summary:** DERIVABLE-VIEW — fidelity escalation is a function
  over region state, not a scheduled mutation.

---

### threat
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `threat.rs:54-77` computes `threat_delta = density_pressure
  − patrol_reduction − decay` every 50 ticks. All three inputs already
  live in the world state (monster_density, NPC positions, time since
  last kill). Fidelity-rank escalation at `threat.rs:118-131` is also a
  function of threat. The "settlement threat drifts to regional" pass at
  lines 140-158 is a smoothing filter that can be read instead of stored.
  Only the chronicle milestone emissions at lines 81-100 need persisting
  — and those are event emissions, not state mutations.
- **Required derived views:**
  `threat(region) = clamp(sum(monster_density_pressure) + recent_attacks(region) − patrol_rate(region) − decay, 0, 100)`,
  `settlement_threat(s) = max_region_threat(s.faction) / 100`
- **Required event types:** `ThresholdCrossed { region, metric, from, to }`
  (for chronicle milestones only)
- **One-line summary:** DERIVABLE-VIEW — threat is a continuous function
  of density and patrols; cron updates aren't needed.

---

### threat_clock
- **Old:** ESSENTIAL (growth pressure)
- **New:** DERIVABLE-VIEW
- **Reframe:** `threat_clock.rs:38-97` runs every 100 ticks. The "clock"
  is the average region threat (`threat_clock.rs:42-46`), which is itself
  derivable. The density increase it emits to regions (`threat_clock.rs:66-77`)
  reproduces the exact "derivable" pattern with extra state — it's a
  pressure function of `(tick, combat_npc_count, alive_npc_count)` that
  can be computed on read. The threshold-crossing chronicle entries at
  `threat_clock.rs:81-97` are events that should fire from the derived
  view as it crosses thresholds.
- **Required derived views:**
  `global_threat_clock(tick) = base_growth * tick + acceleration_term(tick) − suppression(combat_npcs, alive_npcs)`,
  `region_monster_density(region, tick) = base + ∫ growth − kills(region)`
- **Required event types:** `ThresholdCrossed { scope=world, metric=threat, band }`
- **One-line summary:** DERIVABLE-VIEW — the clock is a closed-form
  function of tick + demographic counts.

---

### adventuring
- **Old:** ESSENTIAL (party formation + movement + quest lifecycle)
- **New:** EMERGENT (all of it)
- **Reframe:** Every phase of `adventuring.rs` maps to an NPC action.
  Party formation (`adventuring.rs:38-108`): an NPC with `combat > 10
  ∧ wants_adventure` chooses `FormParty`; the sim groups them by
  co-location + intent. Destination selection (`adventuring.rs:145-189`,
  includes grudge-target, nearest-dungeon, or random): this is the party
  leader emitting `ChooseQuestTarget(dest)`. Party movement
  (`adventuring.rs:367-372`) is handled by `move_target`, so this system
  only sets it. Arrival + rewards (`adventuring.rs:240-366`) are
  `CompleteQuest(quest_id)` / `DungeonClearedEvent` consequences — gold,
  tags, memory, chronicle. Rival clashes (`adventuring.rs:375-497`) are
  a second-order `Attack` from each party, not a special-case system.
- **Required NPC actions:** `FormParty`, `JoinParty(party_id)`,
  `LeaveParty`, `ChooseQuestTarget(pos)`, `EnterDungeon(d)`, `ExitDungeon`,
  `CompleteAdventure`
- **Required derived views:** `eligible_adventurer(npc)`,
  `current_party(npc)`, `party_destination(party)`
- **Required event types:** `PartyFormed { members, leader }`,
  `PartyDisbanded`, `DungeonDepthExplored`, `RelicFound { finder, relic }`,
  `PartyClash { winners, losers }`
- **One-line summary:** EMERGENT — all 499 lines decompose into per-NPC
  actions; the only essential thing is the entity spawn for relics at
  `adventuring.rs:323-364`, which is the `SpawnItem` physics.

---

### quests
- **Old:** ESSENTIAL (generation + acceptance + lifecycle)
- **New:** EMERGENT
- **Reframe:** Three phases, all EMERGENT. (1) Generation
  (`quests.rs:56-105`): a settlement with threat ≥ 0.3 ∧ treasury > 10
  chooses `PostQuest(description, reward)` — this is the settlement
  leader NPC acting. (2) Acceptance (`quests.rs:108-196`): a
  combat-capable NPC evaluates quests and emits `AcceptQuest(quest_id)`.
  (3) Lifecycle (`quests.rs:199-290`): arrival detection is DERIVABLE
  (`at_destination = |pos − dest| < threshold`); completion roll and
  reward is `CompleteQuest(quest_id)`.
- **Required NPC actions:** `PostQuest(threat_level, reward, destination)`,
  `AcceptQuest(quest_id)`, `AbandonQuest(quest_id)`, `CompleteQuest(quest_id)`
- **Required derived views:** `quest_board(settlement) = open_quests filtered by settlement`,
  `at_quest_destination(npc) = |pos − quest.dest| < 5`,
  `quest_expired(q) = q.deadline < now`
- **Required event types:** `QuestPosted { settlement, quest }`,
  `QuestAccepted { quester, quest }`, `QuestCompleted { quest, party }`,
  `QuestFailed { quest, reason }`
- **One-line summary:** EMERGENT — post/accept/complete are three
  distinct NPC actions; the cron dissolves.

---

### quest_lifecycle
- **Old:** ESSENTIAL (progress + expiry + stale)
- **New:** DERIVABLE-VIEW + EMERGENT consequences
- **Reframe:** `quest_lifecycle.rs:46-113` checks arrival, progress,
  expiry, and staleness. Arrival is derivable from party position vs.
  quest destination. Expiry is derivable (`now > deadline_tick`).
  Staleness is derivable (`now − accepted_tick > 200 ∧ progress < 0.01`).
  What remains is the payout on completion (`complete_quest`) and the
  cleanup on fail (`expire_quest` / `check_stale`) — both are EMERGENT
  responses to derived predicates crossing.
- **Required derived views:** `quest_at_destination(q)`,
  `quest_expired(q)`, `quest_stale(q)`, `quest_progress(q)`
- **Required event types:** `QuestDeadlinePassed { quest }`,
  `QuestBecameStale { quest }`, `QuestProgressAdvanced { quest, delta }`
- **One-line summary:** DERIVABLE-VIEW — tick the derived predicates;
  fire events only on crossings; completion/failure is EMERGENT.

---

### seasonal_quests
- **Old:** STUB (skeletal)
- **New:** DERIVABLE-VIEW
- **Reframe:** `seasonal_quests.rs:21-41` drops a flat 15-gold bonus to
  every settlement at season change. This is either a settlement-leader
  `PostSeasonalQuest` action (EMERGENT) or a DERIVABLE-VIEW
  (quests-at-season-boundary computed from the season and threats). The
  current flat-gold implementation is neither; it's a vestige — fold
  into `seasons.rs` or delete.
- **Required derived views:** `seasonal_quest_board(season, region) = templates × threats`
- **Required event types:** (none; quests posted use the same `QuestPosted` event)
- **One-line summary:** DERIVABLE-VIEW — delete the flat-gold kludge;
  derive seasonal quests from season + threats.

---

### bounties
- **Old:** DUPLICATIVE (vs loot)
- **New:** EMERGENT (post/claim) + DERIVABLE-VIEW (who-is-a-target)
- **Reframe:** Three phases in `bounties.rs`. (1) Auto-complete
  (`bounties.rs:64-163`): pays gold to nearby friendlies when a
  high-value target dies. This is an EMERGENT `ClaimBounty(target_id)`
  action by the slayer; dedup with `loot.rs`. (2) Posting
  (`bounties.rs:166-209`): a settlement leader emits
  `PostBounty(target_id, reward)`. (3) Implicit funding
  (`bounties.rs:212-222`): DERIVABLE-VIEW — high-threat region funds
  bounties; this is a settlement-income derivation, not a mutation.
  Who counts as a high-value target (`bounties.rs:227-242`) is also
  DERIVABLE-VIEW (`is_bounty_target(e) = e.level ≥ K ∨ e.faction.hostile`).
- **Required NPC actions:** `PostBounty(target, reward)`,
  `ClaimBounty(target)`
- **Required derived views:** `bounty_targets = filter entities by level + faction`,
  `bounty_funding(region) = threat_level × funding_rate`
- **Required event types:** `BountyPosted`, `BountyClaimed { hunter, target, payout }`,
  `BountyExpired`
- **One-line summary:** EMERGENT actions + DERIVABLE-VIEW of
  target-eligibility; system folds into loot on the claim side.

---

### treasure_hunts
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `treasure_hunts.rs:31-138` periodically rewards NPCs who
  are far from home. The right model is an NPC `StartTreasureHunt()`
  action that commits to a destination, then `FindTreasureStep()` while
  at the destination, then `ReturnTreasure()` to deposit. The
  "distance_multiplier" at `treasure_hunts.rs:86-88` is the derived
  reward-scaling, not a cron input.
- **Required NPC actions:** `StartTreasureHunt(destination)`,
  `FindTreasureStep`, `ReturnTreasure`
- **Required derived views:** `treasure_reward(entity) = f(distance_from_home, discovery_tier)`
- **Required event types:** `TreasureStepFound { hunter, pos, reward }`,
  `ArtifactDiscovered { hunter, artifact }`
- **One-line summary:** EMERGENT — three actions replace the cron; the
  artifact drop is an `ItemSpawned` event like normal loot.

---

### heist_planning
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `heist_planning.rs:50-125` scans NPCs near foreign
  settlements and rolls success/failure. Model it as a multi-phase NPC
  action: `PlanHeist(target)`, `ScoutTarget`, `Infiltrate`, `ExecuteHeist`,
  `Escape`. Each phase corresponds to the existing PHASE_DURATION
  division. Gold transfer on success is the essential physics; damage on
  failure is the essential physics.
- **Required NPC actions:** `PlanHeist(target_settlement)`, `Scout(target)`,
  `Infiltrate(target)`, `ExecuteHeist(target)`, `AbortHeist`
- **Required derived views:** `heist_skill(npc) = f(level, stealth_tag)`,
  `heist_success_prob(npc, target) = f(skill, target_guard_strength)`
- **Required event types:** `HeistAttempted { crew, target, outcome }`,
  `HeistGoldStolen { target, amount, crew }`
- **One-line summary:** EMERGENT — five actions for the five heist phases.

---

### exploration (tile half)
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `exploration.rs:41-146` boosts settlement treasury based
  on NPC count and escalates grid fidelity. All inputs are already in
  state. Tile-reveal is `f(set_of_positions_visited)`. Treasury bonus
  from exploration is a DERIVABLE-VIEW income: `exploration_income(s) =
  alive_npcs_nearby × 0.1`.
- **Required derived views:**
  `tile_explored(tile) = ∃ visit_event(tile)`,
  `exploration_income(settlement) = count(alive_npcs within R) × 0.1`,
  `desired_grid_fidelity(s) = f(nearby_npc_count, threat)`
- **Required event types:** `PositionVisited { entity, tile }` (emitted
  opportunistically from movement integration)
- **One-line summary:** DERIVABLE-VIEW — tile exploration is a set-union
  over visit events; income bonus is a read-time aggregate.

---

### exploration (voxel resource census)
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (materialization of a DERIVABLE-VIEW)
- **Reframe:** Conceptually the per-cell resource count
  (`exploration.rs:534-633`) is a **pure function** of the voxel world:
  `census(cell) = histogram of TARGET_MATERIALS in cell`. It is
  derivable from the voxel grid. But evaluating it lazily per entity
  would be prohibitive (420K voxel reads per NPC — see the implementation
  notes at `exploration.rs:638-645`), so the system materializes the
  view into `state.cell_census` and the per-NPC `known_voxel_resources`
  list. Keep ESSENTIAL, but flag explicitly that this is a
  **performance-driven materialization of a derivable view**, not a
  physics step. If we ever cache the voxel histogram differently, this
  system collapses into a DERIVABLE-VIEW.
- **Required derived views:** `cell_census(cell) = histogram(voxels in cell)`;
  `known_resources(npc) = cells in npc.visible_disk ∩ tick_observed ≥ last`
- **Required event types:** `CellCensusComputed { cell }` (internal
  metric only)
- **One-line summary:** ESSENTIAL *materialization* — voxel census is
  a derivable view, cached because lazy eval is unaffordable.

---

### movement
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (unchanged — the canonical integration step)
- **Reframe:** `movement.rs:11-82` is the one place positions mutate:
  `entity.pos += (dir * speed * dt) / tile_cost`. This is the
  `pos = pos + force × dt` integration. Every other system sets
  `entity.move_target`; this integrates. Keep as the single position
  kernel.
- **Required derived views:** n/a (this is the mutation)
- **Required event types:** `MovementApplied { entity, old_pos, new_pos }`
  (optional — for tile-explored derivation)
- **One-line summary:** ESSENTIAL — the sole integrator of position;
  cannot be reframed.

---

### pathfollow
- **Old:** STUB
- **New:** DEAD
- **Reframe:** `pathfollow.rs:14-16` is a no-op. Delete.
- **One-line summary:** DEAD — delete the file.

---

### travel
- **Old:** DUPLICATIVE (vs agent_inner)
- **New:** EMERGENT
- **Reframe:** `travel.rs:15-65` walks NPCs toward destinations set by
  `Travel` / `Trade` intents and drains carried food. Movement is
  already ESSENTIAL in `movement.rs`. The food drain is a consequence
  of travel duration, not a system — it belongs in the agent's action
  tick. The choice to travel is already EMERGENT in quest/trade actions.
- **Required NPC actions:** `Travel(destination)` (maps to setting
  `move_target` and enabling food drain on the active action)
- **Required derived views:** `traveling(npc) = npc.economic_intent in {Travel, Trade}`
- **Required event types:** `FoodConsumed { entity, amount }`
  (folds into existing `ConsumeCommodity` event)
- **One-line summary:** EMERGENT — NPC chose to travel; food drain is
  an action-tick consequence, not a system.

---

### monster_ecology
- **Old:** ESSENTIAL (spawning + ambient)
- **New:** EMERGENT (most of it) + ESSENTIAL (entity spawn / revive)
- **Reframe:** Five phases in `monster_ecology.rs`. (1) Respawn
  (`monster_ecology.rs:37-128`) revives dead monsters: **conceptually
  an EMERGENT "ambient wilderness spawns a monster" choice**, but the
  mechanical revive (Heal + SetPos) is the essential bookkeeping.
  (2) Settlement attacks via treasury drain
  (`monster_ecology.rs:133-151`): DERIVABLE-VIEW — settlement damage
  from dense monster regions is a function. (3) Migration
  (`monster_ecology.rs:161-218` and `advance_monster_ecology:355-411`):
  a monster chooses `Migrate(direction)` / `SeekFood(target)` /
  `FleeSettlement` as an action. (4) Reproduction
  (`monster_ecology.rs:225-278`): EMERGENT — two nearby monsters choose
  `Breed()`, sim spawns a new entity. (5) Den formation
  (`monster_ecology.rs:286-322`): DERIVABLE-VIEW — a den is a cell
  where `density(monsters within R) ≥ K` for T ticks; the density
  update on region is a physics consequence but its threshold test
  is a view.
- **Required monster actions:** `MigrateToward(pos)`, `SeekFood(settlement)`,
  `FleeSettlement`, `Breed(mate)`, `AttackSettlement(s)`
- **Required derived views:** `den_forming(region)`,
  `settlement_under_attack(s) = monster_density > 80 ∧ raid_roll`,
  `monster_hunger(m, season) = f(local_wild_food)`
- **Required event types:** `MonsterSpawned { entity, pos, cause }`,
  `MonsterBred { parents, child }`, `DenFormed { region }`
- **One-line summary:** EMERGENT — monsters are NPCs too; spawn /
  despawn remain ESSENTIAL physics.

---

### monster_names
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `monster_names.rs:17-128` gives a monster a name after 3+
  NPC kills. The "name" is a pure function of `(kill_count, creature_type,
  hash(monster_id))`. Stat buffs on naming (`monster_names.rs:79-85`) are
  the real mutation — but those can fold into a generic
  `MilestonePromotion` event fired from the derived view when
  `is_named(m)` first flips true. Grudge formation
  (`monster_names.rs:109-124`) is EMERGENT (NPCs choosing to hold a
  grudge on the derived "named monster" event).
- **Required derived views:**
  `monster_name(m) = if kills(m) ≥ 3 then format(prefix, suffix, hash) else entity_display_name(m)`,
  `is_named(m) = kills(m) ≥ 3`
- **Required event types:** `MonsterBecameNamed { monster, name, kills }`
- **One-line summary:** DERIVABLE-VIEW — name is a pure function of
  kill history.

---

### nemesis
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `nemesis.rs:82-307` designates high-level hostile
  monsters as nemesis champions and periodically buffs them. But
  "nemesis" is just a threshold test: `is_nemesis(m) = kill_count(m,
  player_kin) ≥ K` or `m.level ≥ 5 ∧ hostile`. The level-up
  (`nemesis.rs:259-307`) mechanically adds stats every 500 ticks —
  those stats are themselves a function of time-since-designation, i.e.
  `nemesis_stats(m) = base + (now − designated_tick) × per_level_buff`.
  The "slayer reward" on death (`nemesis.rs:83-153`) is an EMERGENT
  `ClaimNemesisBounty` reaction — not state mutation. Spawn/designation
  (`nemesis.rs:159-253`) is a faction-leader action.
- **Required derived views:**
  `is_nemesis(m) = m.kind=Monster ∧ m.hostile ∧ m.level ≥ K`,
  `nemesis_stats(m, now) = base + (now − designated) × scale`
- **Required event types:** `NemesisDesignated { monster, faction }`,
  `NemesisSlain { slayer, nemesis }`
- **One-line summary:** DERIVABLE-VIEW — nemesis status is a threshold
  function; stat growth is a closed-form time function.

---

### wound_persistence
- **Old:** ESSENTIAL
- **New:** DERIVABLE-VIEW
- **Reframe:** `wound_persistence.rs:24-85` heals below-max-HP NPCs at
  rates that depend on `economic_intent` (idle heals fast, traveling
  slower, fighting not at all). Canonically:
  `hp(npc, now) = clamp(last_hp + (now − last_dmg_tick) × regen_rate(activity), 0, max_hp)`.
  If we store `last_damage_tick` per NPC and derive hp on read, no
  periodic heal system is needed.
- **Required derived views:**
  `hp_effective(npc, now) = clamp(npc.last_hp + ticks_since_damage(npc) × regen_rate(npc.activity), 0, max_hp)`
- **Required event types:** (reads existing `DamageDealt` to bump
  `last_damage_tick`)
- **One-line summary:** DERIVABLE-VIEW — regen is a closed-form function
  of time-since-damage and activity.

---

### adventurer_condition
- **Old:** ESSENTIAL (drift + desertion)
- **New:** DERIVABLE-VIEW (drift) + EMERGENT (desertion)
- **Reframe:** `adventurer_condition.rs:33-141` emits status effect
  debuffs (fatigue/stress) and morale deltas based on activity. Same
  pattern as wound_persistence: `stress(npc, now) = base_stress +
  ∫ activity_drift` is a function of activity history. The current
  implementation samples every 10 ticks; the derived view reads any
  tick. Desertion (guarded by loyalty < 15 ∧ stress > 85) is an
  EMERGENT `Desert` action the NPC chooses.
- **Required NPC actions:** `Desert`
- **Required derived views:**
  `stress(npc, now) = clamp(base + Σ activity_drift(activity_at_tick) × dt, 0, 100)`,
  `fatigue(npc, now) = same pattern`,
  `fatigue_debuff_active(npc) = fatigue(npc) > threshold`
- **Required event types:** `NpcDeserted { entity, from_settlement }`
- **One-line summary:** DERIVABLE-VIEW stress/fatigue + EMERGENT
  desert action.

---

### adventurer_recovery
- **Old:** ESSENTIAL
- **New:** DERIVABLE-VIEW (recovery) + EMERGENT (medicine use)
- **Reframe:** `adventurer_recovery.rs:43-181` heals HP and strips
  debuffs at recovery intervals. Same argument as wound_persistence —
  `hp_recovered(npc, now) = function of time and activity`. The
  medicine-accelerated healing (`adventurer_recovery.rs:128-181`) is an
  EMERGENT `UseMedicine` action an NPC chooses when
  `hp/max_hp < 0.8 ∧ settlement.medicine_stock > 0`.
- **Required NPC actions:** `UseMedicine(amount)`
- **Required derived views:**
  `recovered_hp(npc, now)` — same formula as wound_persistence
- **Required event types:** `MedicineConsumed { entity, amount, source }`,
  `DebuffsCleared { entity, cause=recovery }`
- **One-line summary:** DERIVABLE-VIEW passive regen + EMERGENT
  medicine action; duplicates wound_persistence.

---

### cooldowns
- **Old:** ESSENTIAL
- **New:** DERIVABLE-VIEW
- **Reframe:** `cooldowns.rs:14-37` fires a `TickCooldown` delta every
  tick for every alive entity. The cooldown is just `remaining =
  max(0, end_tick − now)`. No state needs to change each tick; read
  `end_tick` stored when the ability fired, compute remaining on read.
  The only mutation is setting `end_tick` on ability cast — which is
  already captured by ability-cast events.
- **Required derived views:**
  `cooldown_remaining(npc, ability) = max(0, npc.cd_end_tick[ability] − now)`,
  `ability_ready(npc, ability) = cooldown_remaining(npc, ability) == 0`
- **Required event types:** (uses existing `AbilityCast` to set `end_tick`)
- **One-line summary:** DERIVABLE-VIEW — cooldown is `max(0, end − now)`;
  no per-tick tick is needed.

---

### death_consequences
- **Old:** ESSENTIAL
- **New:** mostly EMERGENT + ESSENTIAL (inheritance transfer)
- **Reframe:** `death_consequences.rs:31-288` fires on `EntityDied`
  events this tick and performs: inheritance gold transfer, mourning
  (friend grief), funeral chronicle, memorial chronicle, apprentice
  lineage inheritance. Break down: (1) inheritance at
  `death_consequences.rs:84-123` is the only mandatory physics — a
  `BequestEvent` splits the dead NPC's gold to heirs + treasury; this
  is ESSENTIAL because it preserves gold conservation. (2) Mourning
  and funeral attendance (`death_consequences.rs:125-154`) are
  EMERGENT actions — attending NPCs choose `AttendFuneral`, which
  applies the grief / social tick. (3) Apprentice lineage tag-inheritance
  (`death_consequences.rs:156-227`) is an EMERGENT action by the
  apprentice — `InheritMastersTags(master_id)`. (4) Chronicle /
  memorial are event emissions, not state mutation — they fall out of
  the funeral / memorial events. The actual `alive=false` flip already
  happens at damage-application time.
- **Required NPC actions:** `AttendFuneral(dead_id)`,
  `InheritMastersTags(master_id)`, `MourningRite(dead_id)`
- **Required derived views:** `funeral_active(dead, now) = now − death_tick ≤ 20`,
  `is_heir(npc, dead) = npc.home_building_id == dead.home_building_id`,
  `memorial_eligible(dead) = dead.level ≥ 10 ∧ chronicle_mentions(dead) ≥ 3`
- **Required event types:** `BequestEvent { from, to, amount }`
  (essential — gold conservation), `FuneralHeld { dead, attendees }`,
  `MemorialRaised { dead, settlement }`, `ApprenticeLineage { heir, master, tags }`
- **One-line summary:** ESSENTIAL bequest physics + EMERGENT
  funeral/mourning/lineage actions.

---

### sea_travel
- **Old:** ESSENTIAL
- **New:** EMERGENT
- **Reframe:** `sea_travel.rs:26-171` fast-tracks NPCs between coastal
  settlements and rolls sea-monster encounters. The choice to take a
  sea route is an NPC action `SailTo(coastal_dest)`. The sea-monster
  encounter is an `Attack` emitted by an (offscreen) sea-monster — fits
  the standard damage pipeline. `move_speed_mult = 2.0` is a status
  effect (`Boon("sea_speed")`) rather than a system-level mutation.
- **Required NPC actions:** `SailTo(coastal_dest)`, `DisembarkAt(coastal)`
- **Required derived views:** `at_sea(npc) = traveling ∧ no_nearby_settlement`,
  `coastal(settlement) = any region(s) is_coastal`
- **Required event types:** `VoyageStarted { entity, from, to }`,
  `SeaMonsterEncounter { entity }` (fans out to DamageDealt)
- **One-line summary:** EMERGENT — sail as an action; sea-monster strike
  is a standard damage event.

---

### scouting
- **Old:** DUPLICATIVE (vs messengers)
- **New:** EMERGENT
- **Reframe:** `scouting.rs:22-78` shares price reports from settlements
  to traveling NPCs near them. The act of observing is an NPC action
  `ObserveMarket(settlement_id)` that writes a price report onto the
  observer. "Scouting" collapses to "an NPC performing the Observe
  action while near a settlement". The cron disappears.
- **Required NPC actions:** `ObserveMarket(settlement)`, `ShareReport(recipient)`
- **Required derived views:** `fresh_report(npc, s) = ∃ report with tick_observed > now − 200`
- **Required event types:** `PriceReportObtained { observer, settlement }`,
  `PriceReportShared { from, to, settlement }`
- **One-line summary:** EMERGENT — observe + share are NPC actions.

---

### messengers
- **Old:** DUPLICATIVE (vs scouting)
- **New:** EMERGENT
- **Reframe:** `messengers.rs:20-57` shares an NPC's `price_knowledge`
  with the destination settlement when the NPC has a Trade intent. The
  act of delivering info is an NPC action `CarryMessage(from, to,
  payload)` or, simpler, `DeliverReport(report, to)` on arrival. Merges
  with scouting's ShareReport.
- **Required NPC actions:** `CarryMessage(from, to, payload)`,
  `DeliverReport(report, recipient)`
- **Required derived views:** `messages_in_transit(entity)`
- **Required event types:** `MessageDelivered { carrier, from, to, payload }`
- **One-line summary:** EMERGENT — merge with scouting into a single
  "observe/carry/deliver report" action vocabulary.

---

### goal_eval
- **Old:** DEAD (superseded by action_eval)
- **New:** DEAD
- **Reframe:** `goal_eval.rs:28` (`evaluate_goals`) is not invoked from
  the runtime. V1's audit confirmed this (replaced by `action_eval`).
  The file is ~527 lines of legacy code.
- **One-line summary:** DEAD — delete.

---

### world_goap
- **Old:** DEAD (superseded by action_eval)
- **New:** DEAD
- **Reframe:** `world_goap.rs:206` (`evaluate_world_goap`) is not
  dispatched. ~248 lines of legacy code.
- **One-line summary:** DEAD — delete.

---

## Reduction summary

| System                  | v1 class            | v2 class                   |
| ----------------------- | ------------------- | -------------------------- |
| battles                 | ESSENTIAL           | split: ESS damage + EMG attack |
| loot                    | ESSENTIAL           | EMERGENT + ESS spawn       |
| last_stand              | ESSENTIAL           | EMERGENT (ability)         |
| skill_challenges        | ESSENTIAL           | EMERGENT                   |
| dungeons                | ESSENTIAL           | EMERGENT + DERIVABLE-VIEW  |
| escalation_protocol     | EMERGENT-CANDIDATE  | DERIVABLE-VIEW             |
| threat                  | EMERGENT-CANDIDATE  | DERIVABLE-VIEW             |
| threat_clock            | ESSENTIAL           | DERIVABLE-VIEW             |
| adventuring             | ESSENTIAL           | EMERGENT                   |
| quests                  | ESSENTIAL           | EMERGENT                   |
| quest_lifecycle         | ESSENTIAL           | DERIVABLE-VIEW + EMERGENT  |
| seasonal_quests         | STUB                | DERIVABLE-VIEW             |
| bounties                | DUPLICATIVE         | EMERGENT + DERIVABLE-VIEW  |
| treasure_hunts          | EMERGENT-CANDIDATE  | EMERGENT                   |
| heist_planning          | EMERGENT-CANDIDATE  | EMERGENT                   |
| exploration (tile)      | EMERGENT-CANDIDATE  | DERIVABLE-VIEW             |
| exploration (voxel)     | ESSENTIAL           | ESSENTIAL (materialized view) |
| movement                | ESSENTIAL           | ESSENTIAL                  |
| pathfollow              | STUB                | DEAD                       |
| travel                  | DUPLICATIVE         | EMERGENT                   |
| monster_ecology         | ESSENTIAL           | EMERGENT + ESS spawn       |
| monster_names           | EMERGENT-CANDIDATE  | DERIVABLE-VIEW             |
| nemesis                 | EMERGENT-CANDIDATE  | DERIVABLE-VIEW             |
| wound_persistence       | ESSENTIAL           | DERIVABLE-VIEW             |
| adventurer_condition    | ESSENTIAL           | DERIVABLE-VIEW + EMERGENT  |
| adventurer_recovery     | ESSENTIAL           | DERIVABLE-VIEW + EMERGENT  |
| cooldowns               | ESSENTIAL           | DERIVABLE-VIEW             |
| death_consequences      | ESSENTIAL           | EMERGENT + ESS bequest     |
| sea_travel              | ESSENTIAL           | EMERGENT                   |
| scouting                | DUPLICATIVE         | EMERGENT                   |
| messengers              | DUPLICATIVE         | EMERGENT                   |
| goal_eval               | DEAD                | DEAD                       |
| world_goap              | DEAD                | DEAD                       |

**Counts:**

| Class                        | v1 | v2 |
| ---------------------------- | -- | -- |
| ESSENTIAL                    | 17 |  2 (movement, voxel-census materialization) |
| ESSENTIAL (mixed)            |  — |  5 (battles damage, loot spawn, monster_ecology spawn, death bequest, split systems) |
| EMERGENT                     |  — | 16 |
| EMERGENT + something         |  — |  5 |
| DERIVABLE-VIEW               |  — | 11 |
| DUPLICATIVE                  |  4 |  — (folded into EMERGENT) |
| EMERGENT-CANDIDATE           |  7 |  — (resolved to EMERGENT / DERIVABLE) |
| STUB                         |  4 |  — |
| DEAD                         |  — |  3 (pathfollow, goal_eval, world_goap) |

Irreducible mutation count drops from 17 to ~6 discrete event-driven
mutations (see bottom section).

---

## Required action vocabulary

### Core combat
- `Attack(target_id)`
- `Focus(target_id)` — reassign which enemy is "primary"
- `Flee(away_from)`
- `CastAbility(ability_id, target)`
- `PickUp(item_id)`
- `EquipItem(item_id, slot)`
- `ClaimBounty(target_id)`
- `AttemptChallenge(kind, difficulty)`

### Quests and parties
- `PostQuest(threat, reward, destination)`
- `AcceptQuest(quest_id)`
- `AbandonQuest(quest_id)`
- `CompleteQuest(quest_id)`
- `PostBounty(target, reward)`
- `PostSeasonalQuest(season, kind)`
- `FormParty`
- `JoinParty(party_id)`
- `LeaveParty`
- `ChooseQuestTarget(pos)`
- `EnterDungeon(d)`, `ExploreDepth()`, `ExitDungeon()`

### Travel, exploration, messages
- `Travel(destination)`
- `SailTo(coastal_dest)`
- `DisembarkAt(settlement)`
- `ObserveMarket(settlement)`
- `ShareReport(recipient)`
- `CarryMessage(from, to, payload)`
- `DeliverReport(report, recipient)`
- `StartTreasureHunt(destination)`
- `FindTreasureStep`
- `ReturnTreasure`

### Covert / high-skill
- `PlanHeist(target)`
- `Scout(target)`
- `Infiltrate(target)`
- `ExecuteHeist(target)`
- `AbortHeist`

### Life consequences
- `AttendFuneral(dead_id)`
- `InheritMastersTags(master_id)`
- `MourningRite(dead_id)`
- `UseMedicine(amount)`
- `Desert`

### Monsters as NPCs
- `MigrateToward(pos)`
- `SeekFood(settlement)`
- `FleeSettlement`
- `Breed(mate)`
- `AttackSettlement(s)`

---

## Required event types

### Essential (drive mutation)
- `DamageDealt { source, target, amount, kind }` — the primary physics
  input; `hp -= amount` flows from this
- `HealApplied { source, target, amount }`
- `ShieldApplied { source, target, amount }`
- `EntityDied { entity, cause, killer }` — flips `alive=false` on
  `hp ≤ 0`
- `EntitySpawned { entity, pos, cause }` — monsters from
  monster_ecology, parties from adventuring, items from loot
- `ItemDropped { from_entity, item_id, pos }`
- `ItemPickedUp { entity, item_id }`
- `BequestEvent { from, to, amount }` — gold conservation on death
- `PositionVisited { entity, tile }` — feeds tile exploration view
  (optional; can be derived from movement stream)
- `MovementApplied { entity, old_pos, new_pos }`
- `VoxelHarvested { pos, material, amount }` (essential for voxel
  mutation; triggered by EMERGENT `Harvest` action)

### Agentic (capture choices)
- `AttackEmitted { source, target }`
- `AbilityCast { caster, ability, targets }`
- `QuestPosted { settlement, quest }`
- `QuestAccepted { quester, quest }`
- `QuestCompleted { quest, party }`
- `QuestFailed { quest, reason }`
- `QuestProgressAdvanced { quest, delta }`
- `QuestDeadlinePassed { quest }`
- `PartyFormed { members, leader }`
- `PartyDisbanded`
- `PartyClash { winners, losers }`
- `DungeonEntered { party, dungeon }`
- `DungeonDepthExplored { dungeon, depth, party }`
- `RelicFound { finder, relic }`
- `BountyPosted`, `BountyClaimed { hunter, target, payout }`, `BountyExpired`
- `HeistAttempted { crew, target, outcome }`
- `HeistGoldStolen { target, amount, crew }`
- `TreasureStepFound { hunter, pos, reward }`
- `ArtifactDiscovered { hunter, artifact }`
- `VoyageStarted { entity, from, to }`
- `SeaMonsterEncounter { entity }`
- `MessageDelivered { carrier, from, to, payload }`
- `PriceReportObtained { observer, settlement }`
- `PriceReportShared { from, to, settlement }`
- `NpcDeserted { entity, from_settlement }`
- `MedicineConsumed { entity, amount, source }`
- `FuneralHeld { dead, attendees }`
- `MemorialRaised { dead, settlement }`
- `ApprenticeLineage { heir, master, tags }`
- `MonsterBred { parents, child }`
- `DenFormed { region }`

### Observer (for derived views / chronicle)
- `ThresholdCrossed { scope, metric, from, to }` — chronicle driver
- `MonsterBecameNamed { monster, name, kills }`
- `NemesisDesignated { monster, faction }`
- `NemesisSlain { slayer, nemesis }`
- `ChallengeAttempted { entity, kind, outcome }`
- `CellCensusComputed { cell }` — internal performance metric
- `DebuffsCleared { entity, cause }`
- `DungeonHazardApplied { target, amount }`
- `FoodConsumed { entity, amount }`

---

## Required derived views

- `threat(region) = clamp(density_pressure(region) + recent_attacks(region) − patrol_rate(region) − decay, 0, 100)`
- `global_threat_clock(tick) = base_growth × tick + acc(tick) − suppression(combat_npcs, alive_npcs)`
- `grid_fidelity(grid) = fidelity_rank_max(desired_from_threat, desired_from_hostile_count, desired_from_recent_kills)`
- `cooldown_remaining(npc, ability) = max(0, npc.cd_end_tick[ability] − now)`
- `ability_ready(npc, ability) = cooldown_remaining(npc, ability) == 0`
- `hp_effective(npc, now) = clamp(last_hp + ticks_since_damage(npc) × regen_rate(npc.activity), 0, max_hp)`
- `stress(npc, now) = clamp(base + Σ activity_drift(activity) × dt, 0, 100)`
- `fatigue(npc, now) = same pattern`
- `is_nemesis(m) = m.kind=Monster ∧ m.hostile ∧ m.level ≥ K`
- `nemesis_stats(m, now) = base + (now − designated_tick) × per_level_buff`
- `is_named(m) = kills(m) ≥ 3`
- `monster_name(m) = if is_named(m) then format(prefix, suffix, hash(m.id)) else entity_display_name(m)`
- `tile_explored(tile) = ∃ visit_event(tile)`
- `exploration_income(settlement) = count(alive_npcs within R) × bonus_rate`
- `cell_census(cell) = histogram(voxels in cell)` *(materialized for perf)*
- `known_resources(npc) = visible_cells(npc) ∩ tick_observed ≥ last_scan`
- `is_bounty_target(e) = (e.kind=Monster ∧ e.level ≥ K) ∨ (e.kind=Npc ∧ e.faction.hostile ∧ e.level ≥ 3)`
- `bounty_funding(region) = threat_level × funding_rate`
- `at_sea(npc) = traveling(npc) ∧ ¬nearby(npc, any_settlement, 30)`
- `coastal(settlement) = ∃ region : settlement in region ∧ region.is_coastal`
- `quest_board(settlement) = open_quests filtered by settlement.id`
- `at_quest_destination(npc) = |npc.pos − quest.dest| < 5`
- `quest_expired(q) = now > q.deadline`
- `quest_stale(q) = now − q.accepted_tick > 200 ∧ q.progress < 0.01`
- `eligible_adventurer(npc) = combat(npc) > 10 ∧ wants_adventure(npc) ∧ ¬in_party(npc)`
- `current_party(npc) = npc.party_id → party`
- `funeral_active(dead, now) = now − dead.death_tick ≤ 20`
- `is_heir(npc, dead) = npc.home_building_id == dead.home_building_id`
- `memorial_eligible(dead) = dead.level ≥ 10 ∧ chronicle_mentions(dead) ≥ 3`
- `traveling(npc) = npc.economic_intent in {Travel, Trade, Adventuring}`
- `fresh_report(npc, s) = ∃ report in npc.price_knowledge : now − report.tick_observed < 200`
- `dungeon_fidelity(d) = f(monster_density_nearby, party_present)`
- `dungeon_cleared(d) = d.explored_depth ≥ d.max_depth`
- `den_forming(region) = count(monsters within R) ≥ 5 for last T ticks`
- `settlement_under_attack(s) = region.monster_density > 80 ∧ recent_raid_rolled`
- `heist_skill(npc) = f(npc.level, behavior_value(stealth))`
- `heist_success_prob(npc, target) = f(heist_skill(npc), target.guard_strength)`
- `messages_in_transit(entity) = entity.carried_messages`

---

## Truly essential (irreducible) set in this batch

Only six kinds of mutation survive this pass:

1. **Damage application** — event-driven `hp -= amount` on `DamageDealt`
   (also `hp += amount` on `HealApplied`, `shield += amount` on
   `ShieldApplied`). The only way hp changes.
2. **Entity despawn (alive flag flip)** — on `EntityDied`, which fires
   when `hp ≤ 0` in damage application.
3. **Movement integration** — `pos += (dir × speed × dt) / tile_cost`
   in `movement.rs:advance_movement`. The sole position kernel.
4. **Entity spawn** — fires when monster_ecology emits `MonsterSpawned`
   (revive/breed) or adventuring emits relic-drop or loot emits
   `ItemDropped`. This is the `entities.push(...)` or Heal+SetPos
   recycle path.
5. **Voxel mutation** — `VoxelHarvested { pos, material, amount }`
   triggered by an EMERGENT `Harvest` action. The only way the voxel
   grid changes. (Not tracked by any of the 31 combat systems
   directly; it's the essential physics partner for the voxel census
   view.)
6. **Per-cell census materialization** — `state.cell_census[cell] =
   histogram(voxels in cell)` in `exploration.rs:scan_all_npc_resources`.
   Conceptually a derivable view; kept as essential because lazy eval
   would be 420K voxel reads per NPC per scan tick. Flag this one
   explicitly as "performance-driven materialization."

Plus one essential data-bookkeeping event:

7. **BequestEvent on death** — gold conservation requires the dead
   NPC's gold to flow to heirs / treasury atomically with the death.
   This is essential because any derivable alternative (heirs
   "discovering" the gold later) would leak gold across the timing
   gap.

Everything else in the 31-system batch reduces to EMERGENT NPC
actions plus DERIVABLE-VIEWS.
