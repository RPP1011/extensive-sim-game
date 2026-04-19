# World / Terrain / Utility Systems — Strict Reframing

## Methodology

This document re-classifies the 24 world/terrain/utility systems under a **strict event-sourcing + agentic-action** rubric. The v1 doc (`systems_world.md`) called 16 of 24 ESSENTIAL. Under the strict rubric, "essential" means **irreducible physics/bookkeeping**: a primitive the DSL cannot desugar away.

Four bins:

- **ESSENTIAL** — world physics primitives: voxel material write, tile write, structural collapse propagation, entity-id allocation, tick++, and the world's own stochastic event emission (the world-as-actor). Nothing an NPC can choose to do lives here.
- **EMERGENT** — anything that can be rephrased as "an NPC (or settlement leader) chose to do X, and X reduces to a sequence of essential writes". Construction, harvest, barter, migration, retirement, recruitment, class assumption, entering buildings, attending gatherings, lighting a signal, holding a festival — all emergent.
- **DERIVABLE-VIEW** — stored-as-cache values that are a pure function of other state: `season = tick/1200`, `weather = f(season, terrain, noise)`, `settlement.population = count(alive NPCs at s)`, `infrastructure_level = count(buildings at s)`, `occupants(b) = NPCs inside b`, `is_dead_zone = threat > 70`, `current_class = argmax(behavior_profile, thresholds)`. These require a view fn, not a system.
- **DEAD/STUB** — empty bodies or zero-work loops per v1 annotations (`class_progression.rs` noop, `guild_rooms.rs`/`guild_tiers.rs` treasury tax stubs, `weather.rs` stand-in). Delete or fold.

**Cited lines** trace back to `src/world_sim/systems/<file>.rs`.

---

## Per-system reanalysis

### seasons
- **Old:** MIXED (v1 essential)
- **New:** DERIVABLE-VIEW (with one sub-emission kept ESSENTIAL)
- **Reframe:** `current_season(tick) = (tick/1200) % 4` is a pure function (seasons.rs:53-62). Every derived table — `season_modifiers`, `food_production_mult`, `food_consumption_mult`, `price_pressure`, `wilderness_food_for_season` (seasons.rs:65-285) — is a constant lookup. The per-tick side effects are re-expressible: price drift every 50 ticks (seasons.rs:201-218) is a DERIVABLE price = exponential-moving-average toward seasonal pressure (no stored memory needed since it's `settlement.prices` that drifts, but that drift is mechanical, not agentic → move it into a pure "daily market close" view). Morale drift every 10 ticks (seasons.rs:221-233) and winter spoilage / autumn harvest (seasons.rs:169-197) are deterministic functions of (season, settlement). The only non-derivable emission is the `SeasonChanged` world event (seasons.rs:163-167) at transition boundaries — but even that is derivable (`tick % 1200 == 0`) so it's not even an event, just a clock tick downstream readers can pattern-match.
- **Required NPC actions:** none.
- **Required derived views:** `season(tick)`, `season_modifiers(season)`, `food_production_mult(season)`, `price_pressure(season) → prices`, `wilderness_food(terrain, season)`.
- **Required event types:** none (the `SeasonChanged` v1 event is redundant with `season(tick)`).
- **One-line summary:** The entire seasons system is a tick-indexed constant table; delete the system, keep the functions.

---

### weather
- **Old:** STUB/DEAD
- **New:** DEAD (or promote to a thin DERIVABLE-VIEW)
- **Reframe:** The file admits (weather.rs:62-64) "without active_weather on WorldState, we can only apply season-derived effects". `apply_winter_travel_penalty` is a literal no-op (weather.rs:90-98 "now handled by entity.move_speed_mult"). The only live path is `apply_ambient_threat_damage` (weather.rs:102-154), which is just "high-threat region → 0.1% chance damage nearby NPCs" — that belongs to dead_zones/terrain_events, not a weather system. The `WeatherEvent` struct (weather.rs:36-43) is never stored anywhere.
- **Required NPC actions:** none.
- **Required derived views:** `weather(region, tick) = f(season, region.terrain, hash(region.id, tick/7))` **if** weather becomes a real concept. Currently not a real concept.
- **Required event types:** none.
- **One-line summary:** Delete the file; if weather ever matters, it's `f(season, terrain, noise)`, not a system.

---

### dead_zones
- **Old:** EMERGENT-CANDIDATE with stub elements
- **New:** DERIVABLE-VIEW
- **Reframe:** Header comment (dead_zones.rs:43-49) openly proxies a missing `extraction_pressure`/`dead_zone_level` field. Constants `PRESSURE_THRESHOLD`, `SPREAD_RATE`, `RECOVERY_PRESSURE_THRESHOLD`, `PRESSURE_DECAY` (dead_zones.rs:18-28) are declared and never referenced in the function body. Functionally: `is_dead_zone(region) = region.threat_level > 70`. Entity damage (dead_zones.rs:72-78) and stockpile drain (dead_zones.rs:82-99) are `f(severity, proximity_hash)` — a reactive derivation, not state.
- **Required NPC actions:** none (the "damage" effect can be folded into a terrain_events-style world-as-actor emission, but v1 already reclassified that properly).
- **Required derived views:** `is_dead_zone(region) = region.threat_level > 70`, `dead_zone_severity(region) = (region.threat_level - 70) / 30`.
- **Required event types:** none.
- **One-line summary:** Delete; the single non-trivial predicate `threat > 70` is a query, not a system.

---

### terrain_events
- **Old:** ESSENTIAL (world disasters)
- **New:** ESSENTIAL (world-as-actor)
- **Reframe:** This is the canonical world-as-actor system: the world itself is an agent whose "intents" are volcanic eruptions (terrain_events.rs:33-37), floods (terrain_events.rs:38-42), avalanches (terrain_events.rs:43-47), forest fires (terrain_events.rs:48-59), cave collapses (terrain_events.rs:60-64), sandstorms (terrain_events.rs:65-69), corruption pulses (terrain_events.rs:70-74). Each emits `WorldRandomEvent` whose downstream effects (damage, stockpile drain, threat mutation) are mechanically derived from the event kind and region. So strictly speaking, the *emission* is essential (irreducible stochastic decision from the world-agent); the *resolution* is reducible to a derivation once the event is emitted. The `(region.id * 137.5).sin()` synthetic position hack (terrain_events.rs:92-94, 129-130, 184-185, 322-323) is load-bearing but wrong — should be `region.pos`.
- **Required NPC actions:** none — NPCs don't cause terrain events. They observe them.
- **Required derived views:** `terrain_event_damage_radius(event_kind) → f32`, `settlement_affected_by(event, settlement) → bool`.
- **Required event types:** `WorldRandomEvent { kind: Eruption|Flood|Avalanche|ForestFire|CaveCollapse|Sandstorm|CorruptionPulse, region_id, tick }`.
- **One-line summary:** The emission of the event is essential (world-as-actor); downstream damage/stockpile/threat mutations are derivations of the event.

---

### geography
- **Old:** DUPLICATIVE with terrain_events + seasons
- **New:** DERIVABLE-VIEW (delete the system)
- **Reframe:** The file applies "one-shot effects each tick that approximate the original system's gradual changes" (geography.rs:42-45) — it doesn't store any geography state. Each of the 5 branches (geography.rs:47-157) is conditional on `(region.threat, season, settlement.population)` and emits commodity deltas. ForestGrowth overlaps `seasons.food_production_mult` + `terrain_events.forest_fire`. RiverFlood overlaps `terrain_events.flood`. DesertExpansion overlaps `terrain_events.sandstorm`. RoadDegradation is just `Damage(entity, 1)` for NPCs near high-threat regions — already dead_zones. SettlementGrowth is a small treasury tip that is identical to `population.rs`'s tax income. No unique contribution.
- **Required NPC actions:** none.
- **Required derived views:** all its effects are already covered by `seasons(tick)` and `terrain_events` emissions.
- **Required event types:** none.
- **One-line summary:** Delete; it's a duplicate layer on seasons+terrain_events with synthetic region positions.

---

### signal_towers
- **Old:** ESSENTIAL (info diffusion)
- **New:** EMERGENT (NPC actions) + DERIVABLE-VIEW (coverage)
- **Reframe:** The tower is a Building entity; the "system" runs every 7 ticks and does two things: (1) battle damage — hostile entities near tower, 30% damage roll (signal_towers.rs:44-73) — this is just the normal combat loop applied to a building, not a tower-specific rule; (2) scouting — operational towers share price reports to friendly NPCs within range (signal_towers.rs:76-123). Price sharing is a derivable view: `price_knowledge(npc) = union of recent SharePriceReport events OR nearest_tower_coverage(npc.pos)`. The decision to *build* the tower is EMERGENT (settlement-leader action `BuildTower(loc)`). The decision to *light* the signal is EMERGENT (`LightSignal(tower)` — not implemented here but obvious extension). The automatic price-share on cadence should be an emission triggered by the tower's own "intent" (world-as-building-agent); or simply a derived view `sees_prices(npc, settlement) = ∃ operational_tower within R of both`.
- **Required NPC actions:** `BuildTower(loc)` (settlement leader), `LightSignal(tower)` (operator NPC).
- **Required derived views:** `tower_coverage(tower) = {(npc, settlement) : npc in R and settlement in R}`, `sees_prices(npc, settlement)` (replaces `SharePriceReport` broadcast).
- **Required event types:** `TowerBuilt { settlement, pos }`, `SignalLit { tower, theme }` (if we keep signaling as an emergent action).
- **One-line summary:** Tower construction is emergent; the price-share behavior is a derivable view over (operational_tower, settlement, npc) proximity.

---

### timed_events
- **Old:** ESSENTIAL but STUB-ish
- **New:** ESSENTIAL (world-as-actor)
- **Reframe:** World emits positive buffs (trade winds treasury bonus, meteor shower commodity boost, eclipse HoT, harvest moon food, faction summit heal, ancient portal all-commodity boost — timed_events.rs:74-150). Same pattern as `terrain_events`: the *emission* is the world-agent's intent (irreducible stochastic world pulse); the *resolution* is reducible. The `estimate_active_events` function (timed_events.rs:29-45) is a header-acknowledged proxy for missing `timed_events` storage on WorldState — when that storage is added, this becomes a storage-driven emission (still essential).
- **Required NPC actions:** none — NPCs don't cause cosmic events.
- **Required derived views:** `event_selection(tick) = f(tick_hash, season)`, `event_effect_amount(event, settlement) = f(event_kind, settlement.id)`.
- **Required event types:** `WorldRandomEvent { kind: TradeWinds|MeteorShower|Eclipse|HarvestMoon|FactionSummit|AncientPortal, tick }`.
- **One-line summary:** World-as-actor positive pulses; essential emission, derivable effect.

---

### random_events
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (world-as-actor)
- **Reframe:** Same pattern as `timed_events` but the pulses are mixed-sign: treasure (random_events.rs:35-44), harvest bounty (45-54), bandit raid (55-75), plague (76-100), famine (101-112), faction gift (113-122), equipment breakage (123-138), prophecy of doom (139-156), mercenary band (157-172). All are world-emitted; NPCs don't cause them. The String-allocating `Debuff { stat: "morale".to_string() }` (random_events.rs:148-149) is a perf hack, not a semantic issue.
- **Required NPC actions:** none.
- **Required derived views:** `event_selection(tick) = f(tick_hash)`, `event_target_selection(event, settlements) = f(tick)`.
- **Required event types:** `WorldRandomEvent { kind: Treasure|Bounty|Raid|Plague|Famine|Gift|Breakage|Doom|Mercs, tick, target: Option<SettlementId> }`.
- **One-line summary:** Siblings with timed_events — the two modules could merge into a single `WorldRandomEvent` emitter.

---

### voxel_construction
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC action) + ESSENTIAL (voxel write)
- **Reframe:** `advance_blueprint_construction` (voxel_construction.rs:40-115) is called from work.rs per worker tick, not on cadence. It's what an NPC *does* when its intent is "build this blueprint voxel": consume one commodity unit (voxel_construction.rs:81-85), write one voxel (voxel_construction.rs:101), flip `BlueprintVoxel.placed = true` (voxel_construction.rs:104-112). The decision to place this voxel belongs to the NPC (it's chosen via goal_stack/work assignment). The **voxel write itself** (`voxel_world.set_voxel`) is ESSENTIAL physics — that's the primitive. Everything else is EMERGENT NPC action. `attach_blueprint` (voxel_construction.rs:118-149) is settlement-leader-level action (deciding to seed a building footprint). `site_clearing_targets` (voxel_construction.rs:155-190) is a query/view.
- **Required NPC actions:** `PlaceVoxel(pos, material)` (the emergent atomic), `AttachBlueprint(building, blueprint)` (settlement leader), `ClearSite(building)` (derived target list).
- **Required derived views:** `site_clearing_targets(blueprint) → [voxels needing removal]`, `next_unplaced_voxel(blueprint) → Option<(idx, mat, offset)>`, `material_commodity(mat) → commodity_idx`.
- **Required event types:** `VoxelChanged { pos, from_mat, to_mat, by: entity_id, cause: Construction }` (so chroniclers can observe who built what).
- **One-line summary:** NPC chooses `PlaceVoxel`; the voxel write is the essential physics atom.

---

### voxel_harvest
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC action) + ESSENTIAL (voxel write + inventory add)
- **Reframe:** `harvest_tick` (voxel_harvest.rs:17-61) applies damage to the NPC's current `harvest_target`, removes the voxel when broken (voxel_harvest.rs:37), credits the yield to inventory (voxel_harvest.rs:43-45), and chains to the next adjacent voxel (voxel_harvest.rs:48-52). The *decision* to harvest is EMERGENT (NPC chose `HarvestVoxel(pos)` intent; `select_harvest_target` at voxel_harvest.rs:64-73 is the query the NPC uses). The voxel mutation (`voxel_world.mine_voxel`) is ESSENTIAL physics. The inventory write is ESSENTIAL bookkeeping.
- **Required NPC actions:** `HarvestVoxel(pos)` — with internal ticks of damage; under the hood this is a sequence of `MineVoxel(pos, dmg)` essentials.
- **Required derived views:** `find_nearest_harvestable(pos, material, r)`, `required_harvest_material(building_type)`, `yield_for(mat) → Option<(commodity, amount)>`.
- **Required event types:** `VoxelChanged { pos, from_mat, to_mat=Air, by, cause: Harvest }`, `CommodityHarvested { by, commodity, amount }` (optional — can be derived from VoxelChanged).
- **One-line summary:** NPC chooses `HarvestVoxel`; voxel removal + inventory credit are the essential atoms.

---

### construction
- **Old:** ESSENTIAL (room growth automaton)
- **New:** EMERGENT (NPC actions) + ESSENTIAL (tile writes)
- **Reframe:** Currently a cadence-driven automaton (construction.rs:25-91) that auto-grows rooms from `BuildSeed`s: flood-fills, computes interior/boundary, then inserts Floor/Wall/Door tiles. The **tile writes** (construction.rs:71-76, 125-129, 143-148) are essential physics. The **automaton logic** (when to expand vs close, stall detection, door placement) is exactly the kind of thing the DSL should compile away: NPCs choose `PlaceTile(pos, Floor)`, `PlaceTile(pos, Wall)`, `PlaceTile(pos, Door)` actions via `action_eval`, using the same flood-fill + door-position heuristics as *queries* (views), not as system mutations. `BuildSeed` becomes "this is an NPC's (or settlement's) open room-building project" — state on the agent, not a global list. Stall detection (construction.rs:41-49) becomes an agent giving up on a goal.
- **Required NPC actions:** `PlaceTile(pos, tile_type, material)`, `OpenRoomProject(seed_pos, min_size)`, `AbandonRoomProject(seed)`.
- **Required derived views:** `flood_fill(pos, tiles) → (interior, boundary)`, `is_enclosed(boundary, tiles)`, `has_door(boundary, tiles)`, `find_door_position(boundary, tiles)`, `detect_room_function(interior, tiles) → RoomFunction`.
- **Required event types:** `TilePlaced { pos, tile_type, material, by }`, `RoomClosed { seed, interior_size, function }`.
- **One-line summary:** NPC chooses `PlaceTile` using flood-fill views; the tile write is the essential atom — delete the automaton.

---

### structural_tick
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (physics)
- **Reframe:** This is irreducible world physics: "unsupported voxels fall". `structural_tick` (structural_tick.rs:23-79) BFS-walks every dirty chunk, anchors from ground/Granite/z<=0, removes any solid voxel not reached (structural_tick.rs:67-69), emits `StructuralEvent::FragmentCollapse` (structural_tick.rs:71-77). No NPC chooses collapse; it's a consequence of the *combined* voxel field. This is the only system in the batch that is genuinely, unambiguously ESSENTIAL under the strict rubric.
- **Required NPC actions:** none.
- **Required derived views:** `is_anchored(voxel, world)` (BFS from ground — the core algorithm).
- **Required event types:** `StructuralCollapse { chunk, affected_voxels, reason: Natural|Induced }`.
- **One-line summary:** The one irreducible physics system in this batch.

---

### interiors
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC actions) + DERIVABLE-VIEW (occupancy)
- **Reframe:** Two phases: (1) clear stale occupancies (interiors.rs:32-55) — iterates rooms, checks if `room.occupant_id`'s NPC still has `inside_building_id == this_building.id`; if not, clears. This is pure validation of a denormalized cache. (2) NPCs near their target building enter it, claim a room of kind matching their action (interiors.rs:58-177). Both are re-expressible: **occupancy is derivable** — `occupants(b) = [npc for npc in NPCs if inside_building_id == b.id]`; **room occupancy** is derivable — `room_occupant(b, ri) = argmin_distance(npcs_inside_b, room.offset)` or pick-one. The **act of entering** is an NPC action `EnterBuilding(b)` that flips `inside_building_id`. The **act of leaving** is `LeaveBuilding()`. The `(bi, ri) clears` vec at interiors.rs:32-55 is just a consistency sweep on a cache the DSL wouldn't store.
- **Required NPC actions:** `EnterBuilding(b)` (sets `inside_building_id`, `current_room`), `LeaveBuilding()` (clears both).
- **Required derived views:** `occupants(b)`, `room_occupant(b, room_idx)`, `preferred_room_kind(npc.action) → RoomKind`.
- **Required event types:** `EnteredBuilding { npc, building, room }`, `LeftBuilding { npc, building }`.
- **One-line summary:** Entering/leaving is an NPC action; room occupancy is a derived index over NPCs inside.

---

### social_gathering
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC actions)
- **Reframe:** `advance_social_gatherings` (social_gathering.rs:25-210) collects NPCs wanting to socialize, pair-matches them within 10u, starts a 15-tick `Socializing` action on both, boosts `needs.social +25`, accumulates DIPLOMACY/TEACHING tags, exchanges `HeardStory` beliefs about the most dramatic memory. Every one of these is re-expressible as NPC actions: `AttendGathering(loc)`, `ConverseWith(other)`, `ShareStory(other, story_id)`. The `social_npcs` collection (social_gathering.rs:32-46) is just a query for "NPCs wanting social". The O(N²) pair matching (social_gathering.rs:52-73) is choosing an action target. The `record_npc_event` calls (social_gathering.rs:89-101) are the side-effects of the conversation action. The memory/belief mutations (social_gathering.rs:134-192) are the effects of `ShareStory`.
- **Required NPC actions:** `AttendGathering(loc)`, `ConverseWith(other_npc)`, `ShareStory(target, story)` (action with side-effects on both).
- **Required derived views:** `wants_to_socialize(npc) = npc.needs.social < 40 || npc.goal_stack.has(Socialize)`, `best_story(npc) = argmax_impact(memory.events)`, `has_heard(npc, about) = belief.any(HeardStory{about})`.
- **Required event types:** `ConversationStarted { a, b, ticks }`, `StoryShared { teller, listener, about, impact }`.
- **One-line summary:** Gathering/conversing/storytelling are all NPC actions; the system is a scheduler the DSL can compile away.

---

### festivals
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** EMERGENT (settlement-leader action) + DERIVABLE-VIEW
- **Reframe:** The compiled behavior (festivals.rs:49-64) is literally "autumn → +5 FOOD, summer → +10 treasury, spring/winter → nothing". No actual festival state — no attendees, no theme, no duration. Could be deleted and folded into seasons. But the real DF-style festival is emergent: a settlement leader *declares* a festival with theme; NPCs *attend*; attendees gain morale; the settlement pays a cost. That's the DSL target: `HoldFestival(theme)` is a settlement-leader action that posts a `FestivalDeclared` event; NPCs can choose `AttendFestival(settlement)`; `festival_active(settlement) = ∃ FestivalDeclared in last N ticks`.
- **Required NPC actions:** `HoldFestival(theme)` (settlement leader), `AttendFestival(settlement)` (NPC).
- **Required derived views:** `festival_active(s) = ∃ FestivalDeclared{s} in [tick-N, tick]`, `festival_attendees(s, tick) = npcs attending`.
- **Required event types:** `FestivalDeclared { settlement, theme, start_tick, duration }`, `FestivalAttended { npc, festival }`.
- **One-line summary:** Current festivals is dead; real festivals are a settlement-leader emergent action with derivable "active" window.

---

### guild_rooms
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** 36 lines (guild_rooms.rs:1-36). `compute_guild_rooms_for_settlement` derives `room_count = min(treasury/200, 5)` and emits `UpdateTreasury(-room_count * 0.5)` (guild_rooms.rs:26-35). No guild_room state is stored. This is literally a treasury-indexed negative scalar; it's not guild-rooms, it's upkeep.
- **Required NPC actions:** none.
- **Required derived views:** if we actually want guild upkeep: `upkeep(s) = f(buildings_at_s)`; derive from `count(Building entities where entity.settlement_id == s.id)`.
- **Required event types:** none.
- **One-line summary:** Delete; zero guild-room state exists — it's just negative treasury drift.

---

### guild_tiers
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** 34 lines (guild_tiers.rs:1-34). Identical pattern to guild_rooms but positive: `tier = min(treasury/500, 5)`, emits `ProduceCommodity(FOOD, tier * 0.02)` (guild_tiers.rs:26-33). Tiny tip (<0.1 FOOD) indexed by treasury. No tier state stored.
- **Required NPC actions:** none.
- **Required derived views:** `guild_tier(s) = min(s.treasury / 500, 5)` — if needed anywhere, it's a view.
- **Required event types:** none.
- **One-line summary:** Delete; same dead-stub pattern as guild_rooms.

---

### migration
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC action)
- **Reframe:** `compute_migration` (migration.rs:63-168) scans settlements for high threat (>30) or low food (<5), picks the best reachable alternative (migration.rs:84-107), and for each NPC that *knows* the destination (migration.rs:127-133) and rolls <5% (migration.rs:136-139), emits `SetIntent(Travel{destination})` + optional chronicle. Every decision here is the NPC choosing to migrate: `MigrateTo(settlement)` is a pure NPC action. The eligibility ("this settlement has high threat") is an input the NPC observes, not a system rule. The migration chance is an NPC's internal decision threshold. Note: v1 observed that `home_settlement_id` reassignment isn't here — it's presumably handled on arrival; the proper EMERGENT model has `ArriveAtSettlement` set the new home.
- **Required NPC actions:** `MigrateTo(settlement)` (sets EconomicIntent::Travel to dest, later flips home_settlement_id on arrival).
- **Required derived views:** `settlement_attractiveness(s, regions)`, `migration_candidates(npc) = settlements npc.price_knowledge.contains within 100u`, `should_flee(settlement) = threat > 30 || food < 5`.
- **Required event types:** `MigrationEvent { npc, from, to, reason }`.
- **One-line summary:** Migration is a pure NPC action; the system is scheduling dressed up as essential.

---

### retirement
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC action) + ESSENTIAL (entity lifecycle)
- **Reframe:** `compute_retirement_for_settlement` (retirement.rs:149-222) scans NPCs eligible (level≥10, hp_ratio≥0.7, not on hostile grid), rolls 10% (retirement.rs:190-193), emits `Die` + `UpdateTreasury` + optional `UpdateStockpile(FOOD)` for Quartermaster (retirement.rs:203-220). The decision to retire is the NPC's (`Retire()`). The *effect* is entity lifecycle change (Die delta) + settlement bonus; Die is ESSENTIAL bookkeeping (entity transitions alive→dead), the treasury/stockpile bonus is just `UpdateTreasury` emission from the retire action itself. The legacy-type classification (retirement.rs:51-125) is a derived view on `class_tags`.
- **Required NPC actions:** `Retire()` (NPC decides); under the hood: `Die(self)` + `GrantLegacy(home_settlement, legacy_type)`.
- **Required derived views:** `legacy_type(npc.class_tags, hash) → LegacyType`, `legacy_bonus(legacy, level) → f32`, `retirement_eligible(npc) = level>=10 && hp_ratio>=0.7 && !on_hostile_grid`.
- **Required event types:** `Retired { npc, settlement, legacy }`, plus the underlying `EntityDied { id, cause: Retirement }`.
- **One-line summary:** NPC chooses `Retire()`; the Die delta is the essential lifecycle primitive.

---

### progression
- **Old:** ESSENTIAL by action, DERIVATIONAL by design
- **New:** DERIVABLE-VIEW
- **Reframe:** `compute_progression_for_settlement` (progression.rs:54-141) explicitly derives `entity.level = Σ class.level` (progression.rs:68) and back-computes the stat delta since last sync. The comment "XP removed — entity level derived from class levels" (progression.rs:1-10) confirms the derivation story. Stats could be computed on-demand from `npc.classes` every time they're read — no need to store the denormalized `entity.max_hp/attack/armor/move_speed`. Or store them as a cache rebuilt when classes change. Either way, there is no decision being made here; it's a projection from (classes, registry) → stats.
- **Required NPC actions:** none (leveling happens in the class matcher, not here).
- **Required derived views:** `entity_level(npc) = Σ class.level`, `entity_stats(npc, registry) = weighted_sum(class.per_level_stats)`.
- **Required event types:** `LeveledUp { npc, new_level, class }` (optional — observable projection).
- **One-line summary:** Delete; level and stats are pure functions of `npc.classes` — make them views.

---

### class_progression
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** `compute_class_progression_for_settlement` (class_progression.rs:27-54) filters NPCs with `behavior_sum >= 10.0` then the function body is literally one comment line — "XP removed — entity level derived from class levels" (class_progression.rs:52). Nothing is emitted. The module header (class_progression.rs:4-9) acknowledges class acquisition is handled by `ClassGenerator` trait elsewhere.
- **Required NPC actions:** if we add agentic class assumption: `AssumeClass(class)` when `behavior_profile` passes threshold.
- **Required derived views:** `current_class(npc) = class such that behavior_profile passes threshold` (if we want it; otherwise class is just stored).
- **Required event types:** `ClassAssumed { npc, class }`.
- **One-line summary:** Delete the file; emergent version is `AssumeClass(class)` NPC action keyed on behavior_profile view.

---

### recruitment
- **Old:** ESSENTIAL (structural pooling)
- **New:** EMERGENT (settlement-leader action) + ESSENTIAL (entity revive)
- **Reframe:** `compute_recruitment_for_settlement` (recruitment.rs:37-107) is the settlement-leader deciding to birth NPCs from the dead pool when food surplus + pop headroom allow: compute `target_births = min(8*growth_factor, food_surplus/0.3, 8)` (recruitment.rs:61-71), scan dead NPCs at settlement (recruitment.rs:79-85), then unaffiliated pool (recruitment.rs:88-97), revive each via `Heal(200) + SetPos + SetIntent(Produce)` (recruitment.rs:109-130). This is `Recruit(n)` settlement-leader action. The *revival* is ESSENTIAL bookkeeping — flipping a dead slot to alive is an irreducible entity-lifecycle primitive (inverse of Die). The *decision to recruit* (how many, which dead to pick) is EMERGENT.
- **Required NPC actions:** `Recruit(n)` (settlement leader, bounded by food/pop/dead-pool).
- **Required derived views:** `alive_at(s)`, `food_surplus(s)`, `growth_factor(alive, capacity) = max(1 - alive/500, 0.05)`, `dead_pool(s)`, `unaffiliated_dead_pool()`.
- **Required event types:** `RecruitmentEvent { settlement, revived_npc, source: OwnDead|Unaffiliated }`, plus the underlying `EntityRevived { id, pos }`.
- **One-line summary:** Settlement leader chooses `Recruit`; the revive (dead→alive flip) is the essential bookkeeping primitive.

---

### npc_decisions
- **Old:** ESSENTIAL (barter + trade completion)
- **New:** EMERGENT (NPC actions)
- **Reframe:** Comment at line 130 openly states "NPC action scoring now handled by action_eval.rs. Only trade arrival and barter remain here." The trade-arrival path (npc_decisions.rs:40-128) is the NPC's `CompleteTrade` action: arrive within 3u of dest, dump all inventory at dest prices, earn gold, record trade, set back to Produce. The barter path (npc_decisions.rs:152-265) pairs producers of different commodities at same settlement, swaps at local prices bounded by carry capacity — that's `BarterWith(other_npc, give_commodity, want_commodity, amount)`. Both are pure NPC actions with derivable preconditions. The `producers: [(u32,usize,f32); 64]` stack array (npc_decisions.rs:165-180) is a candidate-list view, not state.
- **Required NPC actions:** `CompleteTrade(dest)` (sell all carried at dest prices → earn gold → return to Produce), `BarterWith(other_npc, give, want, amount)` (commodity swap at local prices).
- **Required derived views:** `arrived_at(npc, settlement) = dist(npc.pos, settlement.pos) < 3`, `carry_capacity(npc) = level*5+10`, `total_carried(npc)`, `barter_candidates(settlement) = NPCs with production[0] at s`, `fair_ratio(a_commodity, b_commodity, prices) in 0.33..3`.
- **Required event types:** `TradeCompleted { npc, home, dest, profit }`, `BarterCompleted { a, b, a_gave, b_gave }`.
- **One-line summary:** Trade completion and barter are NPC actions; delete the system, move the logic into `action_eval` with views.

---

### population
- **Old:** MIXED (essential per-storage)
- **New:** DERIVABLE-VIEW (the settlement.population scalar)
- **Reframe:** `compute_population_for_settlement` (population.rs:56-193) is pure scalar-drift math on `settlement.population`: food consumption `= pop * 0.001 * cap(stockpile)` (population.rs:77-88), growth rate from `(food_ratio, treasury, threat)` (population.rs:90-128), clamped by `MIN_POPULATION` / `MAX_POPULATION` (population.rs:141-147), tax income `= pop * 0.001` (population.rs:157-163), surplus stockpile on growth (population.rs:167-177), decline stockpile drain (population.rs:181-192). The key insight: **`settlement.population` as a stored scalar is a redundant view over `count(alive NPCs with home_settlement_id == s.id)`**. In a pure-event model, delete the scalar; derive it from the NPC roster. The food-consumption, tax, and stockpile-drift emissions then hang off *NPC lifecycle events* (Born, Died, MigrateTo) rather than a scalar drift system. Birth/death are handled by `recruitment` and combat.
- **Required NPC actions:** none (births/deaths are Recruit/Die).
- **Required derived views:** `population(s) = count(alive NPCs with home_settlement_id == s.id)`, `food_demand(s) = population(s) * FOOD_PER_POP`, `tax_income(s) = population(s) * TAX_PER_POP`, `growth_shape(food_ratio, treasury, threat, pop, capacity) → f32` (only used to decide Recruit rate — so it moves to recruitment).
- **Required event types:** none (all derivable from per-NPC events).
- **One-line summary:** Delete the scalar drift; population is `count(NPCs at s)`; food/tax/stockpile effects attach to NPC lifecycle events instead.

---

## Reduction summary

| System | Old | New |
|---|---|---|
| seasons | MIXED | DERIVABLE-VIEW |
| weather | STUB/DEAD | DEAD |
| dead_zones | EMERGENT-CAND (stubby) | DERIVABLE-VIEW |
| terrain_events | ESSENTIAL | ESSENTIAL (world-as-actor) |
| geography | DUPLICATIVE | DERIVABLE-VIEW (delete) |
| signal_towers | ESSENTIAL | EMERGENT + DERIVABLE-VIEW |
| timed_events | ESSENTIAL | ESSENTIAL (world-as-actor) |
| random_events | ESSENTIAL | ESSENTIAL (world-as-actor) |
| voxel_construction | ESSENTIAL | EMERGENT + ESSENTIAL (voxel write) |
| voxel_harvest | ESSENTIAL | EMERGENT + ESSENTIAL (voxel write + inv) |
| construction | ESSENTIAL | EMERGENT + ESSENTIAL (tile write) |
| structural_tick | ESSENTIAL | ESSENTIAL (physics) |
| interiors | ESSENTIAL | EMERGENT + DERIVABLE-VIEW |
| social_gathering | ESSENTIAL | EMERGENT |
| festivals | EMERGENT-CAND | EMERGENT + DERIVABLE-VIEW |
| guild_rooms | STUB/DEAD | DEAD |
| guild_tiers | STUB/DEAD | DEAD |
| migration | ESSENTIAL | EMERGENT |
| retirement | ESSENTIAL | EMERGENT + ESSENTIAL (Die) |
| progression | ESSENTIAL-by-action | DERIVABLE-VIEW |
| class_progression | STUB/DEAD | DEAD |
| recruitment | ESSENTIAL | EMERGENT + ESSENTIAL (revive) |
| npc_decisions | ESSENTIAL | EMERGENT |
| population | MIXED | DERIVABLE-VIEW |

**Tallies (strict rubric):**
- **ESSENTIAL (pure irreducible physics/bookkeeping, no overlap):** 1 — `structural_tick`.
- **ESSENTIAL (world-as-actor emitters):** 3 — `terrain_events`, `timed_events`, `random_events`.
- **EMERGENT-primarily (with an essential atom underneath):** 9 — `signal_towers`, `voxel_construction`, `voxel_harvest`, `construction`, `interiors`, `social_gathering`, `festivals`, `migration`, `retirement`, `recruitment`, `npc_decisions` (actually 11; the primary intent is emergent, even if a Die/Revive/SetVoxel/SetTile primitive is cited).
- **DERIVABLE-VIEW (delete, replace with function):** 6 — `seasons`, `dead_zones`, `geography`, `progression`, `population`, plus the cache-layer of `interiors` and the "active" window of `festivals`.
- **DEAD (stubs to delete):** 4 — `weather`, `guild_rooms`, `guild_tiers`, `class_progression`.

vs. v1: 16 ESSENTIAL → strict 4 (only `structural_tick` + 3 world-as-actor emitters genuinely essential).

---

## Required action vocabulary

### NPC actions (commoner)

- `PlaceVoxel(pos, material)` — consume 1 unit of matching commodity, write voxel. (voxel_construction)
- `HarvestVoxel(pos)` — damage voxel, on break: add yield to inventory, chain to neighbor. (voxel_harvest)
- `PlaceTile(pos, tile_type, material)` — write into `tiles` map. (construction)
- `EnterBuilding(b)` — set `inside_building_id`, `current_room`. (interiors)
- `LeaveBuilding()` — clear both. (interiors)
- `AttendGathering(loc)` — walk to social building. (social_gathering)
- `ConverseWith(other)` — start 15-tick Socializing action. (social_gathering)
- `ShareStory(target, story)` — create `HeardStory` belief on target. (social_gathering)
- `AttendFestival(settlement)` — opt-in gain morale boost. (festivals)
- `MigrateTo(settlement)` — set EconomicIntent::Travel; on arrival flip home_settlement_id. (migration)
- `Retire()` — flip to Die + grant legacy bonus. (retirement)
- `AssumeClass(class)` — when behavior_profile passes threshold. (class_progression)
- `CompleteTrade(dest)` — dump inventory at dest prices, earn gold, return to Produce. (npc_decisions)
- `BarterWith(other_npc, give, want, amount)` — swap commodities at local prices with carry-cap check. (npc_decisions)
- `LightSignal(tower)` — broadcast an observable signal; observers learn something. (signal_towers)

### Settlement-leader actions

- `Recruit(n)` — revive n dead NPCs from own pool, then unaffiliated; consume food. (recruitment)
- `HoldFestival(theme)` — post FestivalDeclared event; pay treasury cost; attendees gain morale. (festivals)
- `BuildTower(loc)` — seed a tower Building entity with blueprint at loc. (signal_towers)
- `AttachBlueprint(building, blueprint)` — seed voxel construction. (voxel_construction)
- `GrantLegacy(legacy_type)` — receive side-effect of an NPC's Retire(). (retirement)

---

## Required event types

Events the world actually needs (primitive emissions, not derivations):

- `TilePlaced { pos, tile_type, material, by }` — essential tile-write event.
- `VoxelChanged { pos, from_mat, to_mat, by, cause: Construction|Harvest|Collapse|Event }` — essential voxel-write event.
- `StructuralCollapse { chunk, affected_voxels, reason: Natural|Induced }` — physics event.
- `EntityDied { id, cause: Combat|Retirement|Starvation|... }` — essential lifecycle event.
- `EntityRevived { id, pos, by_settlement }` — essential lifecycle event (inverse of Die).
- `MigrationEvent { npc, from, to, reason }` — observable NPC-action event.
- `FestivalDeclared { settlement, theme, start_tick, duration }` — settlement-leader-action event.
- `FestivalAttended { npc, festival }` — NPC-action event.
- `RecruitmentEvent { settlement, revived_npc, source }` — NPC-lifecycle event.
- `EnteredBuilding { npc, building, room }` / `LeftBuilding { npc, building }` — NPC-action events.
- `ConversationStarted { a, b, ticks }` / `StoryShared { teller, listener, about, impact }` — NPC-action events.
- `Retired { npc, settlement, legacy }` — NPC-action event (paired with EntityDied).
- `ClassAssumed { npc, class }` — NPC-action event (when AssumeClass is chosen).
- `TradeCompleted { npc, home, dest, profit }` / `BarterCompleted { a, b, a_gave, b_gave }` — NPC-action events.
- `TowerBuilt { settlement, pos }` / `SignalLit { tower, theme }` — settlement/NPC events.
- `WorldRandomEvent { kind, target?, tick }` — world-as-actor event (unifies terrain_events + timed_events + random_events under one emission stream with a kind discriminator).

**Explicitly NOT events** (they're derivations, not emissions):

- `Season` / `SeasonChanged` — derivable from tick.
- `Weather` — derivable from (season, terrain, noise) if kept at all.
- `PopulationChanged` — derivable from alive-NPC count at settlement.
- `DeadZoneFormed` — derivable from `threat > 70`.
- `LeveledUp` — derivable from sum(class.level) comparison (can still be observed, but doesn't need to be an event — the NPC's class.level change already is the event).

---

## Required derived views

```
season(tick) = (tick / 1200) % 4
season_modifiers(season) -> {travel_speed, supply_drain, threat, recruit_chance, morale_per_tick}
food_production_mult(season), food_consumption_mult(season)
price_pressure(season) -> [f32; 8]
wilderness_food(terrain, season) -> f32

weather(region, tick) = f(season, region.terrain, hash(region.id, tick/7))   # if needed

is_dead_zone(region) = region.threat_level > 70
dead_zone_severity(region) = max(0, (region.threat_level - 70) / 30)

population(s) = count(alive NPCs with home_settlement_id == s.id)
food_demand(s) = population(s) * 0.001
tax_income(s) = population(s) * 0.001
growth_shape(food_ratio, treasury, threat, pop, cap) -> f32    # moved into recruitment eligibility

infrastructure_level(s) = count(Building entities with settlement_id == s.id)
occupants(b) = [npc for npc in NPCs if inside_building_id == b.id]
room_occupant(b, ri) = <deterministic pick from occupants(b) at that room>
preferred_room_kind(action) -> RoomKind

entity_level(npc) = sum(class.level for class in npc.classes)
entity_stats(npc, registry) = weighted_sum(class.per_level_stats)
current_class(npc, thresholds) = argmax class such that behavior_profile passes
legacy_type(class_tags, hash) -> LegacyType
legacy_bonus(legacy, level) -> f32
retirement_eligible(npc) = level >= 10 && hp/max_hp >= 0.7 && !on_hostile_grid

festival_active(s) = exists FestivalDeclared{s} in (tick - duration .. tick]

tower_coverage(tower) = set of settlements/NPCs within R
sees_prices(npc, settlement) = exists operational_tower with npc and settlement in range
               (replaces SharePriceReport broadcast)

is_anchored(voxel, world) = BFS from ground reaches voxel via solid-face-adjacent chain
site_clearing_targets(blueprint) -> [non-terrain solid voxels in footprint]
next_unplaced_voxel(blueprint) -> Option<(idx, mat, offset)>
material_commodity(mat) -> commodity_idx
yield_for(mat) -> Option<(commodity, amount)>
find_nearest_harvestable(pos, material, r)
required_harvest_material(building_type)

flood_fill(pos, tiles) -> (interior, boundary)
is_enclosed(boundary, tiles), has_door(boundary, tiles), find_door_position(boundary, tiles)
detect_room_function(interior, tiles) -> RoomFunction

settlement_attractiveness(s, regions)
migration_candidates(npc) = settlements npc knows within 100u
should_flee(s) = threat > 30 || food < 5

wants_to_socialize(npc) = needs.social < 40 || goal_stack.has(Socialize)
best_story(npc) = argmax_impact(memory.events filtered to dramatic)
has_heard(npc, about) = beliefs.any(HeardStory{about})

carry_capacity(npc) = level * 5 + 10
total_carried(npc)
barter_candidates(settlement) = NPCs with production[0] at s
fair_ratio(a_c, b_c, prices) in 0.33..3
```

---

## Truly essential (irreducible) set in this batch

The minimum set of primitives the DSL must provide so everything else desugars cleanly:

1. **Voxel material write** — `set_voxel(pos, material)`. Used by harvest, construction, collapse, terrain-events. (voxel_harvest.rs:37, voxel_construction.rs:101, structural_tick.rs:68)
2. **Tile write** — `set_tile(pos, tile)`. Used by construction. (construction.rs:71-76, 125-129, 143-148)
3. **Structural collapse propagation** — "unsupported solids fall" BFS from anchors. Irreducible combinational physics. (structural_tick.rs:23-79, 101-250)
4. **Entity ID allocation / revive** — flipping `alive=false → alive=true` and `alive=true → alive=false`. (recruitment.rs:109-130 revive; retirement.rs:203 die). Die is a projection of HP<=0 in normal play, but as a lifecycle atom it's irreducible — identity persists, state bit flips.
5. **Tick++** — advancing `state.tick`. All cadences and derivations depend on it.
6. **World-random-event emission** — the world-as-actor's stochastic intents (terrain, timed, random). Irreducible because the choice "does volcano erupt this tick" is not an agent decision and not a pure state function (it's a draw from the RNG gated by state).

Everything else in this batch is either an NPC/settlement-leader emergent action built on these primitives, or a derivable view over state.

---

## Notes on re-classification

- v1 called `voxel_construction`, `voxel_harvest`, `construction` ESSENTIAL at the system level. Strict: the *system* is EMERGENT (NPC choosing what to build/harvest); only the underlying `set_voxel` / `set_tile` *atom* is essential.
- v1 called `npc_decisions`, `migration`, `retirement`, `recruitment`, `interiors`, `social_gathering`, `signal_towers` ESSENTIAL. Strict: each is an NPC or settlement-leader decision + effect, fully emergent, with at most a Die/Revive primitive touching the lifecycle atom.
- v1 had `progression` straddling; strict: it's a pure derivation from class levels — delete the system, add a view.
- v1 had `population` MIXED (scalar is state); strict: delete the scalar, derive from NPC count. This collapses the two-layer population abstraction that v1 flagged as a load-bearing pattern.
- `seasons.rs` in v1 had essential emissions (winter spoilage, autumn harvest, festival morale, morale drift, price drift). Strict: every one of these is `f(season, settlement_id)` and season is `f(tick)` — they're all derivations. The only reason they exist as deltas is because in the current architecture, "settlement state changes" must flow through the delta pipeline. A view-based architecture reads `settlement.food_at_tick(t) = base_food - winter_loss(t) + autumn_bonus(t)` directly.
- Unified `WorldRandomEvent` stream: `terrain_events` + `timed_events` + `random_events` all emit the world-agent's stochastic intents. Their handlers differ but the emission is one primitive event type with a kind discriminator. Three files can fuse into one.
