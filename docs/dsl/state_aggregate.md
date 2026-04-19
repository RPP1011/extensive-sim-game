# Aggregate-Level State Catalog

`Group` is the universal social-collective primitive. Every settlement, faction, family, guild, religion, hunting pack, criminal cabal, adventuring party, court, league, or monastery is a `Group` distinguished by its `kind` discriminator. The `Group` section at the end of this doc gives the canonical shape; the per-kind sections that follow describe the additional fields each kind carries.

## Settlement (Group with `kind=Settlement`)
Political, economic, and structural hub for agents. Carries stockpiles, prices, treasury, population, group affiliations, and a fixed location with facilities.

### Identity & Metadata
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique settlement ID | spawn/init | all settlement-aware systems |
| name | String | Display name | init | UI, chronicle, logs |
| pos | vec3 | World-space position. 2D footprint center; `pos.z` derived as `surface_height(pos.xy)` at init. [OPEN] whether multi-level settlements (underground + aboveground) warrant storing a per-layer z. | init | all systems using settlement location |
| grid_id | Option<u32> | NavGrid this settlement owns | init | navigation, pathfinding |
| specialty | SettlementSpecialty | Production/npc focus (Mining, Trade, Farming, Military, Scholar, Port, Crafting) | init | production, npc_spawn, resource nodes |

### Economy
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| treasury | f32 | Gold reserves in coffers | taxes, economy, contracts, looting, theft, conquest, betrayal | economy, contracts, warfare, bankruptcy |
| stockpile | [f32; 8] | Commodity reserves (FOOD, WOOD, IRON, COPPER, HERBS, CRYSTAL, HIDE, MEDICINE, EQUIPMENT) | production, consumption, trade, looting | economy, npc_decisions, trade, production |
| prices | [f32; 8] | Local market prices per commodity | trade_goods, price_controls, seasons, commodity_futures, currency_debasement | economy, trades, npc_decisions, arbitrage |

### Population & Morale
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| population | u32 | Count of alive NPCs with home_settlement_id == id | npc birth/death, migration | economy scaling, resource consumption, threat assessment |

### Politics & Security
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| faction_id | Option<u32> | Owning faction, if controlled | conquest, civil_war, diplomacy | faction_ai, warfare, diplomacy, taxes |
| threat_level | f32 | Danger score (0–1) from nearby monsters, recent losses | monster_density (regional), quest_posting, attacks, threat_scaling | quest_posting urgency, recruitment, building upgrades |
| infrastructure_level | f32 | Building/defense tier (0–5) | construction, upgrades | defense strength, garrison capacity, building work |

### Organization
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| context_tags | Vec<(u32, f32)> | Contextual action modifiers (e.g., plague, festival, war) | events (crisis, seasons, diplomacy) | action_system, skill resolution |
| treasury_building_id | Option<u32> | Entity ID of Treasury building | init (ensure_treasury_buildings) | resource movement, gold transfers |
| service_contracts | Vec<ServiceContract> | Active contracts posted by settlement NPCs | npc_decisions, contract_lifecycle | npc_decisions, contract resolution |
| construction_memory | ConstructionMemory | Per-settlement building event history (short/medium/long-term) | building_ai event logging | building_ai pattern learning |

---

## RegionState
Territorial and environmental layer. Represents large map areas with terrain type, monster populations, faction control, and dungeon sites.

### Identity & Geographic
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique region ID | spawn | all systems referencing regions |
| name | String | Display name | spawn | UI, chronicle |
| pos | vec3 | World-space center. Synthesized at init from the sin/cos layout formula with `z = sea_level` (regions don't have an interior). | spawn | region queries, travel distance |
| terrain | Terrain | Biome type (Plains, Forest, Mountains, Coast, Swamp, Desert, Tundra, Volcano, DeepOcean, Jungle, Glacier, Caverns, Badlands, FlyingIslands, DeathZone, AncientRuins, CoralReef) | spawn | resource spawning, threat, travel speed |
| sub_biome | SubBiome | Terrain variant (Standard, LightForest, DenseForest, AncientForest, SandDunes, RockyDesert, HotSprings, GlowingMarsh, TempleJungle, NaturalCave, LavaTubes, FrozenCavern, MushroomGrove, CrystalVein, Aquifer, BoneOssuary) | spawn | resource yields, threat mods, travel speed |
| elevation | u8 | Terrain height tier (0–4: sea level, foothills, highlands, peaks, summit/sky) | spawn | resource rarity, threat, travel, building placement |

### Ecology
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| monster_density | f32 | Local monster spawn rate multiplier | random_events, monster_spawning, threat_scaling | encounter generation, threat_level calc |
| threat_level | f32 | Current aggregate danger (0–1, accounting for monster density, elevation, recent battles) | monster density, battles, time decay | threat_reports, quest_posting, npc decisions |
| has_river | bool | Whether a river flows through this region | spawn | river_travel, trade_route generation |
| has_lake | bool | Whether a lake exists here | spawn | water-dependent resources, water travel |
| is_coastal | bool | Whether region borders ocean (map edge or adjacent DeepOcean) | spawn | coastal trade, sea creature encounters |

### Dungeon Sites
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| dungeon_sites | Vec<DungeonSite> | Entrances to procedural dungeons (pos, name, explored_depth, max_depth, is_cleared, last_explored_tick, threat_mult) | dungeon_discovery, exploration | quest_posting, adventuring |

### Connectivity
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| neighbors | Vec<u32> | Region IDs of 4-connected grid neighbors | spawn | travel, npc_movement, region_graphs |
| river_connections | Vec<u32> | Region IDs connected by river | spawn | river_travel, trade routes |
| is_chokepoint | bool | True if only 1–2 passable neighbors (strategic) | spawn | warfare, blockade tactics, npc_pathing |
| is_floating | bool | True if FlyingIslands terrain (requires special access) | spawn | access control, encounter generation |

### Politics & Conflict
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| faction_id | Option<u32> | Controlling faction, if any | conquest, civil_war | faction_ai, warfare, taxes, recruitment |
| control | f32 | Faction control strength (0–1) | diplomacy, conquest, unrest | faction power assessment |
| unrest | f32 | Civil unrest (0–1, fuels rebellion) | diplomacy, oppression, riots | civil_war triggers, faction loyalty |

---

## Faction (Group with `kind=Faction`)
Macro-political collective representing governments, military powers, organized cults. Carries military_strength + standings to other groups + tech_level + governance.

### Identity & Relationships
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique faction ID | spawn | all faction-aware systems |
| name | String | Faction name | init | UI, chronicle, logs |
| relationship_to_guild | f32 | Relationship to player guild (-100 to 100) | diplomacy, quest_completion, betrayals | quest_board filtering, npc morale, diplomacy |
| at_war_with | Vec<u32> | List of faction IDs in open warfare | civil_war, diplomacy | warfare, settlement conquest, movement restrictions |

### Military
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| military_strength | f32 | Current fighting capacity (troops, equipment, morale) | recruitment, training, losses, desertion | conquest, diplomacy, battle strength |
| max_military_strength | f32 | Theoretical max capacity from population | population growth, tech_level | scaling, recruitment limits |

### Territory & Resources
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| territory_size | u32 | Count of regions/settlements controlled | conquest, loss_of_control | power assessment, tax base, recruitment |
| treasury | f32 | Faction gold reserves | taxes (from settlements), wars (looting), quests, reparations | military building, mercenary recruitment, diplomacy |

### Politics & Stability
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| diplomatic_stance | DiplomaticStance | Toward another faction (Friendly, Neutral, Hostile, AtWar, Coalition) | diplomacy system | npc decisions, interaction modifiers |
| coup_risk | f32 | Stability metric (0–1, higher = more likely internal upheaval) | oppression, morale, succession events | faction_ai decisions, succession crisis triggers |
| escalation_level | u32 | Conflict intensity (0–5, caps NPC conflict actions) | warfare, diplomacy escalation | warfare intensity cap, treaty enforcement |
| tech_level | u32 | Research/development tier | research system, quests | military bonus, production speeds, ability unlock |
| recent_actions | Vec<String> | Bounded log of recent faction events | all major systems | narrative, NPC knowledge, diplomacy memory |

---

## GuildState
Player faction state — independent from NPC factions.

### Resources
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| gold | f32 | Guild treasury | quest_rewards, missions, expenses | quest_board filtering, upgrades, expenses |
| supplies | f32 | Supply reserve (used for missions) | production, usage, looting | mission readiness, supply checks |

### Reputation & Capacity
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| reputation | f32 | Fame/standing (0–100) | quests, battles, diplomacy | quest_board filtering, npc_recruitment, prices |
| tier | u32 | Guild level (0–5, unlocks features) | reputation milestones | mission generation, capacity unlocks |
| credit_rating | f32 | Borrowing capacity (0–100) | loans, repayment | loan eligibility, interest rates |
| active_quest_capacity | u32 | Max simultaneous active quests | tier unlocks | quest acceptance checks |

---

## Quest & Quest Lifecycle

### Quest (Active)
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

### QuestType Enum
`Hunt`, `Escort`, `Deliver`, `Explore`, `Defend`, `Gather`, `Rescue`, `Assassinate`, `Diplomacy`, `Custom` — tag for behavior mechanics.

### QuestStatus Enum
- `Traveling`: party moving to destination
- `InProgress`: at location, pursuing objective
- `Completed`: objective achieved
- `Failed`: party wiped or deadline passed
- `Returning`: returning to home settlement

### QuestPosting (Board)
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

## FidelityZone
Proximity bubble controlling simulation fidelity level (rich NPC detail vs. statistical abstraction).

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique zone ID | spawn | fidelity_control system |
| fidelity | Fidelity | Richness level enum (High, Medium, Low) | fidelity_control | entity update frequency, detail depth |
| center | vec3 | Zone center (usually settlement pos) | fidelity_control, settlement movement | zone queries |
| radius | f32 | Zone extent in world units | fidelity_control, threat radius | proximity checks |
| entity_ids | Vec<u32> | Entities currently in this zone | entity_movement, spawn/despawn | fidelity state sync |

---

## TradeRoute
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

## ServiceContract
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

## PriceReport
Historical price snapshot for arbitrage and market analysis.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| settlement_id | u32 | Which settlement | price_discovery | npc_trade arbitrage |
| prices | [f32; 8] | Prices at snapshot time | settlement price state | npc_trade profit calc |
| tick_observed | u64 | When recorded | price_discovery | stale check, trend calc |

---

## SettlementSpecialty Enum
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

## DungeonSite
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

## DiplomaticStance Enum
- `Friendly`: +morale, trade bonuses
- `Neutral`: standard interactions
- `Hostile`: -morale, embargo possible
- `AtWar`: open conflict, unit restrictions
- `Coalition`: shared military benefits

---

## Terrain Enum (17 types)
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

## SubBiome Enum (16 types)
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

## EconomyState
Global economic aggregate.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| total_gold_supply | f32 | Sum of all gold (guild + factions + settlements) | economy system | inflation, market sentiment |
| total_commodities | [f32; 8] | Total supply of each commodity | production, consumption, trades | resource scarcity calc, price scaling |

---

## ChronicleEntry
Narrative log entry for historical record.

| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tick | u64 | When event occurred | event logging | timeline, quest narratives |
| category | ChronicleCategory | Event type (Battle, Quest, Diplomacy, Economy, Death, Discovery, Crisis, Achievement, Narrative) | event logging | filtering, narrative theming |
| text | String | Human-readable description | event logging | UI, lore |
| entity_ids | Vec<u32> | Involved entity IDs | event logging | relationship tracking |

---

## WorldEvent Enum (13 variants)
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

## WorldState Top-Level Fields (not covered by per-entity docs)

### Time & RNG
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tick | u64 | Current simulation tick | time system | all systems (time-based logic) |
| rng_state | u64 | Deterministic RNG seed state | next_rand_u32() | all probabilistic systems |

### Indexes & Caches
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| entity_index | Vec<u32> | entity_id → index into entities/hot/cold (rebuilt) | rebuild_entity_cache() | O(1) entity lookups |
| max_entity_id | u32 | Highest entity ID seen (sizes entity_index) | rebuild_entity_cache() | index allocation |
| next_id | u32 | Monotonic entity ID counter | next_entity_id() | entity creation |
| group_index | GroupIndex | Contiguous ranges by (settlement_id, party_id) for batch iteration | rebuild_group_index() | settlement-scoped loops |
| settlement_index | Vec<u32> | settlement_id → index into settlements (rebuilt) | rebuild_settlement_index() | O(1) settlement lookups |

### Spatial
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tiles | HashMap<TilePos, Tile> | Sparse tile modifications (walls, floors, ditches, farmland) | building/construction systems | pathfinding, rendering, collision |
| surface_cache | SurfaceCache | Cached surface height results (vx, vy) → height; lazily populated | exploration system | npc_decisions, resource scanning |
| surface_grid | FlatSurfaceGrid | Dense grid of surface heights (fallback cache for perf) | warm_surface_cache() | npc resource cell census, fast height lookups |
| cell_census | CellCensus | Per-resource-cell material counts (populated by NPC scans) | npc scanning, exploration | resource node discovery, NPC targeting |
| voxel_world | VoxelWorld | 3D chunked terrain (materials, elevation source of truth) | terrain generation, voxel destruction | navigation grid generation, building placement |
| nav_grids | Vec<NavGrid> | Walkable surfaces per settlement (derived from voxel_world) | voxel_world changes | npc pathfinding, building AI |

### Memory & Scratch
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| sim_scratch | SimScratch | Pooled buffers (Vec/HashMap) reused across systems to reduce allocs | all systems | all systems (borrowed, cleared per tick) |

### Collections
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

### Metadata
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| build_seeds | Vec<BuildSeed> | Room automaton queue waiting for processing | building_ai | room_growth, building construction |
| structural_events | Vec<StructuralEvent> | Collapses, fractures logged this tick (cleared at tick start) | building/structural systems | event cascade, building damage |
| registry | Option<Arc<Registry>> | Data-driven entity/ability/item definitions (loaded from dataset/) | init | entity spawning, ability execution |
| region_plan | Option<RegionPlan> | Continental terrain plan (stored for post-init chunk generation) | init | chunk generation reference |
| skip_resource_init | bool | Flag to skip resource node spawning (for building AI scenarios) | init | resource_nodes system |

---

## Derivation Graph

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

### Candidates for On-Demand Recomputation

1. **population** — Currently stored; could be recomputed in O(n_entities) per tick. Trade-off: 10–50 cycle loop vs. one integer field. **Keep cached** (too frequent queries).

2. **threat_level** (region & settlement) — Computable from monster_density + recent events + decay. **Keep cached** (used in quest posting hot loop).

3. **surface_cache** — Pure function of voxel_world; safe to invalidate and regenerate on demand, but expensive. **Keep cached** (used 16K+ times per cell scan).

4. **nav_grids** — Pure function of voxel_world; safe to regenerate when voxel changes. **Keep cached** (pathfinding is hot).

5. **cell_census** — Could be recomputed by iterating all resource nodes; takes 10–20ms per full rescan. **Keep cached** (used for target discovery, not latency-sensitive).

6. **trade_route.strength** — Decay is time-based; could be recomputed if last_activity_tick tracked. **Consider: store last_activity_tick, recompute strength on access**.

7. **is_at_war[faction_a, faction_b]** — Currently list lookup; could query relations graph. **Keep list** (hot path in movement, O(1) vs. O(n)).

---

## ConstructionMemory (Building AI Integration)

Per-settlement construction event log with three tiers:

| Tier | Field | Meaning | Capacity | Decay |
|---|---|---|---|---|
| Short-term | short_term: RingBuffer<ConstructionEvent> | Raw events (all types) | 64 entries | None (circular overwrite) |
| Medium-term | medium_term: RingBuffer<AggregatedPattern> | Patterns with importance > 0.3 | 256 entries | Halves every 500 ticks |
| Long-term | long_term: RingBuffer<StructuralLesson> | Structural lessons, importance > 0.7 | 64 entries | Permanent until contradicted |

Updated by: building_ai event logging system  
Read by: building_ai pattern learning (decide build strategies based on past successes/failures)

---

## Payment, ContractBid, ServiceType

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

## Summary: Which Fields Are Emergent vs. Stored

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

## Group (universal)

The social-collective primitive. A Group represents any collection of agents with shared identity and group-level state: factions, families, guilds, religions, hunting packs, criminal cabals, adventuring parties, settlements, leagues, monasteries, courts. The `kind` discriminator + presence/absence of optional fields differentiates them.

### Identity & Membership
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | GroupId | Unique group ID | spawn / FoundGroup event | all group-aware systems |
| kind | GroupKind | Faction, Family, Guild, Religion, Pack, Party, Settlement, Court, Cabal, Other | spawn | mask predicates, observation features |
| name | String | Display name | init / RenameEvent | UI, chronicle, narrative |
| founded_tick | u64 | When formed | FoundGroup event | tenure, history derivations |
| dissolved_tick | Option<u64> | When (or if) dissolved | DissolveGroup event | active filter, narrative |
| members | Vec<AgentId> | Materialized list of agents whose memberships include this group | derived view from agent.memberships (or cached) | quest party expansion, observation, mask |
| founder_ids | Vec<AgentId> | Original founders (immutable) | spawn | history, narrative |

### Leadership / Governance
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| leader_id | Option<AgentId> | Current leader (None for headless groups like ad-hoc parties) | succession / promotion events | mask predicates (leader actions), observation |
| leadership_chain | Vec<AgentId> | Ordered succession queue (for groups with formal succession) | succession events | succession resolution |
| governance | GovernanceKind | Hereditary, Elective, Council, Theocratic, Ad-hoc, ... | init / charter changes | succession mechanism, mask |
| charter | Option<CharterRef> | Settlement-only: governing charter | charter quests | tax rules, eligibility, recruitment |

### Resources & Capacity
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| treasury | f32 | Group gold reserves | tax events, contract payouts, war loot | mask (can-afford predicates), action eval |
| stockpile | Option<[f32; NUM_COMMODITIES]> | Bulk material stores (settlements, guilds with inventory) | production/consumption events | mask (can-fulfill predicates), trade |
| facilities | Option<FacilityRef> | Settlements only: production buildings, infrastructure links | construction events | production eligibility, settlement bonuses |
| military_strength | Option<f32> | For groups that field forces (factions, packs, mercenary guilds) | recruitment, casualties, training | war predicates, mask, observation |

### Standings (relations to other groups)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| standings | Map<GroupId, Standing> | Per-other-group disposition: Allied / Neutral / Tense / AtWar / Vassal / Suzerain / Excommunicate. **Replaces faction.at_war_with + diplomatic_stance.** | diplomacy events (AllianceFormed, WarDeclared, VassalSworn, ...) | mask predicates (can-attack-other-group, eligible for trade with), observation features, is_hostile derivation |
| standing_history | Vec<StandingEvent> | Bounded history of standing changes | diplomacy events | narrative, AI memory |

### Recruitment & Eligibility
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| recruitment_open | bool | Whether agents can JoinGroup unsolicited | charter / leadership decisions | mask (JoinGroup) |
| eligibility_predicate | EligibilityRef | Reference to the rule(s) for membership eligibility (e.g. "must be Smith family", "must complete oath quest", "must be hostile-creature-type") | charter / init | mask (JoinGroup) |
| dues | Option<DuesSpec> | Recurring contribution required of members | charter | tax events, mask (insolvency check) |

### Activity
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| active_quests | Vec<QuestId> | Quests this group has Posted or Accepted as a party | PostQuest / AcceptQuest cascade | quest lifecycle, observation |
| recent_activity | Vec<EventRef> | Bounded log of recent group-level events | various | narrative, observation, AI memory |

### What's NOT on Group (derived views)

- `population` (use `members.len()`)
- `is_at_war(other)` — read from `standings[other]`
- `is_allied(other)` — read from `standings[other]`
- `wealth_per_member` — `treasury / members.len()`
- `cultural_descriptor` — derived from members' aggregate behavior_profiles
- `reputation_among(other_group)` — derived from cross-group event history

### Per-kind shapes

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
