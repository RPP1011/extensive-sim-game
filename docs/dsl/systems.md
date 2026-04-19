# System Inventory (essential vs emergent)

Strict event-sourcing reclassification of all 158 simulation systems into ESSENTIAL / EMERGENT / DERIVABLE-VIEW / DEAD. Organized by batch (original domain split from the investigation phase).

Per-system schema: { File, Classification, Reframe, Required NPC actions, Required derived views, Required event types, One-line summary }.

## Contents
- [Batch 1 — Economic](#batch-1--economic) — (from `systems.md`)
- [Batch 2 — Social / Inner / Needs](#batch-2--social--inner--needs) — (from `systems.md`)
- [Batch 3 — Combat / Quests / Threat](#batch-3--combat--quests--threat) — (from `systems.md`)
- [Batch 4 — Politics / Faction / Meta](#batch-4--politics--faction--meta) — (from `systems.md`)
- [Batch 5 — World / Terrain / Low-level](#batch-5--world--terrain--low-level) — (from `systems.md`)

---

## Batch 1 — Economic

### Methodology

Every system is re-evaluated against a much tighter rubric than v1. The default answer is no longer "this owns state, so it's essential" — the default answer is "this could be an NPC action, so it's emergent."

Four classes:

- **ESSENTIAL** — encodes an irreducible physics/bookkeeping rule: combat damage emission, voxel material change, tile placement, entity ID allocation (spawn), `alive=false` (despawn), `pos += force × dt`, `tick++`, RNG advance, or a spatial/group index that provably cannot be derived from event history. If there is any way to reframe the system's effect as (derived view) → (NPC chooses action) → (action emits event) → (physics rule applies), it is NOT essential.
- **EMERGENT** — the system is currently a central planner doing something an NPC could choose to do. Replace the system with an action vocabulary on the agent side. The system's effect re-appears as the sum of NPC choices, mediated by events.
- **DERIVABLE-VIEW** — a pure function of current state + event history. No system needed; the "state" is a query. Examples: treasury is `Σ income − Σ spend`, price is `f(stockpile, pop)`.
- **DEAD** — function body is a stub / no-op / noise, or admitted-approximation that has never owned real state.

The rubric forces us to separate **physics** (irreducible world change) from **policy** (who decides when it happens). Systems that only encode policy disappear in favor of NPC agency; only the event that physics consumes survives.

---

### Per-system reanalysis

#### economy
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT (with one DERIVABLE-VIEW half)
- **Reframe:** The three transitions v1 claimed as irreducible are all policy choices that belong on agents. (1) Taxation (`economy.rs:49-89`) is a settlement leader choosing `LevyTax(rate)` and each resident choosing `PayTax(amount)` — the current `tax_eff = 1/(1 + treasury/10000)` is just a decision rule the leader would apply. (2) Admin overhead and maintenance (`economy.rs:106-121` surrounding block) are settlement-scoped spend decisions — `PaySalary(role)`, `PayUpkeep(target)`. (3) Debt repayment (`advance_debt`, `economy.rs` direct-mut) is literally the debtor choosing `Repay(creditor, amount)` — it already encodes a rule ("20 % of income, capped at debt, capped at gold") that NPCs could just follow as a utility. The price formula is DERIVABLE-VIEW: `price[c] = base[c] / (1 + (stockpile[c]/pop)/50)` is a pure function.
- **Required NPC actions:** `LevyTax(rate)`, `PayTax(amount)`, `PaySalary(role, amount)`, `PayUpkeep(target, amount)`, `Repay(creditor_id, amount)`.
- **Required derived views:** `treasury(sid) = init + Σ income − Σ spend`, `price(c, sid) = f(stockpile, pop)`, `outstanding_debt(npc) = Σ borrowed − Σ repaid`.
- **Required event types:** `TaxPayment {from_npc, to_settlement, amount}`, `Upkeep {settlement, target, amount}`, `DebtRepayment {debtor, creditor, amount}`, `MoraleDelta {npc, amount, reason}`, `CreditHistoryDelta {npc, delta}`.
- **Verdict:** The only "system" that remains is the event-application rule (ledger fold). Behavior moves to agents.

#### food
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT
- **Reframe:** Production, wage collection, consumption, and starvation are the exact per-NPC workflow v1 describes in its DSL (`food.rs:103`, `food.rs:153-157`). A working NPC chooses `Produce(commodity, rate × level_mult × scale)` once per work cycle; a resident NPC chooses `Eat(food_source)` or suffers a physics-level starvation damage event emitted by a `hunger >= threshold` trigger. The settlement wage is just the employer's `PaySalary(worker, rate × price[c])` — exactly what economy already does. Nothing in this file is an irreducible rule; it is all policy + accounting.
- **Required NPC actions:** `Produce(commodity, amount)`, `Consume(commodity, amount)` (recipe inputs), `Eat(food_source)`, `BeginWorkShift(building)`, `EndWorkShift`.
- **Required derived views:** `stockpile(c, sid) = Σ produce − Σ consume`, `hunger(npc) = last_hunger + tick_delta − Σ eat_amount`, `recipe_feasibility(c, sid) = min(stockpile[inputs]/needed, 1)`.
- **Required event types:** `Production {producer, settlement, commodity, amount}`, `Consumption {consumer, settlement, commodity, amount}`, `Meal {eater, source, hunger_restore}`, `Starvation {npc, damage, morale_delta}`, `WageEarned {worker, commodity, gold}`.
- **Verdict:** Dissolved into agent actions; only the starvation physics rule (`hp -=` on a hunger threshold event) is an event-driven damage application.

#### crafting
- **Old classification:** DUPLICATIVE / STUB
- **New classification:** DEAD (for the regen half) + EMERGENT (for the ore+wood→luxury half)
- **Reframe:** The natural regen (`crafting.rs:68-76`) is noise that doesn't correspond to any agent action — delete. The luxury recipe is a duplicate of what `food.rs` recipes already encode; an NPC with a "craftsman" profile chooses `Produce(Luxury, 1)` with `Consume(Ore, 2) + Consume(Wood, 2)`. No system needed; this is one row in a recipe table.
- **Required NPC actions:** covered by `Produce` + `Consume` above.
- **Required derived views:** stockpile fold.
- **Required event types:** covered by `Production` / `Consumption`.
- **Verdict:** Delete. Fold the luxury recipe into the shared recipe registry.

#### buildings
- **Old classification:** ESSENTIAL (spawn + voxel stamp + nav rebake)
- **New classification:** mostly EMERGENT with a thin ESSENTIAL core
  - ESSENTIAL: entity spawn (`state.entities.push` for a new Building ID), voxel stamp (`state.voxel_world.set_voxel`), nav grid rebake. These are physics: allocating an ID, changing a voxel material, invalidating a spatial cache.
  - EMERGENT: `advance_construction` is an NPC `WorkOnBuilding(id, amount)` loop; `assign_npcs_to_buildings` is an NPC `ClaimWorkSlot(building)` or `ClaimResidence(building)` choice; `update_building_specializations` is a DERIVABLE-VIEW over `worker_class_ticks`; `compute_buildings` treasury upgrade cost is a leader's `PayUpgrade(building)` choice.
- **Reframe:** The file's 1014 lines collapse to: a BlueprintPlacer rule (voxel stamp when a `PlaceTile(pos, material)` event arrives), an ID allocator (when a `StartConstruction(blueprint)` event arrives), and a nav rebake trigger (when `PlaceTile` or `StartConstruction` events affect nav). Everything else is agent policy. The zone-score matching, blueprint attaching, and worker class tallying are all downstream of individual NPC actions + straightforward reductions.
- **Required NPC actions:** `StartConstruction(blueprint, pos)`, `PlaceTile(pos, material)`, `WorkOnBuilding(building_id, effort)`, `ClaimWorkSlot(building_id)`, `ClaimResidence(building_id)`, `PayUpgrade(building_id, amount)`.
- **Required derived views:** `specialization(building) = argmax(worker_class_ticks)/sum(worker_class_ticks)`, `construction_progress(building) = Σ WorkOnBuilding.effort / blueprint.cost`, `is_built(building) = construction_progress >= 1`.
- **Required event types:** `ConstructionStarted {initiator, building_id, blueprint}`, `TilePlaced {placer, pos, material}` (ESSENTIAL emitter), `ConstructionProgress {worker, building, effort}`, `WorkSlotClaimed {npc, building}`, `ResidenceClaimed {npc, building}`, `BuildingUpgraded {building, tier}`, `NavGridDirty {region, aabb}` (ESSENTIAL trigger).
- **Verdict:** Keep only the physics kernel (tile placement, spawn, nav). The orchestration melts into agent actions and their aggregates.

#### work
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT
- **Reframe:** The work.rs state machine (`work.rs` WorkState: Idle→Traveling→Working→Carrying→Idle) is exactly the "agent plan" v1 admits in its own DSL. Every transition is an agent choice: `GoToWorkplace` (Idle→Traveling), `BeginProduction` (Traveling→Working), `FinishProduction` emits a `Production` event (Working→Carrying), `DepositAtSite` emits `Deposit` (Carrying→Idle). Wage and stockpile mutations are agent `Deposit` + settlement `PaySalary` actions. Forge spawning an item is an NPC action `ForgeItem(recipe)` that emits `ItemCrafted {item_id, owner_id}` which is ESSENTIAL physics (spawn). Eating is `Eat(source)`. Voxel harvest is an NPC `HarvestTile(pos)` action that emits `TilePlaced(pos, Air)` + `ResourceHarvested(node, amount)` — the voxel change is ESSENTIAL but the choice to harvest is not. Even the price-belief update is an agent learning rule: `UpdatePriceBelief(c, observed)`.
- **Required NPC actions:** `GoToWorkplace(building_id)`, `BeginProduction(building_id)`, `FinishProduction`, `DepositAtSite(target_id, commodity, amount)`, `ForgeItem(recipe)`, `HarvestTile(pos)`, `HarvestNode(node_id)`, `Eat(source)`, `PurchaseFromStockpile(settlement, commodity, amount)`, `UpdatePriceBelief(c, observed)`.
- **Required derived views:** `inventory(npc)`, `stockpile(c, sid)`, `hunger(npc)`, `income_rate(npc) = EMA over WageEarned`, `price_beliefs(npc, c) = fold over observations`.
- **Required event types:** `ProductionStarted`, `ProductionFinished`, `Deposit {from, to, commodity, amount}`, `ItemCrafted {crafter, item_entity_id, quality, slot}` (ESSENTIAL: new entity ID), `TileHarvested {harvester, pos, material_before}` (ESSENTIAL: voxel change), `ResourceHarvested {harvester, node_id, amount}`, `WageEarned`, `MealEaten`, `PriceBeliefUpdated`.
- **Verdict:** The file becomes a Vec of agent utilities; the simulation's only job is to apply the events they emit.

#### gathering
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT (with physics event for harvest + despawn)
- **Reframe:** `gathering.rs:advance_plans` is a hard-coded plan interpreter: `Gather, MoveTo, Perform, PlaceBuilding, PayGold, Wait`. Each variant is already a named action (the grammar is right there). Promote them to agent-level actions and delete the plan-step-bookkeeping layer — an agent doesn't need a "plan index" in central state if its own utility function picks the next action each tick. `resource.remaining -=` and depletion's `alive=false` are ESSENTIAL physics on a `HarvestNode` event. Full-entity scans for nearest-resource become a pure spatial query, not state owned by this system.
- **Required NPC actions:** `HarvestNode(node_id, amount)`, `MoveTo(pos)`, `PerformTimed(ticks)`, `PlaceBlueprint(pos, type)`, `PayGold(target, amount)`, `Wait(ticks)`.
- **Required derived views:** `nearest_resource(pos, rtype)` is a spatial-index query; `node.remaining` is a state field on the resource entity; plan progress is just the agent's own memory.
- **Required event types:** `ResourceHarvested {harvester, node_id, amount}`, `NodeDepleted {node_id}` (ESSENTIAL: emits despawn), `Movement {entity, from, to}`, `GoldPaid {from, to, amount}`.
- **Verdict:** The plan-executor vanishes; agents pick actions directly. The event-apply layer handles `remaining -=` and the depletion → `alive=false` transition.

#### trade_goods
- **Old classification:** EMERGENT-CANDIDATE + DUPLICATIVE
- **New classification:** DERIVABLE-VIEW (price) + EMERGENT (caravan)
- **Reframe:** The price formula (`trade_goods.rs:42-45`) is literally `demand/max(stockpile, 0.1)` — pure function. Remove, replace with `fn price(sid, c)`. The caravan arbitrage (`trade_goods.rs:57-119`) is a central planner doing an NPC's job: discover max-min price gap, pick goods, move them, collect profit. This is an NPC `RunCaravan(src, dst, commodity, amount)` action; the settlement's treasury gain is a `GoldReceived` event. The regen duplicates `crafting.rs`; delete.
- **Required NPC actions:** `RunCaravan(src, dst, commodity, amount)` (or decomposed: `BuyFromSettlement`, `Travel`, `SellToSettlement`).
- **Required derived views:** `price(c, sid)`, `stockpile(c, sid)`, `arbitrage_opportunities() = pairs where price_dst/price_src > threshold`.
- **Required event types:** `CaravanDeparted {trader, src, dst, commodity, amount}`, `CaravanArrived {trader, dst, commodity, delivered_amount, transit_loss}`, `GoldReceived {recipient, amount, source}`.
- **Verdict:** Price formula becomes a query; caravan becomes an agent action.

#### trade_routes
- **Old classification:** ESSENTIAL
- **New classification:** DERIVABLE-VIEW (mostly) + EMERGENT (agent history side)
- **Reframe:** v1 claimed `Vec<TradeRoute>` was irreducible because it tracks establishment tick + strength. But every field is derivable: `strength(src, dst) = f(time since last trade, Σ profit)` is a pure fold over `CaravanArrived` events with a decay kernel. `trade_count(src, dst) = count(CaravanArrived)`. The "establish when count >= 3" rule becomes a query that returns which pairs qualify. The chronicle entry on establishment/abandonment is an ESSENTIAL narrative event but can be emitted by a trigger on the view crossing a threshold. `npc.trade_history` is agent-side memory (already path-dependent on agent, but that's per-NPC state naturally).
- **Required NPC actions:** implicit via `RunCaravan` (feeds the fold).
- **Required derived views:** `route_strength(a, b) = Σ(CaravanArrived(a→b).profit × exp(-(now−tick)/tau))`, `established_routes() = pairs where trade_count >= 3`, `route_utility_bonus(a, b) = f(strength)`.
- **Required event types:** `CaravanArrived` (above) is sufficient input. Narrative triggers: `RouteEstablished {a, b}`, `RouteAbandoned {a, b}` emitted when the derived view crosses ±threshold.
- **Verdict:** Persistent Vec goes away; read-time queries replace it. Chronicle entries are threshold-triggered events.

#### trade_guilds
- **Old classification:** EMERGENT-CANDIDATE (price half) + ESSENTIAL (gold half)
- **New classification:** EMERGENT
- **Reframe:** "Settlement with ≥3 traders forms a guild" is a derivable predicate. The price floor `max(0.7, f(state))` is a read-time clamp on the derived price. The guild-funds-best-merchant-for-15 gold is simply an NPC (or guild-leader NPC) taking action `FundMerchant(target, amount)` when they qualify — there is no guild entity required, just a decision rule on the merchants who satisfy the membership predicate. Rival undercutting is a merchant's `AdjustLocalPrice(c, delta)` choice (or via a posted contract to suppress rivals); it should not be a centrally-applied price mutation. NPC tag accumulation is downstream of choice (agent `AccumulateTags` on successful action).
- **Required NPC actions:** `FundMerchant(target, amount)`, `AdjustLocalPrice(commodity, delta)` (or we drop price-writing-as-action and keep price purely derivable), `CoordinateUndercut(rival_settlement, commodity)`.
- **Required derived views:** `is_guild_member(npc)`, `guild_members(sid) = list where trade+neg > 30`, `has_guild(sid) = guild_members.len >= 3`.
- **Required event types:** `GuildFunded {funder, recipient, amount}`, `GuildFormed {sid}` (threshold trigger), `PriceAdjustment {settlement, c, delta, reason}`.
- **Verdict:** Dissolves into agent actions + a threshold-triggered narrative event.

#### contracts
- **Old classification:** ESSENTIAL (advance) + STUB (compute)
- **New classification:** EMERGENT
- **Reframe:** `compute_contracts` is a random treasury nudge (`contracts.rs:53`) — DEAD. `advance_contracts` is the full bidding lifecycle — but every transition is an NPC choice. The contract requester chooses `PostContract(spec)`; eligible bidders choose `BidOnContract(contract_id, bid_amount)`; the requester chooses `AcceptBid(bid_id)` (or a scoring rule auto-accepts the best); payment on completion is `PayContract(provider, amount)`. The `service_contracts: Vec<ServiceContract>` becomes a query over open `ContractPosted` events that haven't received `ContractCompleted`.
- **Required NPC actions:** `PostContract(spec)`, `BidOnContract(contract_id, bid_amount)`, `AcceptBid(bid_id)`, `WithdrawBid(bid_id)`, `MarkContractFulfilled(contract_id)`, `PayContract(provider, amount)`.
- **Required derived views:** `open_contracts(sid)`, `bids_on(contract_id)`, `eligible_bidders(contract_id) = idle npc matching service type`.
- **Required event types:** `ContractPosted {poster, service, max_payment, deadline}`, `BidPlaced {bidder, contract, amount}`, `BidAccepted {contract, bid}`, `ContractCompleted {contract, provider}`, `ContractPayment {from, to, amount}`, `ContractExpired {contract}`.
- **Verdict:** Replace entire 403-line lifecycle with action vocabulary + event-sourced queries.

#### contract_negotiation
- **Old classification:** STUB/DEAD
- **New classification:** DEAD
- **Reframe:** Pure random treasury bonus (`contract_negotiation.rs:8` admits placeholder). Delete. If real negotiation is desired later, it's `NegotiateTerms(contract, counter_offer)` — an agent action that modifies a pending `ContractPosted` before `BidAccepted` fires.
- **Required NPC actions:** (future) `CounterOffer(contract, terms)`.
- **Required event types:** (future) `ContractAmended`.
- **Verdict:** Delete.

#### auction
- **Old classification:** STUB
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** The current file (`auction.rs:30`) is a slow treasury drain — delete. A real auction is just `PostAuction(item, reserve)`, `PlaceAuctionBid(auction, amount)`, `CloseAuction(auction)` — three agent actions mediated by events. No system needed.
- **Required NPC actions:** `PostAuction(item_id, reserve_price, deadline)`, `PlaceAuctionBid(auction_id, amount)`, `CloseAuction(auction_id)`.
- **Required derived views:** `open_auctions(sid)`, `high_bid(auction_id)`.
- **Required event types:** `AuctionPosted`, `AuctionBid`, `AuctionClosed {winner, final_price}`, `ItemTransferred {from, to, item}`, `GoldPaid`.
- **Verdict:** Delete current. Reframe future version as actions.

#### black_market
- **Old classification:** STUB / EMERGENT-CANDIDATE
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Header (`black_market.rs:8-10`) admits approximation; the output is a price-triggered random gold shower — delete. A real black market is `SmugglingDeal(buyer, seller, commodity, amount, price)` with a detection roll against an authority NPC's `InspectMarket` action. Heat, reputation, and discovery are derivable views over those events.
- **Required NPC actions:** `OfferIllicitGoods(commodity, amount, price)`, `PurchaseIllicit(offer_id)`, `InspectMarket(sid)`, `Bribe(target, amount)`.
- **Required derived views:** `heat(sid) = Σ IllicitTrade.amount × weight − Σ InspectMarket × decay`, `reputation(npc)`.
- **Required event types:** `IllicitTrade {buyer, seller, commodity, amount, price}`, `IllicitDetected {inspector, seller}`, `Bribe {from, to, amount}`.
- **Verdict:** Delete current; reframe as agent actions + derived heat.

#### commodity_futures
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Current file is a tick-parity treasury oscillator — delete. Futures contracts are agent actions: `WriteFuture(commodity, amount, strike_price, expiry)`, `BuyFuture(future_id)`, `SettleFuture(future_id)` at expiry.
- **Required NPC actions:** `WriteFuture`, `BuyFuture`, `SettleFuture`.
- **Required derived views:** `open_futures(sid)`, `fair_value(future) = f(price_expectation)`.
- **Required event types:** `FutureWritten`, `FuturePurchased`, `FutureSettled {future, cash_settlement}`.
- **Verdict:** Delete.

#### price_discovery
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT (with persistent agent memory)
- **Reframe:** v1 called `npc.price_knowledge: Vec<PriceReport>` irreducible. But this is *per-agent memory* that the agent naturally owns — it should not be written by a central system. Replace with: agents choose `ObserveLocalPrice(sid)` when at a settlement (emits `PriceObserved`) and `Gossip(target_npc, topic)` when paired up (emits `PriceGossiped`). Each agent's `price_knowledge` is a derivable fold over its own observed+gossiped events, or a direct agent-owned ring buffer updated by those actions. The stochastic pair selection currently in the system is pushed into agents themselves.
- **Required NPC actions:** `ObserveLocalPrice`, `Gossip(partner, topic)`, `RecordPriceObservation(c, sid, value)`.
- **Required derived views:** `agent.price_knowledge = fold over PriceObserved + PriceGossiped`.
- **Required event types:** `PriceObserved {observer, settlement, c, value, tick}`, `PriceGossiped {from, to, report}`.
- **Verdict:** Promote to agent actions; remove central scheduler.

#### price_controls
- **Old classification:** EMERGENT-CANDIDATE (clamp) + ESSENTIAL (shortage/subsidy)
- **New classification:** EMERGENT
- **Reframe:** The ceiling/floor clamp is DERIVABLE (clip the derived price at read time). The stockpile drain on ceiling ("shortage") is the residents choosing to `HoardCommodity` or `BuyUpShortage` when the announced price is below equilibrium — an NPC action. The treasury subsidy on floor is the settlement leader's `PaySubsidy(commodity, amount)` — already fits LevyTax/PayUpkeep style. The act of imposing a control is the leader's `SetPriceCeiling(c, value)` or `SetPriceFloor(c, value)` action.
- **Required NPC actions:** `SetPriceCeiling(c, v)`, `SetPriceFloor(c, v)`, `RemovePriceControl(c)`, `PaySubsidy(c, amount)`, `HoardCommodity(c, amount)`.
- **Required derived views:** `effective_price(c, sid) = clamp(derived_price, ceiling, floor)`, `active_controls(sid)`.
- **Required event types:** `PriceCeilingSet`, `PriceFloorSet`, `SubsidyPaid`, `HoardingEvent`.
- **Verdict:** Actions + a read-time clamp.

#### currency_debasement
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** `currency_debasement.rs` admits no faction currency state and uses a nonsense `(rid - sid).abs() <= 1` proximity heuristic. Delete. Real debasement is a faction leader's `DebaseCurrency(rate)` action, with an event `CurrencyDebased {faction, rate}` that an agent-side price-update rule applies on next observation.
- **Required NPC actions:** `DebaseCurrency(faction, rate)`, `MintCoinage(faction, amount)`.
- **Required derived views:** `currency_purity(faction) = initial × Π(1 − debasement_rate)`.
- **Required event types:** `CurrencyDebased`, `CoinageMinted`.
- **Verdict:** Delete. Reframe as a faction-leader decision.

#### smuggling
- **Old classification:** DUPLICATIVE
- **New classification:** EMERGENT (duplicates the black_market + caravan action set)
- **Reframe:** `smuggling.rs` differs from `trade_goods` caravan only in constants (`smuggling.rs:57-119` vs `trade_goods.rs:57-119`). Both collapse into the same `RunCaravan` action, optionally flagged `illicit`, with detection rolls from the black-market vocabulary. Delete; goods moved via agent actions only.
- **Required NPC actions:** covered by `RunCaravan` + `OfferIllicitGoods` + `InspectMarket`.
- **Required derived views:** arbitrage + heat views above.
- **Required event types:** `CaravanDeparted/Arrived` + `IllicitTrade/Detected`.
- **Verdict:** Delete.

#### economic_competition
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Treasury/price nudge on `treasury/avg_treasury` is pure function of state (`economic_competition.rs:98-119`) and the region-proximity block computes distance-from-origin, which is a bug. Delete. Real trade wars are `DeclareTradeWar(target_faction)`, `ImposeTariff(commodity, rate)`, `ImposeEmbargo(target_faction, commodity)` — agent actions that attach to caravan/trade events as gating or bonus.
- **Required NPC actions:** `DeclareTradeWar`, `ImposeTariff`, `ImposeEmbargo`, `LiftEmbargo`.
- **Required derived views:** `active_embargoes(sid)`, `tariff_rate(sid, c)`, `trade_war_state(a, b)`.
- **Required event types:** `TradeWarDeclared`, `TariffImposed`, `EmbargoImposed`, `TradePenaltyAssessed {trader, settlement, amount}`.
- **Verdict:** Delete. Reframe as faction-leader policy actions.

#### bankruptcy_cascade
- **Old classification:** EMERGENT-CANDIDATE
- **New classification:** DERIVABLE-VIEW
- **Reframe:** The entire effect is `total_loss = Σ|deficit| × 0.15; healthy.treasury -= total_loss / n_healthy` — a pure function of current treasuries. Compute at read time or during the periodic settlement book-close pass. No system needed.
- **Required NPC actions:** none inherent; if contagion is desired via creditor relationships, use `loans`-style `Repay` / `Default` actions.
- **Required derived views:** `insolvent_settlements()`, `systemic_risk_multiplier() = f(insolvent_count/total)`, `correction_transfer(sid) = view`.
- **Required event types:** `SettlementInsolvent {sid}` (threshold trigger), `BankruptcyCorrection {sid, delta}` (if realized at book-close).
- **Verdict:** Becomes a query. No tick-scheduled mutation.

#### corruption
- **Old classification:** STUB/DEAD
- **New classification:** EMERGENT (as-intended) + DEAD (as-written random rolls)
- **Reframe:** Random hashed treasury drain is noise. Real corruption is an official NPC choosing `Embezzle(amount)`, emitting `Embezzlement {actor, settlement, amount}`, with desertion modeled as NPC `Desert(target_settlement)` actions. Morale and inspection are agent-side.
- **Required NPC actions:** `Embezzle(amount)`, `Bribe(target, amount)`, `Investigate(suspect)`, `Desert(reason)`.
- **Required derived views:** `corruption_index(sid) = Σ Embezzlement.amount / treasury × decay`, `integrity(npc) = Σ uncovered_acts`.
- **Required event types:** `Embezzlement`, `InvestigationStarted`, `CorruptionUncovered`, `Desertion`.
- **Verdict:** Delete current. Reframe as agent choices.

#### loans
- **Old classification:** STUB/DEAD / DUPLICATIVE
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Current implementation is a second tax (`loans.rs` flat 5% of gold) unrelated to `npc.debt`. Delete. Real loans are `RequestLoan(lender, amount, terms)`, `GrantLoan(request_id)`, `Repay(loan_id, amount)`, `Default(loan_id)`. The `advance_debt` logic from economy.rs already belongs here as `Repay`.
- **Required NPC actions:** `RequestLoan(amount, terms)`, `GrantLoan(request_id)`, `Repay(loan_id, amount)`, `Default(loan_id)`, `CallDebt(loan_id)`.
- **Required derived views:** `outstanding(npc) = Σ GrantLoan.amount − Σ Repay.amount`, `credit_history(npc) = Σ Repay − Σ Default`, `debt_capacity(npc) = income_rate × horizon`.
- **Required event types:** `LoanRequested`, `LoanGranted {lender, borrower, amount, terms}`, `LoanRepayment {loan, amount}`, `LoanDefault {loan}`.
- **Verdict:** Delete. Reframe. A loan is just a pair of events + a fold.

#### insurance
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Current flat-premium / hp-triggered payout is a tax + reactive gold injection with no policy records. Delete. Real insurance is `BuyPolicy(underwriter, terms)`, `PayPremium(policy, amount)`, `FileClaim(policy, damage)`, `ApproveClaim(claim, amount)`.
- **Required NPC actions:** `BuyPolicy`, `PayPremium`, `FileClaim`, `ApproveClaim`, `DenyClaim`, `CancelPolicy`.
- **Required derived views:** `active_policies(npc)`, `claim_history(npc)`, `reserve(underwriter)`.
- **Required event types:** `PolicyIssued`, `PremiumPaid`, `ClaimFiled`, `ClaimApproved`, `ClaimPaid`, `ClaimDenied`.
- **Verdict:** Delete. Agent actions only.

#### caravans
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT
- **Reframe:** v1 said the on-arrival dump + commission is path-dependent and not reconstructible. But the dump is literally "the trader NPC, upon arriving, chooses `DepositAtSettlement(commodity, amount)` and `TipCommission(settlement, amount)`." These are discrete agent actions at arrival; the tick-scheduled proximity check becomes an agent-side arrival trigger. Raid damage is already `Damage` (combat event) — agents choose `Attack(trader)`, physics applies the damage.
- **Required NPC actions:** `DepositAtSettlement(sid, commodity, amount)`, `TipCommission(sid, amount)`, `Attack(target)` (existing).
- **Required derived views:** `at_destination(npc) = dist(npc.pos, npc.dest.pos) < arrival_radius`.
- **Required event types:** `Deposit`, `GoldPaid`, `Damage` (existing).
- **Verdict:** Dissolved into agent actions on arrival.

#### traveling_merchants
- **Old classification:** DUPLICATIVE
- **New classification:** EMERGENT (same vocabulary as caravans)
- **Reframe:** Identical pattern to `caravans` — merchant arrives somewhere, offloads inventory, takes payment. Merge both into the same `DepositAtSettlement` + `SellToSettlement` action set. Merchant also shares price reports: `Gossip(sid, price_report)`.
- **Required NPC actions:** covered by `DepositAtSettlement`, `SellToSettlement(sid, c, amount)`, `Gossip`.
- **Required derived views:** nearest settlement query.
- **Required event types:** covered above.
- **Verdict:** Delete; unified with caravans.

#### mercenaries
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Current: threat-gated treasury drain or random betrayal damage — delete. Real mercenaries are NPCs with `ContractForHire(employer, terms, duration)`, `Desert(employer)`, `Betray(employer)` actions. Wages are `PaySalary`. No central system.
- **Required NPC actions:** `ContractForHire`, `Desert`, `Betray`, `PaySalary` (existing).
- **Required derived views:** `loyalty(mercenary) = f(wages_paid, treatment_events)`.
- **Required event types:** `MercenaryHired`, `MercenaryDeserted`, `MercenaryBetrayed`.
- **Verdict:** Delete. Reframe as agent choices on existing contract vocabulary.

#### supply
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT + ESSENTIAL (damage physics)
- **Reframe:** Traveler food drain is an agent `ConsumeTravelRations` choice each tick (or an automatic agent rule). Starvation damage on empty inventory is a physics rule on a `StarvationThreshold` event (just like food.rs — same damage kernel). The self-to-self `TransferCommodity` hack goes away — use `Consume`.
- **Required NPC actions:** `ConsumeTravelRations`, `Eat(source)` (shared with food).
- **Required derived views:** `travel_hunger(npc) = f(last_eat_tick, distance_traveled)`, `inv.food(npc)`.
- **Required event types:** `Consumption` (shared), `Starvation {npc, damage}` (ESSENTIAL damage application).
- **Verdict:** Merge into the shared `food` event/action set; one physics rule for starvation damage.

#### supply_lines
- **Old classification:** ESSENTIAL (but overlaps caravans raid)
- **New classification:** EMERGENT (duplicates caravans raid + a patrol check)
- **Reframe:** Interdiction of trader cargo is `RaiderAttackCaravan(trader)` — a hostile NPC choice with a `CargoSeized` event. Patrol protection is just a spatial predicate that modulates the hostile's decision (or causes them to avoid/abort the attack). The `ConsumeCommodity { settlement_id: entity.id }` type pun (`supply_lines.rs:119`) is a bug that disappears when the action is `SeizeCargo(trader, c, amount)`.
- **Required NPC actions:** `RaidCaravan(trader)`, `SeizeCargo(trader, c, amount)`, `Patrol(area)` (patrol's only effect is its own presence in the spatial index — no write).
- **Required derived views:** `hostile_nearby(trader)`, `patrol_nearby(trader)` (spatial queries).
- **Required event types:** `RaidAttempted`, `CargoSeized {victim, attacker, c, amount}`, `RaidDeterredByPatrol`.
- **Verdict:** Delete; same action vocabulary as caravans + combat.

#### equipment_durability
- **Old classification:** ESSENTIAL (advance) + STUB (compute)
- **New classification:** ESSENTIAL (physics: durability decay + despawn on break) — but narrower than v1
- **Reframe:** The compute half is confirmed DEAD (no emission). The advance half has a real physics role: `item.durability -= loss` is a physical-integration step (like HP decay), and `item.alive = false` on break is a despawn. These remain ESSENTIAL. HOWEVER, the stat-rollback on unequip (`owner.attack_damage -= eq`, etc.) is NOT physics — it is the recomputation of a derived view. NPC effective stats should be a DERIVABLE view over `equipped_items`, not a mutated mirror, so equip/unequip just changes the slot and stats re-derive on read. The decision to unequip a broken item is the agent's `UnequipItem(item)` choice triggered by a `durability==0` event.
- **Required NPC actions:** `UnequipItem(item_id)` (triggered by break event or on demand).
- **Required derived views:** `effective_stats(npc) = base + Σ equipped_items.modifiers × durability_frac`.
- **Required event types:** `DurabilityDecay {item, delta, cause}`, `ItemBroken {item}` (ESSENTIAL: emits despawn `alive=false`), `ItemUnequipped {npc, item, slot}`.
- **Verdict:** Keep the physics kernel (durability integration + break→despawn). Delete the stat-mirror write. Add an agent action for unequip.

#### equipping
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT
- **Reframe:** The matching/upgrade logic is exactly the "NPC shops for better gear" agent choice. Each NPC chooses `ClaimItem(item_id, slot)` when they see better gear at their settlement — the central nested loop over items×slots×npcs becomes an agent utility. Emitting Equip/Unequip deltas becomes the two events `ItemEquipped {npc, item, slot}` / `ItemUnequipped`. No central assignment system needed.
- **Required NPC actions:** `ClaimItem(item_id, slot)`, `EquipItem(item_id)`, `UnequipItem(item_id)`, `DropItem(item_id)`.
- **Required derived views:** `unclaimed_items(sid)`, `effective_quality(item) = base_quality × durability_frac`, `upgrade_opportunity(npc) = best unclaimed − currently equipped`.
- **Required event types:** `ItemClaimed`, `ItemEquipped {npc, item, slot}`, `ItemUnequipped`.
- **Verdict:** Promote to agent actions.

#### resource_nodes
- **Old classification:** ESSENTIAL
- **New classification:** ESSENTIAL (narrowed)
- **Reframe:** This is one of the very few truly irreducible systems in the batch.
  - `spawn_initial_resources` allocates new entity IDs with positions and resource data — physics-level entity spawn. ESSENTIAL.
  - `tick_resource_regrowth` is a time-integration on a physical quantity: `remaining += rate × dt` clamped at `max_capacity`. This is the same form as `pos += v × dt` or `hp -= dot × dt`. ESSENTIAL physics rule.
  - Depletion (`alive = false` when `remaining <= 0` and not renewable) is a despawn. ESSENTIAL.
  - `find_nearest_resource` is a DERIVABLE spatial query (replace with a spatial index; not a system).
- **Required NPC actions:** none (agents interact via `HarvestNode`; this file owns only the world-side physics).
- **Required derived views:** `nearest_resource(pos, rtype)` is a spatial query, not owned here.
- **Required event types:** `ResourceRegrown {node, delta}` (optional, most sims don't need to log every tick of regrowth), `NodeDepleted {node}` (emits despawn), `ResourceSpawned {node, rtype, pos}` (emits spawn).
- **Verdict:** Keep. This is the baseline of what "essential" means in this batch.

#### infrastructure
- **Old classification:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Output is pure function of `(treasury, population)` with no infrastructure state (`infrastructure.rs` header admits this). Delete. Real infrastructure is `BuildRoad(from, to)`, `RepairBridge(id)`, `PayMaintenance(asset_id, amount)` — agent actions that spawn Infrastructure entities (same shape as buildings: `StartConstruction` + `WorkOnBuilding` + `PlaceTile`).
- **Required NPC actions:** `BuildRoad(from, to)`, `RepairBridge(id)`, `PayMaintenance(asset_id, amount)`.
- **Required derived views:** `road_network()`, `settlement_connectivity(a, b) = BFS over road graph`, `maintenance_deficit(asset)`.
- **Required event types:** `RoadBuilt`, `BridgeRepaired`, `MaintenancePaid`, `AssetDeteriorated`.
- **Verdict:** Delete. Share construction vocabulary with `buildings`.

---

### Reduction summary

| System | Old | New | Replacement |
|---|---|---|---|
| economy | ESSENTIAL | EMERGENT + DERIVABLE-VIEW | NPC actions: `LevyTax`, `PayTax`, `Repay`; view: `treasury = Σ income − Σ spend`; `price = f(stockpile, pop)` |
| food | ESSENTIAL | EMERGENT | Actions: `Produce`, `Consume`, `Eat`; physics event: `Starvation` damage |
| crafting | DUPLICATIVE | DEAD | Merge recipe into shared registry; delete regen |
| buildings | ESSENTIAL | EMERGENT + thin ESSENTIAL kernel | Actions: `StartConstruction`, `WorkOnBuilding`, `ClaimWorkSlot`, `ClaimResidence`; physics: `TilePlaced` + entity spawn + nav rebake |
| work | ESSENTIAL | EMERGENT | Actions: `GoToWorkplace`, `BeginProduction`, `Deposit`, `ForgeItem`, `HarvestTile` |
| gathering | ESSENTIAL | EMERGENT + physics events | Actions replace plan-executor; physics: `NodeDepleted` → despawn |
| trade_goods | EMERGENT-CANDIDATE | DERIVABLE-VIEW + EMERGENT | `fn price(c,sid)`; action: `RunCaravan` |
| trade_routes | ESSENTIAL | DERIVABLE-VIEW | `route_strength(a,b) = fold over CaravanArrived` |
| trade_guilds | EMERGENT-CANDIDATE/ESSENTIAL | EMERGENT | Actions: `FundMerchant`, `AdjustLocalPrice`; threshold event: `GuildFormed` |
| contracts | ESSENTIAL (advance) | EMERGENT | Actions: `PostContract`, `BidOnContract`, `AcceptBid`, `PayContract` |
| contract_negotiation | STUB | DEAD | Delete |
| auction | STUB | DEAD → EMERGENT | Future: `PostAuction`, `PlaceAuctionBid`, `CloseAuction` |
| black_market | STUB | DEAD → EMERGENT | Future: `OfferIllicitGoods`, `InspectMarket`, derived heat |
| commodity_futures | STUB | DEAD → EMERGENT | Future: `WriteFuture`, `BuyFuture`, `SettleFuture` |
| price_discovery | ESSENTIAL | EMERGENT | Actions: `ObserveLocalPrice`, `Gossip`; per-agent memory fold |
| price_controls | ESSENTIAL (shortage) | EMERGENT | Actions: `SetPriceCeiling`, `PaySubsidy`; read-time clamp |
| currency_debasement | STUB | DEAD → EMERGENT | Future: `DebaseCurrency` faction action |
| smuggling | DUPLICATIVE | EMERGENT | Same action set as `caravans` + black_market |
| economic_competition | STUB | DEAD → EMERGENT | Future: `DeclareTradeWar`, `ImposeTariff`, `ImposeEmbargo` |
| bankruptcy_cascade | EMERGENT-CANDIDATE | DERIVABLE-VIEW | Query at book-close |
| corruption | STUB | DEAD → EMERGENT | Future: `Embezzle`, `Investigate`, `Desert` |
| loans | STUB/DUPLICATIVE | DEAD → EMERGENT | Future: `RequestLoan`, `GrantLoan`, `Repay`, `Default` |
| insurance | STUB | DEAD → EMERGENT | Future: `BuyPolicy`, `FileClaim`, `ApproveClaim` |
| caravans | ESSENTIAL | EMERGENT | Arrival actions: `DepositAtSettlement`, `TipCommission` |
| traveling_merchants | DUPLICATIVE | EMERGENT | Same vocabulary as caravans + `Gossip` |
| mercenaries | STUB | DEAD → EMERGENT | Future: `ContractForHire`, `Desert`, `Betray` |
| supply | ESSENTIAL | EMERGENT + physics | Action: `ConsumeTravelRations`; physics: `Starvation` damage |
| supply_lines | ESSENTIAL | EMERGENT | Actions: `RaidCaravan`, `SeizeCargo`, `Patrol` |
| equipment_durability | ESSENTIAL | ESSENTIAL (physics kernel) + EMERGENT (unequip policy) | Physics: durability integration + break→despawn; derived stats |
| equipping | ESSENTIAL | EMERGENT | Actions: `ClaimItem`, `EquipItem`, `UnequipItem` |
| resource_nodes | ESSENTIAL | ESSENTIAL | Spawn + `remaining += rate × dt` + `alive = false` on depletion |
| infrastructure | EMERGENT-CANDIDATE | DEAD → EMERGENT | Delete stub; actions: `BuildRoad`, `RepairBridge`, `PayMaintenance` |

**Counts under strict rubric:**
- ESSENTIAL (narrow physics only): **2** — `resource_nodes`, `equipment_durability` (kernel only)
- ESSENTIAL kernel embedded in otherwise-EMERGENT system: **1** — `buildings` (spawn/voxel/nav only)
- EMERGENT (replaced by NPC actions): **17**
- DERIVABLE-VIEW (pure function, no system): **2** — `trade_routes`, `bankruptcy_cascade` (plus the price-formula halves of `economy` and `trade_goods`)
- DEAD (delete outright or replace with future EMERGENT design): **13** — `crafting`, `contract_negotiation`, `auction`, `black_market`, `commodity_futures`, `currency_debasement`, `smuggling`, `economic_competition`, `corruption`, `loans`, `insurance`, `mercenaries`, `infrastructure`

---

### Required action vocabulary (deduplicated, batch-level)

#### Taxes / settlement finance
- `LevyTax(rate)` — economy
- `PayTax(amount)` — economy
- `PaySalary(role_or_npc, amount)` — economy, food, work
- `PayUpkeep(target, amount)` — economy
- `PaySubsidy(commodity, amount)` — price_controls
- `PayMaintenance(asset_id, amount)` — infrastructure

#### Production / consumption
- `Produce(commodity, amount)` — food, work, crafting
- `Consume(commodity, amount)` — food, work, crafting
- `Eat(food_source)` — food, supply
- `ConsumeTravelRations` — supply
- `ForgeItem(recipe)` — work (emits ESSENTIAL spawn)
- `HarvestTile(pos)` — work (emits ESSENTIAL voxel change)
- `HarvestNode(node_id, amount)` — work, gathering (consumes `remaining`)

#### Movement / work state
- `GoToWorkplace(building_id)` — work
- `MoveTo(pos)` — gathering, work
- `BeginProduction(building_id)` — work
- `FinishProduction` — work
- `Deposit(target_id, commodity, amount)` / `DepositAtSettlement(sid, c, amount)` — work, caravans, traveling_merchants
- `PerformTimed(ticks)` — gathering
- `Wait(ticks)` — gathering

#### Construction
- `StartConstruction(blueprint, pos)` — buildings (emits ESSENTIAL spawn)
- `PlaceBlueprint(pos, type)` — gathering, buildings
- `PlaceTile(pos, material)` — buildings (emits ESSENTIAL voxel change)
- `WorkOnBuilding(building_id, effort)` — buildings
- `ClaimWorkSlot(building_id)` — buildings
- `ClaimResidence(building_id)` — buildings
- `PayUpgrade(building_id, amount)` — buildings
- `BuildRoad(from, to)` / `RepairBridge(id)` — infrastructure

#### Trade / markets
- `RunCaravan(src, dst, commodity, amount)` — trade_goods, smuggling, caravans
- `TipCommission(sid, amount)` — caravans
- `SellToSettlement(sid, c, amount)` — traveling_merchants
- `PurchaseFromStockpile(sid, c, amount)` — work (eat-from-stockpile path)
- `Gossip(partner, topic)` — price_discovery, traveling_merchants
- `ObserveLocalPrice` — price_discovery
- `RecordPriceObservation(c, sid, value)` — price_discovery
- `UpdatePriceBelief(c, observed)` — work
- `AdjustLocalPrice(c, delta)` — trade_guilds
- `FundMerchant(target, amount)` — trade_guilds
- `SetPriceCeiling(c, v)` / `SetPriceFloor(c, v)` / `RemovePriceControl(c)` — price_controls
- `HoardCommodity(c, amount)` — price_controls

#### Contracts / obligations
- `PostContract(spec)` — contracts
- `BidOnContract(contract_id, bid)` — contracts
- `AcceptBid(bid_id)` — contracts
- `WithdrawBid(bid_id)` — contracts
- `MarkContractFulfilled(contract_id)` — contracts
- `PayContract(provider, amount)` — contracts
- `CounterOffer(contract, terms)` — contract_negotiation (future)

#### Auctions / futures (future EMERGENT)
- `PostAuction(item_id, reserve, deadline)`, `PlaceAuctionBid(auction_id, amount)`, `CloseAuction(auction_id)` — auction
- `WriteFuture(c, amount, strike, expiry)`, `BuyFuture(future_id)`, `SettleFuture(future_id)` — commodity_futures

#### Lending / insurance (future EMERGENT)
- `RequestLoan(amount, terms)`, `GrantLoan(request_id)`, `Repay(loan_id, amount)`, `Default(loan_id)`, `CallDebt(loan_id)` — loans
- `BuyPolicy(underwriter, terms)`, `PayPremium(policy, amount)`, `FileClaim(policy, damage)`, `ApproveClaim(claim, amount)`, `DenyClaim`, `CancelPolicy` — insurance

#### Faction-level policy (future EMERGENT)
- `DeclareTradeWar(target)`, `ImposeTariff(c, rate)`, `ImposeEmbargo(faction, c)`, `LiftEmbargo` — economic_competition
- `DebaseCurrency(faction, rate)`, `MintCoinage(faction, amount)` — currency_debasement

#### Illicit (future EMERGENT)
- `OfferIllicitGoods(c, amount, price)`, `PurchaseIllicit(offer_id)`, `InspectMarket(sid)`, `Bribe(target, amount)` — black_market, smuggling
- `Embezzle(amount)`, `Investigate(suspect)`, `Desert(reason)` — corruption
- `ContractForHire(employer, terms, duration)`, `Desert(employer)`, `Betray(employer)` — mercenaries
- `RaidCaravan(trader)`, `SeizeCargo(trader, c, amount)`, `Patrol(area)` — supply_lines

#### Items / equipment
- `ClaimItem(item_id, slot)` — equipping
- `EquipItem(item_id)` — equipping
- `UnequipItem(item_id)` — equipping, equipment_durability (break trigger)
- `DropItem(item_id)` — equipping

---

### Required event types

#### Ledger / finance events
- `TaxPayment { from_npc, to_settlement, amount }`
- `Upkeep { settlement, target, amount }`
- `SubsidyPaid { settlement, c, amount }`
- `WageEarned { worker, commodity_or_gold, amount }`
- `GoldPaid { from, to, amount, reason }`
- `GoldReceived { recipient, amount, source }`
- `MoraleDelta { npc, amount, reason }`
- `CreditHistoryDelta { npc, delta }`
- `DebtRepayment { debtor, creditor, amount }`

#### Production / consumption events
- `Production { producer, settlement, commodity, amount }`
- `Consumption { consumer, settlement, commodity, amount }` (recipe inputs)
- `Meal { eater, source, hunger_restore, heal }`
- `Starvation { npc, damage, morale_delta }` (ESSENTIAL damage application)
- `Deposit { from, to, commodity, amount }`

#### Voxel / construction / world-mutation events (ESSENTIAL)
- `TilePlaced { placer, pos, material }`
- `TileHarvested { harvester, pos, material_before }`
- `ConstructionStarted { initiator, building_id, blueprint }` (ESSENTIAL: allocates new ID)
- `ConstructionProgress { worker, building, effort }`
- `BuildingUpgraded { building, tier }`
- `NavGridDirty { region, aabb }` (ESSENTIAL: spatial cache invalidation trigger)
- `WorkSlotClaimed { npc, building }`
- `ResidenceClaimed { npc, building }`

#### Resource / item / entity lifecycle (ESSENTIAL spawn/despawn)
- `ResourceSpawned { node, rtype, pos }`
- `ResourceHarvested { harvester, node_id, amount }`
- `ResourceRegrown { node, delta }` (optional logging; the integration is the physics rule)
- `NodeDepleted { node }` → emits despawn
- `ItemCrafted { crafter, item_entity_id, quality, slot }` (ESSENTIAL: new entity ID)
- `ItemBroken { item }` → emits despawn
- `DurabilityDecay { item, delta, cause }`
- `ItemEquipped { npc, item, slot }`
- `ItemUnequipped { npc, item, slot }`
- `ItemClaimed { npc, item }`

#### Trade / market events
- `CaravanDeparted { trader, src, dst, commodity, amount }`
- `CaravanArrived { trader, dst, commodity, delivered, transit_loss }`
- `PriceObserved { observer, settlement, c, value, tick }`
- `PriceGossiped { from, to, report }`
- `PriceCeilingSet { settlement, c, v }`
- `PriceFloorSet { settlement, c, v }`
- `PriceAdjustment { settlement, c, delta, reason }`
- `HoardingEvent { hoarder, c, amount }`
- `GuildFormed { sid }` (threshold trigger)
- `GuildFunded { funder, recipient, amount }`

#### Contract / auction / loan / insurance events
- `ContractPosted { poster, service, max_payment, deadline }`
- `BidPlaced { bidder, contract, amount }`
- `BidAccepted { contract, bid }`
- `ContractCompleted { contract, provider }`
- `ContractPayment { from, to, amount }`
- `ContractExpired { contract }`
- `ContractAmended { contract, new_terms }` (future)
- `AuctionPosted`, `AuctionBid`, `AuctionClosed { winner, final_price }`, `ItemTransferred { from, to, item }`
- `FutureWritten`, `FuturePurchased`, `FutureSettled { future, cash_settlement }`
- `LoanRequested`, `LoanGranted { lender, borrower, amount, terms }`, `LoanRepayment { loan, amount }`, `LoanDefault { loan }`
- `PolicyIssued`, `PremiumPaid`, `ClaimFiled`, `ClaimApproved`, `ClaimPaid`, `ClaimDenied`

#### Narrative / threshold triggers
- `RouteEstablished { a, b }`, `RouteAbandoned { a, b }`
- `SettlementInsolvent { sid }`
- `BankruptcyCorrection { sid, delta }`
- `Embezzlement { actor, settlement, amount }`
- `InvestigationStarted`, `CorruptionUncovered`, `Desertion`
- `MercenaryHired`, `MercenaryDeserted`, `MercenaryBetrayed`
- `TradeWarDeclared`, `TariffImposed`, `EmbargoImposed`, `TradePenaltyAssessed`
- `CurrencyDebased { faction, rate }`, `CoinageMinted`
- `IllicitTrade { buyer, seller, c, amount, price }`, `IllicitDetected`, `Bribe`
- `RaidAttempted`, `CargoSeized { victim, attacker, c, amount }`, `RaidDeterredByPatrol`
- `RoadBuilt`, `BridgeRepaired`, `MaintenancePaid`, `AssetDeteriorated`

---

### Required derived views (cached or lazy)

#### Ledger folds
- `treasury(sid) = initial + Σ income − Σ spend` — over `TaxPayment`, `GoldPaid`, `Upkeep`, `SubsidyPaid`, `ContractPayment`, `BankruptcyCorrection`
- `gold(npc) = initial + Σ WageEarned + Σ received − Σ paid` — agent-side
- `outstanding_debt(npc) = Σ LoanGranted.amount − Σ LoanRepayment.amount`
- `credit_history(npc) = Σ Repay − Σ Default`
- `debt_capacity(npc) = income_rate × horizon`
- `income_rate(npc) = EMA over WageEarned`

#### Stockpile / price / demand
- `stockpile(c, sid) = Σ Production − Σ Consumption − Σ Deposit_out + Σ Deposit_in`
- `price(c, sid) = base[c] / (1 + (stockpile[c]/pop)/50)` — pure function, canonical
- `effective_price(c, sid) = clamp(price(c,sid), floor, ceiling)` — when controls active
- `currency_purity(faction) = initial × Π(1 − debasement_rate)`
- `arbitrage_opportunities() = { (src,dst,c) | price(c,dst)/price(c,src) > threshold ∧ stockpile(c,src) > min }`

#### Agent state
- `hunger(npc) = last_hunger + tick_delta − Σ Meal.hunger_restore`
- `travel_hunger(npc) = f(last_eat_tick, distance_traveled)`
- `inventory(npc) = Σ pickups − Σ drops/transfers`
- `effective_stats(npc) = base + Σ equipped_items.modifiers × durability_frac` — replaces the stat-mirror write in equipment_durability
- `price_beliefs(npc, c) = fold over PriceObserved + PriceGossiped (with recency window)`

#### Social / relational
- `route_strength(a, b) = Σ CaravanArrived(a→b).profit × exp(-(now−tick)/τ)`
- `established_routes() = pairs where count(CaravanArrived) ≥ 3`
- `is_guild_member(npc) = home=sid ∧ trade+negotiation > 30`
- `has_guild(sid) = |guild_members| ≥ 3`
- `loyalty(mercenary) = f(wages_paid, treatment_events)`
- `reputation(npc) = f(uncovered_acts, public_deeds)`
- `heat(sid) = Σ IllicitTrade.amount × weight − Σ InspectMarket × decay`
- `corruption_index(sid) = Σ Embezzlement.amount / treasury × decay`

#### Structural
- `construction_progress(building) = Σ WorkOnBuilding.effort / blueprint.cost`
- `is_built(building) = construction_progress ≥ 1`
- `specialization(building) = argmax(worker_class_ticks) / Σ worker_class_ticks`
- `active_policies(npc)`, `open_futures(sid)`, `open_auctions(sid)`, `open_contracts(sid)` — slices over lifecycle-event folds
- `active_embargoes(sid)`, `tariff_rate(sid, c)`, `trade_war_state(a, b)`
- `road_network()`, `settlement_connectivity(a, b) = BFS over road graph`
- `insolvent_settlements() = { sid | treasury(sid) < 0 }`
- `systemic_risk_multiplier() = f(|insolvent|/|settlements|)`
- `nearest_resource(pos, rtype)` — spatial-index query, not a fold
- `hostile_nearby(trader)`, `patrol_nearby(trader)` — spatial-index queries

---

### Truly essential (irreducible) set in this batch

Only three survive the strict test:

1. **resource_nodes** — `spawn_initial_resources` (entity ID allocation + pos), `tick_resource_regrowth` (`remaining += rate × dt`, clamped at capacity; `alive = false` on non-renewable depletion). Everything here is time-integration on a physical quantity or a lifecycle transition.

2. **equipment_durability** (advance kernel only) — durability integration (`durability -= loss` per tick of wear) plus break despawn (`alive = false` when `durability <= 0`). The stat-rollback is NOT physics and disappears into a derived view.

3. **buildings** (thin kernel only) — when a `StartConstruction` event arrives, allocate a new entity ID; when a `PlaceTile` event arrives, call `voxel_world.set_voxel`; when either happens inside a nav-tracked region, emit `NavGridDirty`. The 1014-line orchestration above that kernel is entirely EMERGENT (agent choices + reductions).

Everything else in the batch — all taxation, production scheduling, trade routing, contract lifecycles, market pricing, guild formation, equipping policy, bankruptcy propagation, and the entire "without X state" family — is either a policy that belongs on NPC agents (EMERGENT), a pure function of state (DERIVABLE-VIEW), or dead code (DEAD).

The consequence: **30 of 32 systems collapse into an action vocabulary + an event-fold index.** The ECS DSL needs maybe 60-80 agent action verbs and ~50 event types to cover this entire batch, with a dozen derived views defined as named queries. The remaining irreducible kernel is small enough to live inside a single ~200-line physics module covering resource integration, durability integration, spawn/despawn, voxel writes, and nav invalidation.

---

## Batch 2 — Social / Inner / Needs

### Methodology

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

### Per-system reanalysis

#### agent_inner
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (reduced) + DERIVABLE-VIEW (majority)
- **Reframe:** `agent_inner.rs` (1563 lines) is the *one* place that emits `MemoryEvent`s into `npc.memory.events` via `record_npc_event` (line 1537). That emission IS the event-sourcing primitive — it's the bookkeeping write that makes everything else reconstructable. Note how `record_npc_event` itself (lines 1553-1562) already *couples* event recording with emotion writes — `joy+=impact×0.5, pride+=impact×0.3` on positive events and `fear/grief/anger` spikes on negative ones. Under the strict rubric even that coupling is wrong: the function should record the event and return; the emotion field should be a view (`emotions(npc) = fold_over_recent_events`). What's **not** essential is the 1400+ lines of derived-state math piled on top: emotion spikes on needs (`spike_emotions_from_needs`), belief decay (0.95/tick, line 28), personality drift (`drift_personality_from_memory` at line 862), aspiration recomputation every 500 ticks, coworker social graph maintenance (lines 650–688), price-belief decay, context-tag accumulation from emotions across the 9 emotion-tag blocks. All of these are `f(memory.events, needs, time_since_last_tick)` and should be evaluated lazily as views, not churned every 10 ticks over every NPC. The only things that must remain imperative: (1) `record_npc_event` on externally-observed world events (deaths, attacks, conquests, trades) — but as a pure appender, (2) `npc.needs.*` drift (physics: hunger/shelter depletion is actual time-integrated entropy, not a function of events), (3) perception-driven `npc.known_resources` insertion via the resource_grid lookup (observation itself is an event and the insertion IS the event-handler side effect).
- **Required NPC actions:** none — this *is* the event bus, not an action.
- **Required derived views:** `emotions(npc) = f(recent memory.events)`, `mood(npc) = f(needs, emotions)`, `beliefs(npc) = fold(memory.events → belief deltas, decay=time_since)`, `personality_drift(npc) = integrate(memory.events × trait_kernels)`, `aspiration(npc) = latest unmet-need projection`, `coworker_familiarity(a,b) = co_shifts_count / decay`, `relationship_trust(a,b)`, `action_outcomes_ema(npc, action_kind)`.
- **Required event types:** `Witnessed { observer, event_id, impact, tick }` (central emission), `NeedsTicked { npc, Δhunger, Δshelter, Δsocial, tick }`, `ResourceObserved { observer, resource_id, kind, pos, tick }`, `FriendDied { observer, friend_id, tick }`, `WasAttacked { observer, attacker, tick }` — most of these already exist as `MemEventType`, so the rewrite is really a renaming + view extraction.
- **Summary:** Keep only the event-recorder core + raw physics needs; migrate all derived fields (emotions, beliefs, personality drift, aspiration, relationship trust) to lazy-evaluated views over `memory.events`. Estimated reduction: 1563 → ~300 lines of essential mutation.

#### action_eval
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (the agentic decision engine) + DERIVABLE-VIEW (its scoring inputs)
- **Reframe:** `action_eval.rs` (1215 lines, ~7.9% of tick — heaviest per-tick system in this batch). `evaluate_and_act` (line 202) runs every 5 ticks; the two-phase pattern (score → defer → execute) with typed snap-grids (line 40: `TypedSnapGrids { resources, buildings, combatants }`) is the argmax-over-candidates implementation. This is *the* function that chooses which DSL action each NPC performs — it's irreducibly essential, it's the top of the agentic pipeline, and every EMERGENT reclassification in this document assumes this engine will grow additional candidate actions for the social/political verbs listed below. However, the internal scoring is pure derivation: `utility = f(need_urgency × personality × aspiration.need_vector × cultural_bias × action_outcomes_EMA × distance)`. The utility function is a query, not a stateful system. In the strict rubric, *decision dispatch = ESSENTIAL; scoring rules = DERIVABLE*. Candidate actions and scoring kernels should be declared in the DSL (one declaration per action type with conditions, targets, and utility expression); the engine is just `argmax(enumerated_candidates)`. A secondary observation: `npc.action_outcomes` (EMA state currently persisted on NpcData) is also derivable from the `ActionChosen` event log — but for hot-path reasons it's probably worth memoizing. The cleanup is orthogonal to the rubric.
- **Required NPC actions:** the entire emergent action vocabulary (see Required Action Vocabulary section). Currently ~13 `CandidateAction` variants (Eat, Harvest, Attack, Flee, Build, Work, Move, Idle, …); needs expansion to cover social/political actions (`SwearOath`, `Train`, `MarkWanted`, `ProposeMarriage`, `SignDemonicPact`, `Court`, `BegPardon`, `PostBounty`, `JoinReligion`, `BeginPlot`, `ConsumePotion`, `PracticeHobby`, …). Each new action is one DSL declaration, not a new Rust file.
- **Required derived views:** `utility_score(npc, candidate) = f(needs, personality, aspiration, emotions, cultural_bias, action_outcomes, distance)`, `candidate_targets(npc, action_type, grid)`, `action_outcomes_ema(npc, action_kind) = fold(ActionChosen events where npc matches, decay)`.
- **Required event types:** `ActionChosen { npc, action, target, score, tick }` — so the scoring of previous choices becomes reconstructable for the EMA without storing `action_outcomes` on the NPC. `ActionFailed { npc, action, reason, tick }` — for negative-outcome EMA updates.
- **Summary:** Decision dispatch is irreducible; the utility math and action-outcome EMA are derivable from the `ActionChosen` event history; candidate action set grows to absorb the 12 EMERGENT systems below.

#### action_sync
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `action_sync.rs` (113 lines) writes `npc.action` as a view over `(work_state, goal_stack.current, move_target, current_intention)`. This is a literal "compute a field from fields" system with no external triggers and no event side-effects. Delete it, replace with `fn npc_action(&npc, &entity) -> NpcAction` called at the read sites. Persisting it was a rendering convenience, not simulation state.
- **Required derived views:** `npc_action(npc, entity) -> NpcAction`.
- **Required event types:** none.
- **Summary:** Pure projection; inline as a method.

#### moods
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** DERIVABLE-VIEW
- **Reframe:** `moods.rs` (232 lines). Mood as a categorical state is never stored on NpcData. Every invocation re-derives a pseudo-mood from `hp_ratio` + `fidelity_zone.fidelity` and emits `Morale ±0.5..3.0` deltas across three branches (low-HP-in-combat, high-HP-in-combat, idle-high-HP). That is textbook "view with side effects." The morale nudges are redundant with agent_inner's emotion spikes (which already run every 10 ticks). The contagion phase at lines 141–153 is marked "deterministic roll" in comments and is unimplemented. Strict rubric: the entire compute body deletes. What's valuable in this file is the infrastructure: the `Mood` enum, the `MoodCause` enum, `MoodSnapshot`, and the ~7 query helpers (`mood_combat_multiplier`, `is_reckless`, etc.) — all of which *consume* a derived mood. Those stay, but the view they consume becomes `fn mood(npc) -> Mood` defined in terms of `emotions(npc)`, `needs(npc)`, and `hp_ratio`. Morale itself becomes a view: `morale(npc) = baseline + Σ recent_event_morale_impacts × decay`.
- **Required NPC actions:** none.
- **Required derived views:** `mood(npc) -> Mood`, `morale(npc) -> f32`, `mood_combat_multiplier(mood)`, `is_reckless(mood)`, `mood_snapshot(npc) -> MoodSnapshot`.
- **Required event types:** none (consumes existing memory.events + needs).
- **Summary:** Pure function of NPC state; delete the compute body, keep the enum + query helpers as the view API.

#### fears
- **Old:** STUB/DEAD + EMERGENT-CANDIDATE
- **New:** DEAD
- **Reframe:** `fears.rs` (140 lines). Phase 1 (acquisition) body is empty; Phase 4 (mentorship) is `let _ = npc_ids`; Phase 5 (contagion) is empty. Phase 2 is a 2-branch HP-ratio morale drain that exactly duplicates agent_inner's fear spike. NpcData has a `fears: Vec<u8>` field that this system never touches. Delete. If phobia acquisition ever gets implemented, it belongs as a reactive `memory.events → BeliefType::Phobia(source)` derivation, not a scanning system.
- **Required NPC actions:** `FleeFromPhobicSource` (already covered by `Flee`).
- **Required derived views:** `phobias(npc) = f(memory.events matching traumatic_attack)`.
- **Required event types:** none (reuses `WasAttacked`).
- **Summary:** Delete; phobia acquisition is a belief derivation from memory.

#### personal_goals
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `personal_goals.rs` (184 lines). Fires a chronicle + morale bump when `behavior_profile[tag]` crosses a narrow threshold window. The *goal itself* is an NPC's choice of aspiration — that's agentic. In the new model, NPC picks `PursueGoal(kind)` as an action (or `action_eval` enumerates goal-pursuit candidates from `aspiration.need_vector`); reaching the threshold emits a `GoalAchieved { npc, kind, tier }` event; the chronicle and title systems derive pride/fame from the event. No per-tick scan needed.
- **Required NPC actions:** `PursueGoal(kind)`, `DeclareMastery(domain)`.
- **Required derived views:** `goal_progress(npc, kind) = npc.behavior_value(kind.tag) / kind.threshold`.
- **Required event types:** `GoalAchieved { npc, kind, tier, tick }`.
- **Summary:** Goal pursuit is agentic, completion is a threshold event; scan-and-fire body deletable.

#### hobbies
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `hobbies.rs` (180 lines). Idle NPCs drift toward adjacent skills via 0.1–0.3 behavior-tag nudges every 30 ticks. "Pursuing a hobby" is *definitionally* an NPC choice: when morale > 30 and economic_intent is Produce/Idle, the NPC selects a `PracticeHobby(domain)` action. The domain is `argmax(mining,trade,combat,faith)` — a derived pick, fine. The effect (tag accumulation) becomes a normal action-outcome event.
- **Required NPC actions:** `PracticeHobby(domain)`.
- **Required derived views:** `hobby_domain(npc) = argmax_domain(behavior_profile)`.
- **Required event types:** `HobbyPracticed { npc, domain, tag_gain, tick }`.
- **Summary:** NPCs *do* hobbies; becomes a low-priority candidate action in the utility scorer.

#### addiction
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT (via `ConsumePotion` action) + DERIVABLE-VIEW (withdrawal rule)
- **Reframe:** `addiction.rs` (94 lines) uses `status_effects` debuff count as a proxy for addiction state and applies `Damage{0.5|1.5} + Slow(0.7, 3s)`. The addictive *act* (consuming something that leaves a Debuff) is an NPC choice — model it as `ConsumePotion(kind)` in the DSL action space. The withdrawal damage is a status-effect-driven rule, not a standalone system: "if entity.has_debuff(Withdrawal) and hp_ratio<0.5 → Damage(1.5)" lives in the status-effect evaluator alongside all other DoTs.
- **Required NPC actions:** `ConsumePotion(kind)`, `ConsumeIntoxicant(kind)`.
- **Required derived views:** `is_addicted(npc) = count_recent(ConsumePotion) > N within window`.
- **Required event types:** `PotionConsumed { npc, kind, tick }`, `WithdrawalTick { npc, tick }`.
- **Summary:** Consumption is agentic; withdrawal is a status-effect rule; delete system.

#### npc_relationships
- **Old:** DUPLICATIVE
- **New:** DERIVABLE-VIEW
- **Reframe:** `npc_relationships.rs` (241 lines). Emits `UpdateEntityField{Morale,+0.5}` (friends) and `AddBehaviorTags{COMBAT+0.2}` (rivals) based on behavior-profile cosine similarity. Pure secondary heuristic, never writes to `npc.relationships`. Agent_inner already maintains the primary relationship graph via event-driven updates (coworker ticks, trade partners, theory-of-mind `observe_action`). Replace with: `relationship(a,b) = f(shared_events_in_memory between a and b, decay)` and `rivalry(a,b) = behavior_tag_overlap(a,b) × combat_ratio`. Delete compute; keep zero cost.
- **Required NPC actions:** `BefriendNpc(target)`, `DeclareRival(target)` — but these are rare; most relationships are emergent from co-experience, which is already `memory.events`.
- **Required derived views:** `relationship(a,b)`, `rivalry(a,b)`, `top_friend(npc)`.
- **Required event types:** none (reuses `MadeNewFriend`, `TradedWith` in memory.events).
- **Summary:** Relationships are a view over shared memory events.

#### npc_reputation
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** DERIVABLE-VIEW (for reputation) + DEAD (for the heal pulse)
- **Reframe:** `npc_reputation.rs` (74 lines) doesn't compute reputation at all — it emits a `Heal{5}` for every injured NPC at a settlement that has a healer (by `class_tags contains "healer"|"cleric"|"priest"`). That's a passive-aura effect that belongs to the healer's `PassiveEffects.aura_radius` (already on NpcData) or to a building-sourced Heal. Actual reputation should be a view: `reputation(npc) = Σ witnessed_memory_events involving npc, weighted by settlement observers`. Delete heal pulse; keep reputation as a view.
- **Required NPC actions:** none (healer's "tend wounded" is their active action).
- **Required derived views:** `reputation(npc) = Σ_witnessed_events_in_settlement weighted`, `has_healer(settlement)`.
- **Required event types:** `Witnessed { observer, subject, event_kind, tick }`.
- **Summary:** Reputation is a derived view; the heal pulse folds into the passive-aura evaluator.

#### romance
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `romance.rs` (169 lines). Pair-wise within settlement (capped at 32 NPCs near Inn/Temple/Market/GuildHall), 15% pair roll + cosine-similarity gate (`AFFINITY_THRESHOLD=0.5`), emits `DIPLOMACY+0.3, RESILIENCE+0.2` tag nudges and a chronicle entry on strong bonds. No persistent romance state — no `romantic_partner_id`, no "in love" flag. The effect is the behavior-tag nudge, identical in pattern to `hobbies`. Replace with agentic actions: an NPC with high compatibility + proximity + morale>30 selects `Court(target)` as a candidate action when loitering at a social building; repeated successful `Court` actions emit `Courted { a, b, tick }` events; the derived view `romantic_interest(a,b) = decayed_count(Courted events between a and b)` drives further escalation (into `ProposeMarriage`). The pair-cap at 32 and the social-building proximity check both fall out naturally as gates on `Court`'s utility score. Marriage is a separate pair of actions (`ProposeMarriage` + `AcceptMarriage`). This also gives the system a missing mechanism: one-sided pursuit, rejection, courtship rivalry — all naturally emergent from two NPCs making independent decisions.
- **Required NPC actions:** `Court(target)`, `FlirtAt(target)` (lighter, non-committal variant), `RebuffAdvances(suitor)` (explicit rejection — raises suitor's grief/anger).
- **Required derived views:** `compatibility(a,b) = cosine(behavior_profile_a, behavior_profile_b)`, `romantic_interest(a,b) = decayed_count(Courted events)`, `social_building_nearby(npc) = any Inn|Temple|Market|GuildHall within R`.
- **Required event types:** `Courted { a, b, tick }`, `CourtshipRebuffed { suitor, target, tick }`.
- **Summary:** Courting is a choice; emergent drift becomes a candidate action at social buildings; pair iteration disappears.

#### rivalries
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** `rivalries.rs` (107 lines). All phases commented out. Query helpers return hardcoded zeros. Rivalry is fully derivable from `relationship(a,b) = f(memory.events where both present, weighted negative by combat/conflict events)`. Delete the compute body; if a `DeclareRival` action is ever added, it emits `RivalryDeclared { a, b, reason }`.
- **Required NPC actions:** `DeclareRival(target)` (optional — mostly emergent).
- **Required derived views:** `rivalry(a,b)`.
- **Required event types:** none (already covered by memory).
- **Summary:** Delete; rivalries are a view.

#### companions
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** DERIVABLE-VIEW
- **Reframe:** `companions.rs` (115 lines). Emits `+0..+1.5 morale` scaled by `min(3, friendly_neighbors) × 0.5`. No companion state. This is a transparent ambient-morale function: `morale += ambient_social_boost(grid_friendly_count)`. Fold into the `mood(npc)` derivation; no system needed.
- **Required NPC actions:** none.
- **Required derived views:** `ambient_social_boost(grid, npc) = min(3, friendly_count) × 0.5`.
- **Required event types:** none.
- **Summary:** Pure grid-count lookup; moves into `mood()` view.

#### party_chemistry
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `party_chemistry.rs` (75 lines). `pop≥2 → always-on +5% buff`. Zero chemistry tracked. This is a settlement property, not an NPC one: `settlement.ambient_buff = if pop>=2 { 1.05 } else { 1.0 }`. Evaluate at damage/stat read sites.
- **Required NPC actions:** none.
- **Required derived views:** `settlement_ambient_buff(settlement)`.
- **Required event types:** none.
- **Summary:** Replace with a settlement-level passive multiplier read at the combat math.

#### nicknames
- **Old:** STUB/DEAD
- **New:** DEAD (replaced by DERIVABLE-VIEW)
- **Reframe:** `nicknames.rs` (93 lines). All granting code commented; no `GrantNickname` delta exists. Replace with `nickname(npc) = top_tag_name(npc.behavior_profile)` as a query. If nicknames need to persist (titles-style), bake into `titles` as a separate tier.
- **Required NPC actions:** none.
- **Required derived views:** `nickname(npc) = top_tag_name(behavior_profile)`.
- **Required event types:** none.
- **Summary:** Delete system; nickname is a view.

#### legendary_deeds
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `legendary_deeds.rs` (103 lines). Fires chronicle + morale bump on 15 behavior-profile thresholds × 2 tiers. No state persists; `deeds: Vec<u8>` on NpcData is never written. Collapse to: `is_legendary(npc) = any(npc.behavior_value(tag) >= deed_table[tag].threshold)`, and `recent_legendary_crossings(npc, window) = deed_table × behavior_history in window`. Chronicle generation becomes a subscriber to the general `BehaviorThresholdCrossed` event emitted once when `behavior_value` crosses a band.
- **Required NPC actions:** none (thresholds crossed as side-effect of action outcomes).
- **Required derived views:** `is_legendary(npc)`, `legendary_tier(npc, tag)`, `chronicle_mentions(npc)`.
- **Required event types:** `BehaviorThresholdCrossed { npc, tag, tier, tick }`.
- **Summary:** Legendary status is a view over the behavior profile; scan-and-fire deletes.

#### folk_hero
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** `folk_hero.rs` (101 lines). Body empty. Fame/folk-hero status is cleanly derivable: `fame(npc) = Σ chronicle_mentions(npc) + Σ witnessed_positive_events`. If settlements should ever *choose* to elevate a local hero, it's an agentic action (`CrownFolkHero(target)` taken by a settlement leader NPC). Delete compute body.
- **Required NPC actions:** `CrownFolkHero(target)` (leader-only).
- **Required derived views:** `fame(npc)`, `folk_hero_of(settlement) = argmax_fame(residents)`.
- **Required event types:** `FolkHeroCrowned { npc, settlement, tick }`.
- **Summary:** Delete; fame is a view, crowning is an action.

#### trophies
- **Old:** STUB/DEAD
- **New:** DEAD (or EMERGENT if revived)
- **Reframe:** `trophies.rs` (84 lines). All bodies empty. If trophies are ever implemented, the acquisition is agentic: NPC loots a `Trophy { source_kill_id }` as part of an `ClaimTrophy(corpse)` action; the passive bonuses are a status-effect rule keyed off inventory contents. Delete compute.
- **Required NPC actions:** `ClaimTrophy(corpse)`.
- **Required derived views:** `trophy_bonus(npc) = Σ trophy_effects(inventory)`.
- **Required event types:** `TrophyClaimed { npc, kind, source, tick }`.
- **Summary:** Delete; trophy claim is an action, passive bonus is a view.

#### mentorship
- **Old:** ESSENTIAL (partial)
- **New:** EMERGENT
- **Reframe:** `mentorship.rs` (176 lines). Currently `compute_mentorship` scans per-settlement, finds Library/Workshop/GuildHall at `construction_progress≥1.0` (lines 44–57), pre-collects up to 128 NPCs near one (within 15 units squared = 225, lines 77–81), skips NPCs in High-fidelity grid (combat), insertion-sorts by level descending (lines 94–100), and pairs mentors (`mentor.level ≥ apprentice.level + 3`) with up to 2 apprentices each. Emits: `Heal{xp×0.05}` to apprentice, `AttackDamage+0.1` base stat gain, staggered `Hp+10` level-up, `AddBehaviorTags{TEACHING+2,LEADERSHIP+1}` to mentor and `DISCIPLINE+1` to apprentice. These stat gains *feel* essential only because a system has to issue them — but the natural model is: apprentice chooses `Train(teacher=mentor_id, building=X)` as an action; mentor chooses `Teach(student=app_id)`; the successful pairing emits `TrainingSession { mentor, student, skill_gain, stat_deltas, tick }`; stat deltas are *the effect of the action* (same way harvesting is the effect of `Harvest` — the inventory write is essential, but it happens as the action's consequence, not from a global scan). No scan-all-pairs system needed — both sides opt in. Bonus: this removes the arbitrary `mentor_used ≤ 2` quota (replaced by natural contention for the mentor's time slot) and the fidelity-grid skip (apprentice-initiated: if you're in combat, you can't choose Train).
- **Required NPC actions:** `Train(teacher, building)` (apprentice), `Teach(student)` (mentor — can be implicit matching), `OfferApprenticeship(target)` (mentor-initiated recruitment).
- **Required derived views:** `eligible_mentors(npc, settlement) = residents.level >= npc.level + 3 near knowledge_building`, `xp_gain(mentor_level, apprentice_level) = BASE_MENTOR_XP × (1 + mentor_level/20)`, `stat_gain_per_session(apprentice_level)`.
- **Required event types:** `TrainingSession { mentor, student, xp_gain, stat_deltas, tick }`, `LevelUp { npc, tier, tick }`, `ApprenticeshipFormed { mentor, student, tick }`.
- **Summary:** Training is a mutual action at a knowledge building; delete the scanning compute; the stat-delta writes still happen but as the action's handler, not a pass.

#### marriages
- **Old:** STUB/DEAD
- **New:** DEAD — replaced by two EMERGENT actions
- **Reframe:** `marriages.rs` (82 lines). Entire body commented. The actual `spouse_id`/`children` link is set by `family.rs` (not in this batch). In the strict model, marriage formation is unambiguously agentic: `ProposeMarriage(target)` + `AcceptMarriage(suitor)`; the formal link write (`spouse_id = partner_id`) happens on accept and emits `MarriageFormed { a, b, tick }`. Children are a separate biological process. Delete this file entirely.
- **Required NPC actions:** `ProposeMarriage(target)`, `AcceptMarriage(suitor)`, `RefuseMarriage(suitor)`, `AnnulMarriage(spouse)`.
- **Required derived views:** `marriage_eligibility(a,b)`, `is_married(npc)`.
- **Required event types:** `MarriageProposed { a, b, tick }`, `MarriageFormed { a, b, tick }`, `MarriageAnnulled { a, b, tick }`.
- **Summary:** Delete; replaced by two agentic actions with a formal link write on accept.

#### bonds
- **Old:** STUB/DEAD + DUPLICATIVE
- **New:** DEAD (compute) + DERIVABLE-VIEW (helpers)
- **Reframe:** `bonds.rs` (166 lines). Not dispatched. The compute body iterates O(n²) pairs and does nothing (`let _ = (a,b);`). The query helpers (`bond_strength`, `morale_bonus`, `combat_power_multiplier`, `has_battle_brothers`, `average_group_bond`) over `state.adventurer_bonds: HashMap<(u32,u32), f32>` are read by other systems — keep them, but redefine `bond_strength` as a view over shared `memory.events` (shared-battles count, co-survived count) and drop the `adventurer_bonds` HashMap. Bond growth is then automatically event-sourced.
- **Required NPC actions:** none (bonds form from shared combat experience, which is already memory events).
- **Required derived views:** `bond_strength(a,b) = f(shared_battle_events, shared_survival_events, decay)`, `has_battle_brothers(npc)`.
- **Required event types:** none (reuses `WasInBattle`, `WonFight` memory events).
- **Summary:** Delete compute; rewrite helpers as views over memory.events.

#### memorials
- **Old:** DUPLICATIVE (not dispatched)
- **New:** DEAD
- **Reframe:** `memorials.rs` (118 lines). Not dispatched (already commented out in mod.rs: "redundant (grief morale handled by agent_inner)"). Grief on death is precisely `f(memory.events, FriendDied records)`. Delete. If commemorative monuments ever exist, they're buildings constructed via a `BuildMemorial(target_corpse)` action.
- **Required NPC actions:** `BuildMemorial(target)` (optional).
- **Required derived views:** `grief(npc)` already in the emotion view.
- **Required event types:** `MemorialBuilt { builder, target, tick }` (optional).
- **Summary:** Delete; grief is a view, memorials are buildings.

#### journals
- **Old:** STUB/DEAD (not dispatched)
- **New:** DEAD
- **Reframe:** `journals.rs` (86 lines). Not dispatched. Body is empty conditionals. Comment in mod.rs says it all: "journal state not on entities, all logic in agent_inner memory." `memory.events` IS the journal. Delete.
- **Required NPC actions:** `WriteJournal(content)` if exportable narrative is ever wanted (purely cosmetic side-effect of reflection).
- **Required derived views:** `journal(npc) = format_memory_events(npc)`.
- **Required event types:** none (memory.events already serve).
- **Summary:** Delete; memory.events IS the journal.

#### cultural_identity
- **Old:** ESSENTIAL (context_tags) + EMERGENT (reinforcement)
- **New:** DERIVABLE-VIEW + EMERGENT (adoption)
- **Reframe:** `cultural_identity.rs` (162 lines). `advance_cultural_identity` (line 53) aggregates resident behavior profiles every 500 ticks and mutates `settlement.context_tags` to hold the dominant culture tag out of 8 candidate clusters (Warrior/Scholar/Merchant/Artisan/Farming/Survivor/Seafaring/Storytelling — lines 90–98). Then reinforces: `+0.5` bonus tag to each resident (lines 121–131) and pushes a chronicle entry on first solidification (lines 152–159). The aggregation itself is a pure function: `settlement_culture(sid) = argmax_culture(Σ behavior_profiles of residents / npc_count)` with gates score>1.0 and residents≥5 (lines 80, 105). Compute lazily at read time — no need to write `context_tags` ahead of consumers. The reinforcement step is conceptually *adoption* — an NPC aligning with their local culture because they keep acting in culture-aligned ways. That's not a mandatory broadcast from a ruler; it's the NPC's own action outcomes reinforcing their profile. If an NPC chooses `Train` at a Library in a Scholar settlement, their RESEARCH tag naturally grows — you don't need a separate "culture gives you +0.5" pass. Delete the reinforcement loop; accept that culture is a view over what NPCs have actually been doing. Emit `CultureEmerged { settlement, culture, first_tick }` once per culture-transition (handled by a subscriber to behavior-aggregate changes), and `CultureAdopted { npc, culture, tick }` when an NPC's cultural_alignment view crosses a threshold — for chronicle/narrative purposes, not for mechanical effects.
- **Required NPC actions:** `InternalizeCulturalNorm(culture)` — implicit, emerges from other action outcomes in the culture's tag bundle.
- **Required derived views:** `settlement_culture(settlement)`, `cultural_alignment(npc) = cosine(npc.behavior_profile, settlement_culture(npc.home))`, `culture_score(settlement, culture_type)`.
- **Required event types:** `CultureEmerged { settlement, culture, first_tick }`, `CultureAdopted { npc, culture, tick }`, `CultureShifted { settlement, from, to, tick }`.
- **Summary:** Culture detection is a view over aggregate resident behavior; reinforcement deletes (it was double-counting action effects); two events for chronicle transitions.

#### titles
- **Old:** ESSENTIAL (name mutation)
- **New:** DERIVABLE-VIEW (with rare EMERGENT formalization)
- **Reframe:** `titles.rs` (105 lines) mutates `npc.name` permanently via `format!("{} {}", old_name, title)` at line 43 — which felt essential because once prepended, `" the Oathkeeper"` is stuck. But: the *decision* in `determine_title` (lines 48–105) is a pure function of `(fulfilled_oaths, spouse_alive, friend_deaths, attacks_survived, hp_ratio, gold, trade_tag_value, starvation_count, children.len)`. The priority order (Oathkeeper → Avenger → Bereaved → Unbroken → Merchant Prince → Enduring → Patriarch) is fixed logic. Evaluate `title(npc)` lazily. Render `"{npc.base_name} {title(npc)}"` at display time — no persistent mutation. The only reason to *persist* a title is if it was formally bestowed (coronation, knighthood), which is unambiguously an agentic act by a ruler: `BestowTitle(target, title)` emits `TitleBestowed { grantor, grantee, title, tick }` and the title becomes immutable afterward. Informal titles (Bereaved, Unbroken, Patriarch) are derivations; formal titles are events.
- **Required NPC actions:** `BestowTitle(target, title)` (ruler only), `RevokeBestowal(target)` (ruler retracts a formal title — rare).
- **Required derived views:** `title(npc)`, `has_formal_title(npc)`, `display_name(npc) = "{base_name} {title(npc)}"`.
- **Required event types:** `TitleBestowed { grantor, grantee, title, tick }`, `TitleRevoked { grantor, grantee, title, tick }`.
- **Summary:** Informal titles are a view over state + oath log; formal titles are an agentic bestowal event; `npc.name` mutation is deleted in favor of `display_name(npc)`.

#### oaths
- **Old:** ESSENTIAL
- **New:** EMERGENT
- **Reframe:** `oaths.rs` (159 lines). Currently `advance_oaths` (line 35) runs three phases every 200 ticks: Phase 1 (lines 42–83) rolls a 5% swear chance per NPC across 3 kinds (Loyalty/Vengeance/Protection), Phase 2 (lines 86–130) checks fulfillment via kind-specific conditions (vengeance = target dead; protection = 1000 ticks elapsed; loyalty = 1500 ticks AND still at home settlement), Phase 3 (lines 133–153) marks hostile Loyalty-oath holders as broken and bumps DECEPTION tags. Swearing an oath is the canonical agentic act: a random 5% roll is a placeholder for "NPC chose to swear." Replace: `SwearOath(kind, target)` is an NPC action chosen when `memory.events` + emotion state warrant it (grudge-belief present → Vengeance; high Faith+Discipline + home settlement → Loyalty; settlement under threat → Protection). Fulfillment and breaking are pure derivations: `oath_fulfilled(oath, state) = condition_met(oath.kind, state)`, `oath_broken(oath, state) = team_changed_after_swearing || abandoned_home_settlement`. The `oaths: Vec<Oath>` field becomes an append-only log (structural commitment persists); fulfillment/broken *flags* become views over `(oath, current state)` — no need to mutate `.fulfilled=true` on a polling pass; just evaluate whenever someone asks. Chronicle "3+ fulfilled oaths → Oathkeeper" is handled by `titles.rs`.
- **Required NPC actions:** `SwearOath(kind, target_id)`, `BreakOath(oath_idx)` (explicit betrayal — distinct from derived "broken by team-switch"), `AttemptFulfillOath(oath_idx)` (for Vengeance: actively pursue target).
- **Required derived views:** `fulfilled_oaths(npc)`, `active_oaths(npc)`, `broken_oaths(npc)`, `vengeance_target(npc) = first BeliefType::Grudge`, `oath_pride_bonus(npc) = 0.6 × fulfilled_count`.
- **Required event types:** `OathSworn { swearer, kind, target, terms, tick }`, `OathFulfilled { oath_id, tick }` (fired once when the fulfillment-view transitions to true), `OathBroken { oath_id, tick }`.
- **Summary:** Swearing is an agentic action; the Oath struct is immutable after swearing; fulfillment/break are views that emit one-shot events on transition.

#### grudges
- **Old:** STUB/DEAD
- **New:** DEAD (compute) + DERIVABLE-VIEW (status)
- **Reframe:** `grudges.rs` (105 lines). Body all commented; no grudge struct in state. Query helpers (`grudge_combat_bonus`, `acts_recklessly`) currently return hardcoded defaults. Grudges already live as `BeliefType::Grudge(target_id)` in `npc.memory.beliefs`, which is populated by agent_inner's belief-formation logic from memory events (a sequence of `WasAttacked` or `FriendDied` entries mentioning the same entity forms a Grudge belief). oaths.rs reads this belief for Vengeance target selection. A grudge is literally "I remember entity X harmed my friend/me" — exactly what `memory.beliefs.filter(BeliefType::Grudge)` returns. The grudge strength is `count(negative memory events mentioning target) × decay`. There is no separate grudge state to maintain. Delete the compute body; rewrite the two query helpers as views over beliefs/memory. `Forgive(target)` is the only agentic action that needs to exist — an NPC explicitly retiring a grudge belief after reconciliation — but this is optional; most grudges naturally fade via the existing 0.95/tick belief decay.
- **Required NPC actions:** `Forgive(target)` (optional — explicitly retires a grudge belief).
- **Required derived views:** `grudges(npc) = memory.beliefs.filter(Grudge)`, `grudge_intensity(npc, target) = count_negative_memory_events(npc, target) × decay`, `grudge_combat_bonus(npc, target) = grudge_intensity × scale`, `acts_recklessly(npc) = max_grudge_intensity(npc) > threshold`.
- **Required event types:** `GrudgeFormed { holder, target, cause_event_id, tick }` (derived — fires when belief crystallizes from accumulated memory events), `GrudgeResolved { holder, target, tick }` (fires on Forgive action or natural belief decay below threshold).
- **Summary:** Grudges are beliefs are memory derivations; delete system; keep helpers as views.

#### secrets
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT + DERIVABLE-VIEW
- **Reframe:** `secrets.rs` (217 lines). Six secret types × 2 no-ops, keyed by `entity.id % 8` (deterministic at spawn), firing once per threshold crossing within a narrow window. Writes: `Morale ±5..±15` on reveal, `AddBehaviorTags` related-skill boost, `UpdateTreasury{-stolen}` for spy-type reveals, and a chronicle entry. Two separate concerns need unbundling: (1) the *existence* of a secret is fixed at birth — better to emit `SecretAssigned { npc, kind, tick }` at character creation (via birth-time subscriber to `NpcSpawned`) and evaluate `has_secret(npc, kind)` by querying that event log, rather than recomputing `id % 8` on every pass. This retains determinism while making the secret first-class. (2) The *reveal condition* (behavior-tag sum crossing a threshold window) is a pure derivation from action history — subscribe to `BehaviorThresholdCrossed` and emit `SecretRevealed { npc, kind, observer, tick }`; the morale shift and tag boost happen as the event handler's side-effects, not from a scanning pass. (3) The spy-type treasury drain IS a real conserved economic transfer — that fires a `TransferGold(settlement → 0, stolen_amount)` in the reveal handler. The system compute body deletes; the reveal handler is a handful of lines in the event subscriber layer. Alternative model: NPCs could *choose* to reveal a secret via an explicit `ConfessSecret` action (e.g. under stress or in confession to a priest), giving the system dual pathways — forced reveal via threshold, voluntary reveal via action.
- **Required NPC actions:** `RevealSecret(target)` (deliberate reveal to a specific observer), `HideSecret` (active concealment — raises stress), `ConfessSecret` (voluntary unburdening, usually to priest or friend).
- **Required derived views:** `has_secret(npc, kind) = any SecretAssigned events for npc`, `reveal_risk(npc) = f(stress, behavior_crossings_near_threshold)`, `known_secrets(observer) = SecretRevealed events where observer matches`.
- **Required event types:** `SecretAssigned { npc, kind, tick }` (birth-time), `SecretRevealed { npc, kind, observer, tick }`, `SecretKept { npc, kind, stress_delta, tick }` (fires on near-miss threshold for stress feedback).
- **Summary:** Secret assignment is a birth-time event; reveal is either an action or a derived threshold event; compute body deletes.

#### intrigue
- **Old:** STUB/DEAD
- **New:** DEAD (to be rebuilt as EMERGENT)
- **Reframe:** `intrigue.rs` (63 lines). Body commented. Real intrigue — plots, alliances, betrayals — is *entirely* agentic: `BeginPlot(goal, targets)`, `RecruitConspirator(target)`, `BetrayFriend(target)`, `InformOn(conspirator)`. No scanning system can capture it; the whole mechanic is NPC decisions. Delete skeleton; when rebuilt, it's a tree of agentic actions with `Plot` as an aggregate entity.
- **Required NPC actions:** `BeginPlot(goal, targets)`, `RecruitConspirator(target)`, `ExecutePlot(plot_id)`, `InformOn(conspirator)`, `BetrayFriend(target)`.
- **Required derived views:** `active_plots(region)`, `conspirator_count(plot)`, `plot_readiness(plot)`.
- **Required event types:** `PlotBegun`, `PlotJoined`, `PlotExecuted`, `PlotBetrayed`, `ConspiratorInformed`.
- **Summary:** Delete; intrigue is a constellation of agentic actions, not a system.

#### religion
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** EMERGENT (joining a faith) + DERIVABLE-VIEW (heal aura)
- **Reframe:** `religion.rs` (69 lines). Heals injured NPCs and adds `FAITH+1, RITUAL+0.5` at settlements with treasury≥50 (proxying for "temple active"). The *faith-follower* status is agentic: `JoinReligion(faith)`, `PerformRitual(faith)`, `LeaveReligion`. Heal-aura should come from Temple building's `PassiveEffects.aura_radius`, not a treasury proxy. Delete the system; attach effects to the Temple entity.
- **Required NPC actions:** `JoinReligion(faith)`, `PerformRitual(faith, target)`, `LeaveReligion`.
- **Required derived views:** `religion_of(npc)`, `faithful_count(faith, settlement)`.
- **Required event types:** `ReligionJoined { npc, faith, tick }`, `RitualPerformed { priest, faith, target, tick }`.
- **Summary:** Religion adherence is an action; heal aura is a building passive; delete the treasury-proxy compute.

#### demonic_pacts
- **Old:** EMERGENT-CANDIDATE
- **New:** EMERGENT
- **Reframe:** `demonic_pacts.rs` (114 lines). Uses `status_effects` debuff count as a proxy for pact tier. Three tiers of consequence: 1+ debuff → `Damage{1.0}`, 2+ debuffs → `TransferGold{10.0}` pact-holder-to-home-settlement (tithe to a demonic patron via a regular-settlement account — economically coherent because the demon is presumably a settlement-aligned cultist), 3+ debuffs → 15% roll for `Damage{15.0}` to a random ally (possession). The proxy-via-status-count pattern is the tell: the system lacks a `Pact` entity, so it's reading whatever's handy. Fix properly: `SignDemonicPact(demon, terms)` is an NPC action chosen when desperate (low-hp, starving, grieving — high need urgency + negative emotions); emits `PactSigned { npc, demon, clauses, tick }` which writes a `Pact` log entry; each clause then runs as its own declared status-effect rule in the status-rule evaluator (e.g. "if PactTier ≥ 2, emit TransferGold(npc → demon_settlement, 10) every 7 ticks"). Breaking the pact (`BreakPact`) is an action that emits `PactBroken` and strips the clauses. The system compute body deletes; the clauses become declarative status rules; all three mechanical effects happen as rule consequences, not from a scan.
- **Required NPC actions:** `SignDemonicPact(demon, terms)`, `BreakPact(demon)`, `PerformBlackRitual(demon)` (maintenance / power-up).
- **Required derived views:** `pact_tier(npc) = count(active_pact_clauses(npc))`, `is_cursed(npc) = pact_tier(npc) > 0`, `pact_patrons(npc) = distinct demons in active pacts`.
- **Required event types:** `PactSigned { npc, demon, clauses, tick }`, `PactBroken { npc, demon, tick }`, `PactTithe { npc, amount, recipient, tick }`, `PossessionStrike { npc, victim, tick }`.
- **Summary:** Signing is an action; clauses are declarative status rules; the scan deletes.

#### divine_favor
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** DERIVABLE-VIEW
- **Reframe:** `divine_favor.rs` (99 lines). Uses `settlement.treasury` buckets (>100 heal+treasury, <10 damage) — a "prosperity feedback." No divine_favor state. Collapse with `religion`, `npc_reputation`, and `reputation_stories` into a single settlement-level derivation: `settlement_divine_blessing(s) = f(treasury, faith_count, recent_rituals)` → small heal/damage/treasury deltas at read sites. No system needed.
- **Required NPC actions:** none.
- **Required derived views:** `settlement_divine_blessing(settlement)`, `settlement_divine_wrath(settlement)`.
- **Required event types:** `DivineFavorTicked { settlement, delta, tick }` (optional chronicle).
- **Summary:** Treasury proxy collapses into a shared settlement derivation; delete.

#### biography
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW (explicit)
- **Reframe:** `biography.rs` (569 lines) is *already* a pure function returning a `String` — it's never called by the simulation, only by UI/CLI. It's not a system. Move out of `systems/` entirely into a `views/` or `queries/` module. Zero per-tick cost today; should have been zero DSL surface too.
- **Required NPC actions:** none.
- **Required derived views:** `biography(npc) -> String` — exactly what the file does.
- **Required event types:** none.
- **Summary:** Not a system; relocate to a query module.

#### reputation_stories
- **Old:** EMERGENT-CANDIDATE
- **New:** DERIVABLE-VIEW
- **Reframe:** `reputation_stories.rs` (47 lines). 12 lines of logic: rich settlements get `treasury+=3`, poor get `treasury-=2`. That's a derived economic rule — settlement's treasury drift as a function of its own treasury. Not "stories," not reputation. Fold into the settlement economy step or delete as trivial.
- **Required NPC actions:** none.
- **Required derived views:** `treasury_momentum(settlement)`.
- **Required event types:** none.
- **Summary:** Two-branch drift rule; fold into economy or delete.

#### wanted
- **Old:** ESSENTIAL (partial — bounty payouts + threat reduction)
- **New:** EMERGENT
- **Reframe:** `wanted.rs` (260 lines, `compute_wanted` at line 44). Three phases: (1) settlements with `threat_level ≥ MIN_THREAT_FOR_WANTED=0.15` (line 64) post `WorldEvent::Generic` bounty posters for nearby hostile monsters/faction NPCs within `WANTED_RANGE_SQ=900` (line 81); (2) when hostiles die near a grid, share bounty among killers via `TransferGold(settlement → collector)`; (3) reduce settlement threat by `dead_hostile_count × 0.02`. Under the strict rubric: posting a bounty is unambiguously the settlement leader's agentic choice — a global scan is a proxy for "every frustrated ruler auto-posts." Replace with `PostBounty(target, reward)` taken by a ruler NPC (or by a mayor/council member via council.rs); it emits `BountyPosted { target, reward, funder, tick }` which writes a row to a `WorldState.bounties` log. Paying out is a derivation from the `EntityDied` event matched against outstanding bounties — runs as an event subscriber on death, not a scanning pass. The `TransferGold` IS a real physics write (gold is conserved), so that stays — but it fires as a one-shot reducer inside the death-event handler, triggered by the event, not by a pass over entities. Threat reduction (`threat -= 0.02 × recent_hostile_deaths_near(settlement)`) is a trivial derivation — either lazily computed at read time or updated incrementally by the same death-event handler. Result: zero per-tick cost when there are no deaths; essential writes still happen, but triggered by events not cadence.
- **Required NPC actions:** `PostBounty(target, reward)` (ruler), `ClaimBounty(corpse, poster)` (killer — implicit from kill event but explicit for contested corpses).
- **Required derived views:** `active_bounties(settlement)`, `bounty_on(target) = max posted bounty`, `bounty_collectors_near(corpse, radius)`.
- **Required event types:** `BountyPosted { target, reward, funder, tick }`, `BountyClaimed { claimant, target, funder, reward, tick }`, `BountyExpired { target, tick }`, `ThreatReduced { settlement, amount, cause, tick }`.
- **Summary:** Posting is agentic; payouts and threat-reduction are event-driven reducers on `EntityDied`; compute body deletes.

---

### Reduction summary

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

#### Bucket counts

| Bucket | v1 | v2 |
|---|---|---|
| ESSENTIAL | 7 | 2 (agent_inner reduced + action_eval) |
| EMERGENT | 15 | 12 (agentic actions) |
| DERIVABLE-VIEW | 0 | 13 |
| DEAD/STUB | 10 | 10 (all stubs stay dead; 8 ex-live systems join them as "delete") |

v1's "EMERGENT-CANDIDATE" label collapsed nearly everything that was "small drift + tag accumulation" into a single bucket. v2 splits them: if the field is a pure function, it's a VIEW; if the write represents an NPC *choosing* to do something, it's EMERGENT and becomes an action. The essential set shrinks from 7 to 2 — with `agent_inner` further reduced to its event-emitter + needs-physics core.

---

### Required action vocabulary

The DSL action space must grow from ~13 mechanical actions (Eat, Harvest, Attack, Flee, Build, Work, Move, Idle, …) to cover the social/cultural/political surface now done by scanning systems. Verbs required:

#### Relationships & family
- `Court(target)` — flirt/romance progression
- `ProposeMarriage(target)`
- `AcceptMarriage(suitor)`
- `RefuseMarriage(suitor)`
- `AnnulMarriage(spouse)`
- `BefriendNpc(target)` (optional — mostly emergent from shared events)
- `DeclareRival(target)` (optional)
- `BetrayFriend(target)`
- `Forgive(target)`

#### Oaths, titles, faith
- `SwearOath(kind, target_id)`
- `BreakOath(oath_idx)`
- `AttemptFulfillOath(oath_idx)`
- `BestowTitle(target, title)` — ruler-only
- `JoinReligion(faith)`
- `LeaveReligion`
- `PerformRitual(faith, target)`
- `SignDemonicPact(demon, terms)`
- `BreakPact(demon)`

#### Skill & self-improvement
- `Train(teacher, building)` — apprentice-initiated
- `Teach(student)` — mentor-initiated
- `PracticeHobby(domain)`
- `PursueGoal(kind)` / `DeclareMastery(domain)`
- `ConsumePotion(kind)` / `ConsumeIntoxicant(kind)`
- `WriteJournal(content)` (optional cosmetic)

#### Bounties & law
- `PostBounty(target, reward)` — settlement leader
- `ClaimBounty(corpse, poster)`
- `MarkWanted(target)`

#### Secrets & intrigue
- `RevealSecret(target)` / `HideSecret` / `ConfessSecret`
- `BeginPlot(goal, targets)` / `RecruitConspirator` / `ExecutePlot(plot_id)` / `InformOn(conspirator)`

#### Commemoration
- `BuildMemorial(target)` (optional)
- `ClaimTrophy(corpse)` (optional)
- `CrownFolkHero(target)` (settlement leader, optional)

---

### Required event types

Event-sourcing is the backbone of the new model. Every derived view needs an event history to fold over. Required events:

#### Central observation event
- `Witnessed { observer, subject, event_kind, tick, impact }` — the primary emission channel for reputation, relationship, and mood derivations. Drops into `npc.memory.events` of the observer.

#### Needs & physics
- `NeedsTicked { npc, Δhunger, Δshelter, Δsocial, tick }`
- `ResourceObserved { observer, resource_id, kind, pos, tick }`

#### Agent decisions
- `ActionChosen { npc, action, target, score, tick }`
- `GoalAchieved { npc, kind, tier, tick }`
- `HobbyPracticed { npc, domain, tag_gain, tick }`
- `BehaviorThresholdCrossed { npc, tag, tier, tick }` — umbrella for legendary-deed/secret/title firing

#### Relationships & family
- `Courted { a, b, tick }`
- `MarriageProposed / MarriageFormed / MarriageAnnulled { a, b, tick }`
- `RivalryDeclared { a, b, reason, tick }`
- `MadeNewFriend { a, b, tick }` (exists — keep)
- `TradedWith { a, b, tick }` (exists — keep)

#### Oaths, titles, faith
- `OathSworn { swearer, kind, target, terms, tick }`
- `OathFulfilled { oath_id, tick }`
- `OathBroken { oath_id, tick }`
- `TitleBestowed { grantor, grantee, title, tick }`
- `ReligionJoined { npc, faith, tick }`
- `RitualPerformed { priest, faith, target, tick }`
- `PactSigned { npc, demon, clauses, tick }`
- `PactBroken { npc, demon, tick }`
- `PactTithe { npc, amount, tick }`

#### Training & progression
- `TrainingSession { mentor, student, xp_gain, stat_deltas, tick }`
- `LevelUp { npc, tier, tick }`

#### Consumption & status
- `PotionConsumed { npc, kind, tick }`
- `WithdrawalTick { npc, tick }` (or fold into status-effect eval log)

#### Bounties & law
- `BountyPosted { target, reward, funder, tick }`
- `BountyClaimed { claimant, target, funder, reward, tick }`
- `BountyExpired { target, tick }`

#### Secrets & intrigue
- `SecretAssigned { npc, kind, tick }` (birth-time)
- `SecretRevealed { npc, kind, observer, tick }`
- `PlotBegun / PlotJoined / PlotExecuted / PlotBetrayed / ConspiratorInformed`

#### Culture
- `CultureEmerged { settlement, culture, first_tick }`
- `CultureAdopted { npc, culture, tick }`

#### Commemoration
- `FolkHeroCrowned { npc, settlement, tick }` (optional)
- `MemorialBuilt { builder, target, tick }` (optional)
- `TrophyClaimed { npc, kind, source, tick }` (optional)

#### Grudges
- `GrudgeFormed { holder, target, cause_event_id, tick }` (derived from memory events)
- `GrudgeResolved { holder, target, tick }`

---

### Required derived views

These are all pure functions — no storage, no per-tick writes. Computed on demand at read sites, optionally memoized at tick boundaries for hot paths.

#### Per-NPC state
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

#### Social / reputation
- `reputation(npc) = Σ Witnessed events involving npc, weighted by observer settlement`
- `fame(npc) = Σ chronicle_mentions(npc) + Σ witnessed_positive_events`
- `nickname(npc) = top_tag_name(behavior_profile)`
- `title(npc) = determine_title(fulfilled_oaths, spouse_state, friend_deaths, attacks, gold, trade_tag, starvations, children)`
- `has_formal_title(npc) = any TitleBestowed where grantee==npc`
- `is_legendary(npc) = any behavior_value(tag) >= deed_table[tag].threshold`
- `legendary_tier(npc, tag) = thresholds_crossed(npc.behavior_value(tag))`
- `folk_hero_of(settlement) = argmax_fame(residents)`

#### Pairwise
- `relationship(a,b) = f(shared memory.events between a and b, recency, positive/negative)`
- `rivalry(a,b) = behavior_tag_overlap(a,b) × combat_ratio`
- `bond_strength(a,b) = shared_battle_events + shared_survival_events, decayed`
- `has_battle_brothers(npc) = any bond_strength(npc, _) > threshold`
- `compatibility(a,b) = cosine(behavior_profile_a, behavior_profile_b)`
- `romantic_interest(a,b) = recent Courted events / decay`
- `marriage_eligibility(a,b)`

#### Oaths
- `fulfilled_oaths(npc) = npc.oaths.filter(fulfilled)`
- `active_oaths(npc) = npc.oaths.filter(!fulfilled && !broken)`
- `broken_oaths(npc) = npc.oaths.filter(broken)`
- `oath_fulfilled(oath, state) = condition_met(oath.kind, state)`
- `oath_broken(oath, state) = team_changed || abandoned_home`
- `vengeance_target(npc) = first BeliefType::Grudge in beliefs`

#### Settlement / world
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

#### Action scoring
- `utility_score(npc, candidate) = f(needs, personality, aspiration, emotions, cultural_bias, action_outcomes_EMA, distance)` — the inside of `action_eval`, now declared in DSL
- `eligible_mentors(npc, sid) = residents.level >= npc.level + 3 near knowledge_building`
- `xp_gain(mentor_level, apprentice_level) = BASE × (1 + mentor_level/20)`
- `goal_progress(npc, kind) = npc.behavior_value(kind.tag) / kind.threshold`
- `reveal_risk(npc) = f(stress, behavior_crossings)`

#### Biography
- `biography(npc) -> String` (already pure in v1; just move out of `systems/`)
- `journal(npc) -> FormattedHistory` = format_memory_events(npc)

---

### Truly essential (irreducible) set in this batch

After strict re-classification, only **two** systems in this batch must remain as direct-mutation code:

#### 1. `agent_inner.rs` — reduced to event emitter + needs physics

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

#### 2. `action_eval.rs` — the agentic decision engine

Keep all ~1215 lines of it — but recognize that the *utility scoring rules* inside are pure derivations that should be DSL-declared, not hand-coded in Rust. The engine's essential work is: enumerate candidates, score each via `utility_score(npc, c)`, argmax, dispatch.

**Everything else deletes or becomes a view.** The per-tick cost for social systems drops from agent_inner 3.4% + ~12 mini-systems (estimated 1–2% combined) to just agent_inner's reduced core plus a handful of event-handler subscribers that fire only when their trigger event is emitted. Hot paths (`mood`, `emotions`, `relationship`, `reputation`) are memoized per tick at read sites.

#### Key insight

The v1 taxonomy ("ESSENTIAL = 7") was still counting any system that performed a state write as essential, even when that write was a pure function of readable state plus time. Under the strict rubric, the test is: *can the written field be reconstructed by replaying the event log from t=0 against the entity's birth state?* For every social-layer field except `npc.memory.events` itself (and needs physics), the answer is yes — so the system is either an event subscriber (EMERGENT) or a lazy view (DERIVABLE-VIEW), never an imperative per-tick writer.

#### Secondary observations

- **The narrow-window threshold-fire pattern is the same thing as a threshold-transition event.** v1 identified a unified `ThresholdFire` kernel merging personal_goals / legendary_deeds / secrets / titles / nicknames. Under event sourcing, this becomes even simpler: emit `BehaviorThresholdCrossed { npc, tag, tier }` whenever `behavior_value` crosses a band (detected as the side-effect of the `accumulate_tags` call), and subscribers generate chronicle entries + title flags. No kernel, no scan, no window-management.

- **Every scanning-based relationship system (npc_relationships, rivalries, bonds, romance, memorials) is reconstructable from `shared_memory_events(a, b)`.** If NPC A and NPC B are present in each other's memory.events with positive impact, they're friends. If negative, rivals. If both survived a battle (`WasInBattle` event with overlapping participants), they're battle brothers. The graph doesn't need maintenance — it's a projection over each NPC's memory.

- **Proxy-based systems (addiction via debuff count, religion/divine_favor via treasury, demonic_pacts via debuff count, moods/fears via hp_ratio) are the opposite of event sourcing: they invent a proxy state variable because they lack a proper event.** Adding the proper event (`PotionConsumed`, `PactSigned`, `RitualPerformed`, `WasAttacked`) makes the proxy unnecessary and the system deletable.

- **Stubs are stubs because the *action* was never implemented.** marriages, rivalries, nicknames, folk_hero, trophies, journals, grudges, intrigue were all stubbed because their primary verb ("get married," "become rivals," "earn a nickname," "become a folk hero," "keep a trophy," "write a journal," "hold a grudge," "plot a betrayal") is an NPC *choice* and the team correctly sensed that a scanning system couldn't capture it. The scanning body was never written because it wouldn't have been right. Under the new model, these entries become DSL action declarations, not Rust files.

- **`chronicle.push` is scattered across 6 systems in this batch.** Replace with a single `on_chronicle_event(event)` subscriber that formats event descriptions for the chronicle. Events like `OathFulfilled`, `TitleBestowed`, `CultureEmerged`, `BountyClaimed`, `MarriageFormed`, `LevelUp` all produce chronicle entries via one centralized formatter, not per-system string-building.

- **The dispatch model changes.** v1 had systems running on a fixed cadence (every 10, 17, 50, 100, 200, 500 ticks). The new model has only event subscribers (fire when event emitted) plus the two always-on essentials (`action_eval` every 5 ticks, `agent_inner` reduced core every 10 ticks). The aggregate per-tick cost for social systems drops to near-zero in quiet periods and rises naturally when things are actually happening (deaths, kills, level-ups) — which matches what you want for scalability.

### Migration shape

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

---

## Batch 3 — Combat / Quests / Threat

### Methodology

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

### Per-system reanalysis

#### battles
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

#### loot
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

#### last_stand
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

#### skill_challenges
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

#### dungeons
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

#### escalation_protocol
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

#### threat
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

#### threat_clock
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

#### adventuring
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

#### quests
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

#### quest_lifecycle
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

#### seasonal_quests
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

#### bounties
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

#### treasure_hunts
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

#### heist_planning
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

#### exploration (tile half)
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

#### exploration (voxel resource census)
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

#### movement
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

#### pathfollow
- **Old:** STUB
- **New:** DEAD
- **Reframe:** `pathfollow.rs:14-16` is a no-op. Delete.
- **One-line summary:** DEAD — delete the file.

---

#### travel
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

#### monster_ecology
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

#### monster_names
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

#### nemesis
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

#### wound_persistence
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

#### adventurer_condition
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

#### adventurer_recovery
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

#### cooldowns
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

#### death_consequences
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

#### sea_travel
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

#### scouting
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

#### messengers
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

#### goal_eval
- **Old:** DEAD (superseded by action_eval)
- **New:** DEAD
- **Reframe:** `goal_eval.rs:28` (`evaluate_goals`) is not invoked from
  the runtime. V1's audit confirmed this (replaced by `action_eval`).
  The file is ~527 lines of legacy code.
- **One-line summary:** DEAD — delete.

---

#### world_goap
- **Old:** DEAD (superseded by action_eval)
- **New:** DEAD
- **Reframe:** `world_goap.rs:206` (`evaluate_world_goap`) is not
  dispatched. ~248 lines of legacy code.
- **One-line summary:** DEAD — delete.

---

### Reduction summary

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

### Required action vocabulary

#### Core combat
- `Attack(target_id)`
- `Focus(target_id)` — reassign which enemy is "primary"
- `Flee(away_from)`
- `CastAbility(ability_id, target)`
- `PickUp(item_id)`
- `EquipItem(item_id, slot)`
- `ClaimBounty(target_id)`
- `AttemptChallenge(kind, difficulty)`

#### Quests and parties
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

#### Travel, exploration, messages
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

#### Covert / high-skill
- `PlanHeist(target)`
- `Scout(target)`
- `Infiltrate(target)`
- `ExecuteHeist(target)`
- `AbortHeist`

#### Life consequences
- `AttendFuneral(dead_id)`
- `InheritMastersTags(master_id)`
- `MourningRite(dead_id)`
- `UseMedicine(amount)`
- `Desert`

#### Monsters as NPCs
- `MigrateToward(pos)`
- `SeekFood(settlement)`
- `FleeSettlement`
- `Breed(mate)`
- `AttackSettlement(s)`

---

### Required event types

#### Essential (drive mutation)
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

#### Agentic (capture choices)
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

#### Observer (for derived views / chronicle)
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

### Required derived views

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

### Truly essential (irreducible) set in this batch

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

---

## Batch 4 — Politics / Faction / Meta

### Methodology

Under the **strict rubric**, the v1 classification was too generous — it labelled ~17 of 36 systems as ESSENTIAL because they *write* faction/settlement/entity state. That test is wrong: the question is whether the write is *irreducible physics/bookkeeping* or whether it models *a choice a simulated NPC could make*. Any state change that corresponds to a plausible in-world decision (declare war, form an alliance, abdicate, betray, marry, have a child, spy, found a town, swear a prophecy, vote in council) belongs to the **NPC action space**, not to a privileged top-down "politics system". The politics module then reduces to:

1. **ESSENTIAL** — irreducible bookkeeping: `entity.spawn` (ID alloc + `alive=true`), chronicle emission (event recording is the event log itself), and *possibly* the leader-class decision function (parallel to `action_eval` for combat NPCs). Team flip is defensible as essential *only* because the entity store currently represents allegiance as a direct field; the event-sourced alternative is `npc.faction_id = derive_from_latest(FactionMembershipChanged)`.
2. **EMERGENT** — any state change that an NPC could choose as an action. These systems are replaced by entries in the NPC action vocabulary plus reducer logic driven by the emitted event.
3. **DERIVABLE-VIEW** — pure queries over chronicle / world_events / primary state. Legendary status, era names, culture labels, reputation, faction tech tiers, war exhaustion, haunted sites, lineages — all computable on demand.
4. **DEAD/STUB** — empty bodies (`war_exhaustion`, `defection_cascade`, `faction_tech`) and noise emitters whose own source comments admit "without X state, we approximate" (`charter`, `choices`, `rival_guild`, `victory_conditions`).

Aggressively applying this rubric collapses the v1 count of 14-18 ESSENTIAL systems to a core of **4**: `chronicle` (event recording), `settlement_founding.spawn` (ID alloc only), `family.births` (ID alloc only), `faction_ai` (as the leader-level decision function). Everything else is either agentic, derivable, or dead.

---

### Per-system reanalysis

#### faction_ai
- **Old:** ESSENTIAL (owns `SettlementConquered`, regenerates `military_strength`).
- **New:** ESSENTIAL *(as decision function)*, with most *effects* reframed as EMERGENT consequences of leader-NPC actions.
- **Reframe:** `src/world_sim/systems/faction_ai.rs:12-42, 219-251` runs a per-faction AI loop: regen strength, pick a stance reaction, occasionally attack the weakest rival. Every `match` arm models a choice a **faction leader NPC** would make — "declare war on the weakest neighbour", "reclaim lost territory", "raise taxes to recover from unrest". This is structurally the same as `action_eval.rs` for combat NPCs. Treat `faction_ai` as the *decision head* for leader-class NPCs (parallel to `action_eval`), and emit each branch as an NPC action that fans out to events (`DeclareWar`, `ReclaimSettlement`, `LevyTax`, …). Strength regen = book-keeping `Tick` applied to a physics state (`military_strength`) and may remain inside the system or be derived from `Σ recruitments − casualties` across the chronicle.
- **Required NPC actions (leader):** `DeclareWar(opponent)`, `ReclaimSettlement(target)`, `LaunchConquest(target_settlement)`, `RaiseTaxes(rate)`, `RecruitMilitia()`, `SignPeace(opponent)`.
- **Required derived views:** `military_strength(f) = base + Σ recruitments − Σ casualty_events` (or retained as a primary field for perf, updated only as `RecruitmentHappened` / `UnitDied` events apply).
- **Required event types:** `WarDeclared`, `SettlementConquered`, `ReclaimAttempted`, `MilitaryRecruited`.
- **One-line:** Leader decision head stays; all war/conquest branches become leader NPC actions.

#### diplomacy
- **Old:** ESSENTIAL (modifies `relationship_to_guild`, owns trade income).
- **New:** EMERGENT.
- **Reframe:** `diplomacy.rs:21-70` iterates faction pairs, auto-adjusts `relationship_to_guild` for peaceful pairs and runs trade income. Each threshold crossing ("relation > 20 → trade income") is a *diplomatic outcome* — what actually happened in the world is that a **leader NPC of faction A proposed a trade accord** which B accepted. Relation drift under a shared threat is the result of either leader choosing `MutualDefenseTalks(third_party_threat)`. Trade income is a recurring settlement-level consequence of a standing `TradeAgreementSigned` event.
- **Required NPC actions (leader):** `ProposeTradeAccord(target)`, `OpenDiplomaticChannel(target)`, `OfferGift(target, amount)`, `BreakRelations(target)`.
- **Required derived views:** `relation(f_a, f_b, tick) = sum of TradeAccord, Alliance, WarDeclaration, GiftSent, Betrayal events with recency decay`.
- **Required event types:** `TradeAccordSigned {a, b, terms}`, `DiplomaticGiftSent {from, to, amount}`, `RelationsBroken {a, b, cause}`.
- **One-line:** Every diplomacy threshold is the consequence of a leader action, not a background drift.

#### espionage
- **Old:** ESSENTIAL (drains enemy `military_strength`, applies spy damage).
- **New:** EMERGENT.
- **Reframe:** `espionage.rs:68-100` walks hostile-faction NPCs near enemy settlements and auto-fires a drain-and-get-caught roll. This is a **commoner NPC action** (`Spy(target_faction)`) — any NPC with `STEALTH` / `DECEPTION` tags can choose to spy when positioned near an enemy town. The drain-to-enemy-strength is the consequence of a `SpyMissionSucceeded` event. The detection/wound path is the enemy settlement's *counter* action or a derived "detection chance" view, not a separate system.
- **Required NPC actions (commoner):** `Spy(target_faction)`, `Sabotage(target_settlement)`, `InfiltrateCouncil(target_faction)`.
- **Required derived views:** `spies_active(f) = Σ SpyMissionStarted − SpyMissionEnded for faction f`.
- **Required event types:** `SpyMissionStarted {agent, target_faction}`, `SpyMissionSucceeded {agent, target, impact}`, `SpyCaught {agent, defender_settlement}`.
- **One-line:** Spying is an NPC action, not a top-down drain.

#### counter_espionage
- **Old:** ESSENTIAL (kills/wounds spies).
- **New:** EMERGENT (settlement-leader or guard-NPC reaction) + DERIVABLE-VIEW (detection chance).
- **Reframe:** `counter_espionage.rs:51-185` lets each settlement passively roll detection against nearby hostiles and kills/wounds them. The action model is: settlement leader NPC chooses `StandingOrder.Counterspy` once; whenever an enemy spy event fires nearby, a guard NPC action `ArrestSpy(agent)` or `ExecuteSpy(agent)` resolves. Morale boosts for allies come from the `SpyExecuted` chronicle entry, not from a system-wide broadcast.
- **Required NPC actions (leader/guard):** `SetStandingOrder(Counterspy, level)`, `ArrestSpy(agent)`, `ExecuteSpy(agent)`, `ExileSpy(agent)`.
- **Required derived views:** `detection_strength(s) = 0.08 + sqrt(pop)*0.04 + outpost_bonus` (kept as a formula used by the arrest action's resolve check).
- **Required event types:** `SpyArrested {agent, settlement}`, `SpyKilled {agent, settlement}`.
- **One-line:** Counter-intel is a standing order + reactive guard action, not a global sweep.

#### war_exhaustion
- **Old:** STUB/DEAD (body commented out).
- **New:** DEAD + DERIVABLE-VIEW (when revived).
- **Reframe:** `war_exhaustion.rs:45-236` is entirely commented out. Don't revive it as a system — it is definitionally a derived view: `exhaustion(f) = duration_at_war × casualty_rate × treasury_drain_rate`. A leader's `SignPeace` action can *consult* this view but no system needs to write it.
- **Required NPC actions:** none.
- **Required derived views:** `war_exhaustion(f) = Σ_over_current_war (casualty_events × w1 + treasury_deltas × w2 + duration × w3)`.
- **Required event types:** none new (reads existing casualty / treasury / war events).
- **One-line:** Delete the system; keep the formula as a view consulted by `SignPeace`.

#### civil_war
- **Old:** ESSENTIAL (mutates `escalation_level`, spawns crisis).
- **New:** EMERGENT.
- **Reframe:** `civil_war.rs:55-236` auto-ignites civil war when strength < 30 & unrest > 0.70 and then drains state until collapse/loyalist-win. Civil war is *always* a chain of NPC choices: a dissident leader NPC chooses `DeclareCivilWar(loyalist_side)`, citizens choose `JoinFactionSide(rebel|loyalist)`, the outcome resolves when one side's adherents drop below a threshold (derived). `escalation_level` becomes `civil_war_phase(f) = derive_from(latest CivilWarDeclared − resolution_events)`.
- **Required NPC actions:** `DeclareCivilWar(target_leader)` (leader/council), `JoinFactionSide(side)` (any citizen), `FleeCivilWar(destination)` (citizen), `SurrenderCivilWar()` (side-leader).
- **Required derived views:** `civil_war_status(f) = { phase, rebel_support, loyalist_support, duration }` from the action events.
- **Required event types:** `CivilWarDeclared {faction, instigator, grievance}`, `CivilWarSideJoined {citizen, side}`, `CivilWarResolved {victor, mechanism}`.
- **One-line:** Civil war is a cascade of faction-join actions around an inciting declaration.

#### council
- **Old:** EMERGENT-CANDIDATE.
- **New:** EMERGENT.
- **Reframe:** `council.rs:39-99` tallies NPC faction distribution per settlement and nudges relationships/morale. Under the strict rubric, the *vote* is the core action: council members of a settlement each emit `VoteOnIssue(issue, choice)`, and the outcome is the majority view derived from the event log. The relationship/morale nudges are consequences of the resolved-vote event, not a continuous nudge.
- **Required NPC actions (council-member):** `VoteOnIssue(issue_id, choice)`, `AbstainVote(issue_id)`, `TableMotion(text)`, `FilibusterMotion(issue_id)`.
- **Required derived views:** `council_outcome(s, issue_id) = argmax(votes_by_choice)`, `council_composition(s) = histogram(npc.faction_id for home_settlement_id==s)`.
- **Required event types:** `CouncilVoteCast {voter, issue, choice}`, `CouncilMotionResolved {issue, outcome}`.
- **One-line:** Council is pure NPC voting — the system is a derived vote-tally, not a continuous relationship drift.

#### coup_engine
- **Old:** ESSENTIAL (owns `coup_risk`, forces regime change).
- **New:** EMERGENT.
- **Reframe:** `coup_engine.rs:33-127` accumulates a `coup_risk` scalar per faction, then rolls for a coup attempt. A coup is the archetypal NPC action: an ambitious member of the ruling class chooses `LaunchCoup(target_leader)`. The success/failure branch resolves based on immediate state (loyalty, garrison, outside support), not a pre-rolled `coup_risk` scalar. Risk factors (unrest, treasury, escalation) become the *utility inputs* the ambitious NPC uses when deciding whether to act, not a stored resource.
- **Required NPC actions (ambitious member):** `LaunchCoup(target_leader)`, `PledgeLoyalty(leader)`, `BribeGarrison(amount)`, `FleeAfterFailedCoup(destination)`.
- **Required derived views:** `coup_conditions(f) = {unrest_avg, treasury_ratio, escalation, leader_legitimacy}` — fed as inputs to the ambitious NPC's action score.
- **Required event types:** `CoupAttempted {instigator, target_leader, success}`, `CoupSuppressed {instigator, defender}`, `RegimeChanged {faction, new_leader, mechanism}`.
- **One-line:** Coup is an action; `coup_risk` becomes the utility function evaluating it.

#### defection_cascade
- **Old:** STUB/DEAD (body commented out).
- **New:** DEAD + EMERGENT (when revived).
- **Reframe:** `defection_cascade.rs:35-163` is gone. Its planned behaviour — one NPC defects, bonded allies follow — is not a system but a **cascade of `Defect(new_faction)` actions** triggered by social-bond utility spikes when a friend defects. The depth cap of 3 becomes a rule inside each NPC's decision model.
- **Required NPC actions:** `Defect(new_faction)`, `OfferDefectionBribe(target, amount)`.
- **Required derived views:** `defection_chain(seed_npc) = BFS over bond graph starting from seed until utility threshold fails`.
- **Required event types:** `NPCDefected {npc, old_faction, new_faction, trigger}`.
- **One-line:** Cascade is `Defect` actions propagating along bonds — no system.

#### alliance_blocs
- **Old:** ESSENTIAL (TransferGold, military buffs).
- **New:** EMERGENT.
- **Reframe:** `alliance_blocs.rs:31-120` auto-transfers gold between friendly factions and buffs shared military strength. Every transfer is the executed term of a standing `AllianceTreaty` — the leaders *chose* to form the alliance (`FormAlliance`) and *chose* its aid clauses. The periodic aid transfers are the scheduled execution of the treaty's terms, best modelled as an `AllianceAidTickDue` timer plus the accepting leader's `DisburseAllianceAid` action, or inlined into a deterministic consequence of the treaty event.
- **Required NPC actions (leader):** `FormAlliance(partners, terms)`, `BreakAlliance(target)`, `DisburseAllianceAid(target, amount)`, `CallAllianceToWar(enemy)`.
- **Required derived views:** `is_ally(a, b) = exists(AllianceFormed without later AllianceBroken)`, `bloc(f) = transitive closure of current alliances`.
- **Required event types:** `AllianceFormed {factions, terms}`, `AllianceBroken {former_partners, breaker}`, `AllianceAidSent {from, to, amount, reason}`.
- **One-line:** Alliances are signed treaties with scheduled aid events, not a passive transfer loop.

#### vassalage
- **Old:** ESSENTIAL (tribute transfers, rebellion surges).
- **New:** EMERGENT.
- **Reframe:** `vassalage.rs:53-290` handles auto-vassalage (weak factions drift toward strong ones), tribute, and rebellion. Each of these is an action: `SwearVassal(lord)` on the petitioner's side, `AcceptVassal(petitioner)` on the suzerain's, `RemitTribute(amount)` by the vassal each cycle, `Rebel()` when resentment boils over. The "drift" in relationship during auto-vassalage is post-hoc narrative for what was really a sequence of diplomatic overtures and pledges.
- **Required NPC actions (leader):** `SwearVassal(lord)`, `AcceptVassal(petitioner)`, `DemandTribute(vassal, amount)`, `RemitTribute(lord, amount)`, `Rebel(lord)`, `ReleaseVassal(vassal)`.
- **Required derived views:** `vassal_of(f) = chain of latest VassalageOath − ReleaseVassal events`.
- **Required event types:** `VassalageOathSworn {vassal, lord, terms}`, `TributeRemitted {vassal, lord, amount}`, `VassalRebelled {rebel, former_lord}`.
- **One-line:** Vassalage is an oath + tribute action pair; rebellion is also an action.

#### faction_tech
- **Old:** STUB/DEAD (body commented out).
- **New:** DERIVABLE-VIEW.
- **Reframe:** `faction_tech.rs:32-189` is all commented out. Rather than store tech levels, compute them: `faction_tech_level(f, axis) = sum of TechInvestment events for that axis by that faction`. The "investment" is the leader action `InvestInTech(axis, amount)`. Milestone bonuses become derived multipliers any consumer system references (e.g., combat ability modifiers).
- **Required NPC actions (leader):** `InvestInTech(axis, amount)`, `PoachScholar(target_faction)`, `FoundAcademy(settlement)`.
- **Required derived views:** `faction_tech(f, axis) = Σ InvestInTech deltas with diminishing returns` ; `has_tech_milestone(f, axis, tier) = faction_tech(f, axis) ≥ tier_threshold`.
- **Required event types:** `TechInvested {faction, axis, amount}`, `TechMilestoneCrossed {faction, axis, tier}` (optional — can be derived).
- **One-line:** Tech level = `Σ InvestInTech`, no mutable tech field needed.

#### warfare
- **Old:** ESSENTIAL (SOT for inter-faction war state, declares/ends wars).
- **New:** EMERGENT.
- **Reframe:** `warfare.rs:25-164` auto-declares war when the sum of cross-faction NPC grudges exceeds 50. Grudges exist and can be tallied — but *the decision to declare war* is a leader NPC action. Leader's utility consults the grudge tally (derived view), the current treasury, strength differential, and chooses. Peace is analogously `SignPeace(enemy)` by a leader when exhaustion view fires.
- **Required NPC actions (leader):** `DeclareWar(opponent, casus_belli)`, `SignPeace(enemy, terms)`, `RatifyTreaty(draft)`, `DemandReparations(enemy, amount)`.
- **Required derived views:** `grievance_matrix(a, b) = Σ over npcs_of(a) Grudge beliefs toward npcs_of(b)`, `at_war(a, b) = exists(WarDeclared(a,b)) without later PeaceSigned(a,b)`.
- **Required event types:** `WarDeclared {aggressor, defender, casus_belli, tick}`, `PeaceSigned {former_combatants, terms, tick}`.
- **One-line:** Leader chooses; grievances are *inputs* to that choice, not the cause.

#### succession
- **Old:** ESSENTIAL (transfers leadership, may flip runner-up to Hostile).
- **New:** EMERGENT (+ ESSENTIAL bookkeeping for team flip, until team is derived).
- **Reframe:** `succession.rs:16-152` hard-promotes the top-level-sum NPC whenever a leader dies and optionally flips the runner-up to `Hostile`. Under strict: the leader death emits `LeaderDied`; each senior council member then chooses `VoteForSuccessor(candidate)`, and the runner-up chooses `AcceptSuccessor()` or `Rebel()`. The "top candidate auto-wins" is replaced by the *council-vote reducer*, identical to `council` but scoped to a successor question.
- **Required NPC actions (council / candidate):** `VoteForSuccessor(candidate)`, `DeclareCandidacy()`, `AcceptSuccessor(winner)`, `Rebel()` (reused from civil_war), `Abdicate()` (incumbent choice before death).
- **Required derived views:** `current_leader(s) = most-recent Succeeded event for s`.
- **Required event types:** `LeaderDied {settlement, predecessor}`, `SuccessorVoteCast {voter, candidate}`, `LeaderSucceeded {settlement, predecessor, successor, mechanism}`.
- **One-line:** Council votes choose the successor; Rebel is a citizen action.

#### legends
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `legends.rs:18-175` detects legendary NPCs from `(chronicle_mentions ≥ 5, classes ≥ 2, friend_deaths ≥ 3)`, renames them to `" the Legendary"`, boosts settlement morale, and triggers world-wide mourning on their death. Legend status is **100% derivable** — the name mutation is a cache. The settlement morale bonus is a view consulted by mood calculations (`settlement_legend_halo(s) = +0.06 if a legend lives there`). The world-mourning effect *is* a one-shot state change, but it is the consequence of the existing `EntityDied` event plus the derived `is_legendary` predicate — no system logic needed beyond "emit `LegendMourned` event" when an `EntityDied` lands on a legend.
- **Required NPC actions:** none.
- **Required derived views:** `is_legendary(npc) = mention_count(npc) ≥ 5 AND class_count(npc) ≥ 2 AND friend_deaths(npc) ≥ 3`; `legend_halo(s) = max_legend_presence_bonus for home_settlement==s`; `legendary_name(npc) = base_name + " the Legendary" if is_legendary else base_name`.
- **Required event types:** `LegendAscended {npc, criteria}` (optional, for narrative), `LegendMourned {npc, tick}` (derived from `EntityDied` + predicate).
- **One-line:** Legend is a predicate; the mass-grief is a consequence of `EntityDied` when the predicate holds.

#### prophecy
- **Old:** ESSENTIAL.
- **New:** EMERGENT (issuance) + DERIVABLE-VIEW (fulfillment).
- **Reframe:** `prophecy.rs:19-245` stores `state.prophecies[]` with a `fulfilled` bool. Issuance is a prophet NPC action `IssueProphecy(condition, effect)`. Fulfillment is a pure query: `fulfilled(p) = exists(event in world_events post p.tick matching p.condition)`. The dramatic effects (monster surge, morale boost, stockpile gift) are one-shot state transitions — they are the *consequence* of the fulfillment event `ProphecyFulfilled`, which can be emitted by a thin reducer rather than a 245-line system.
- **Required NPC actions (prophet):** `IssueProphecy(condition, effect)`, `Recant(prophecy_id)`, `InterpretProphecy(target_prophecy, reading)`.
- **Required derived views:** `prophecy_fulfilled(p) = exists_event_matching(p.condition, tick > p.issued_tick)`, `active_prophecies = prophecies where !fulfilled`.
- **Required event types:** `ProphecyIssued {prophet, condition, effect, tick}`, `ProphecyFulfilled {prophecy_id, tick, trigger_event}`.
- **One-line:** Prophecy is issue-once + derivable-predicate; fulfillment consequences fire from a generic reducer.

#### outlaws
- **Old:** ESSENTIAL (flips team to Friendly on redemption; raids transfer gold).
- **New:** EMERGENT.
- **Reframe:** `outlaws.rs:15-179` handles three phases: raid, camp detection, redemption. Every phase is an NPC action: `Raid(target_npc)` by the outlaw, `FormBandit Camp(cluster_leader)` by a self-appointed camp leader, `SeekRedemption(settlement, amount)` by the outlaw himself. The team flip to Friendly is the resolution of `RedemptionAccepted` by the target settlement's leader. "Becoming an outlaw" is a first-class action `BecomeOutlaw()` chosen by disaffected NPCs or forced by `Exile(target)` from a settlement leader.
- **Required NPC actions (commoner):** `BecomeOutlaw()`, `Raid(target_npc)`, `FormBanditCamp(members)`, `SeekRedemption(settlement, gold_offer)`, `JoinBanditCamp(camp_id)`.
- **Required NPC actions (leader):** `Exile(target_npc)`, `AcceptRedemption(outlaw, payment)`, `PostBounty(outlaw, reward)`.
- **Required derived views:** `is_outlaw(npc) = latest of {BecameOutlaw, RedemptionAccepted} is BecameOutlaw`, `bandit_camps = cluster(outlaw_positions, radius)`.
- **Required event types:** `BecameOutlaw {npc, trigger}`, `RaidSucceeded {outlaw, victim, gold}`, `RedemptionAccepted {outlaw, settlement, payment}`.
- **One-line:** Outlaw status is event-derived; every state change is an action.

#### settlement_founding
- **Old:** ESSENTIAL (spawns new settlement entity + colonist reassignment).
- **New:** ESSENTIAL *(spawn bookkeeping only)* + EMERGENT *(the decision)*.
- **Reframe:** `settlement_founding.rs:20-178` checks overcrowding and auto-launches 8 colonists. The decision — *should we leave?* — is a **commoner + aspiring-leader action**: a charismatic NPC chooses `LeadFoundingExpedition(target_region)`, and fellow residents choose `JoinExpedition(leader)`. The actual `state.settlements.push(new_s)` + `entity_id = next_id()` allocation at the moment of arrival is the irreducible bookkeeping step. The colonist reassignment (`home_settlement_id = new_s.id`) is the consequence of `ExpeditionArrived`.
- **Required NPC actions (leader-candidate):** `LeadFoundingExpedition(target_region)`, `ReturnFromFailedExpedition()`.
- **Required NPC actions (commoner):** `JoinExpedition(leader)`, `RefuseExpedition(leader)`.
- **Required derived views:** `is_overcrowded(s) = pop(s) / housing(s) > 1.5`, `viable_target_regions = settleable regions > min_dist from existing settlements`.
- **Required event types:** `ExpeditionLaunched {leader, members, target}`, `ExpeditionArrived {members, new_settlement_id}` (spawn point), `ExpeditionFailed {leader, cause}`.
- **One-line:** The decision is NPC choice; only the entity-ID allocation on arrival is essential.

#### betrayal
- **Old:** ESSENTIAL (flips team, steals treasury, seeds grudges).
- **New:** EMERGENT.
- **Reframe:** `betrayal.rs:26-137` auto-selects treacherous NPCs, steals treasury, flips team, seeds grudges. The entire thing is **one NPC action**: `Betray(faction)`. Its resolution is the consequence event `BetrayalCommitted`, which: (1) optionally transfers gold on the same tick if the betrayer also chose `StealTreasury(settlement)`, (2) moves the betrayer's `faction_id` (or emits `FactionMembershipChanged`), (3) provokes residents' `FormGrudge(betrayer)` reactions on subsequent ticks. The 50-grudge mass insert at end of `advance_betrayal` collapses into each resident NPC's own grudge-formation utility responding to the event.
- **Required NPC actions (commoner):** `Betray(faction)`, `StealTreasury(settlement, amount)`, `FleeAfterBetrayal(destination)`, `FormGrudge(target)` (reactive to BetrayalCommitted).
- **Required derived views:** `treachery_score(npc) = STEALTH_tag + DECEPTION_tag − compassion*penalty` used as utility input.
- **Required event types:** `BetrayalCommitted {traitor, victim_faction, theft_amount}`, `GrudgeFormed {holder, target, cause}`.
- **One-line:** Betrayal is one action; grudges are reactive actions.

#### family
- **Old:** ESSENTIAL (marriage, birth entity creation).
- **New:** ESSENTIAL *(birth entity alloc)* + EMERGENT *(the decision to marry / have children)*.
- **Reframe:** `family.rs:26-252` auto-matches compatible NPCs at 15% and auto-creates children after 2000 ticks married at 5%. Under strict: two NPCs each choose `Marry(target)` (bi-directional consent), spouse_id update is the consequence. Child creation is choose `HaveChild(spouse)` — the entity `push` + new NPC ID allocation + blended profile is the irreducible bookkeeping at resolution. Marriage compatibility formula becomes the *utility input* both sides' decisions consult.
- **Required NPC actions (commoner):** `Marry(target)`, `Divorce(spouse)`, `HaveChild(spouse)`, `AdoptChild(candidate)`, `LeaveSpouse(destination)`.
- **Required derived views:** `marriage_compatibility(a, b) = 1 − |social_drive_a − social_drive_b| − 0.5*|compassion_a − compassion_b|`, `child_cap_reached(couple) = |children| ≥ 3`, `settlement_at_cap(s) = pop(s) ≥ 300`.
- **Required event types:** `MarriageFormed {spouses, settlement}`, `MarriageEnded {former_spouses, cause}`, `ChildBorn {parents, child_id, settlement}`.
- **One-line:** Decisions are actions; only `ChildBorn` entity-ID allocation + profile blend is essential bookkeeping.

#### haunted
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `haunted.rs:22-138` clusters `EntityDied` positions, then pushes fear/anxiety to nearby NPCs and flips a `LocationDangerous` belief. The cluster is a derived query. The belief-push on exposed NPCs is better modelled as each NPC's perception loop: when a commoner NPC walks within range of a haunted site (derived from recent deaths), they choose (or auto-react) `FormBelief(LocationDangerous)`. No top-down system required; `is_haunted(pos)` is a pure view.
- **Required NPC actions:** `FormBelief(LocationDangerous(site_id))` (perception reaction).
- **Required derived views:** `is_haunted(pos, tick) = cluster(death_positions in world_events where tick − t < window).any(|c| |c − pos|² < 900 AND c.count ≥ 5)`, `haunted_sites(tick) = {centres of qualifying clusters}`.
- **Required event types:** none new — reads `EntityDied`. Optional `HauntedSiteRecognized {pos, first_tick}` for chronicle.
- **One-line:** `is_haunted(pos)` is a window query over `EntityDied`.

#### world_ages
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `world_ages.rs:22-93` scans a 2400-tick chronicle window and names the era. Pure chronicle classifier with zero gameplay effect. Replace with an on-demand view `current_world_age(tick) = classify(chronicle_window_stats(tick − 2400, tick))` called by UI / save-game / biography exports.
- **Required NPC actions:** none.
- **Required derived views:** `current_world_age(tick)`, `age_history = running label over chronicle windows`.
- **Required event types:** none.
- **One-line:** Pure classifier view over chronicle windows.

#### chronicle
- **Old:** EMERGENT-CANDIDATE (note: "this system IS the narrative log emitter, but only for treasury/pop milestones").
- **New:** ESSENTIAL *(chronicle.push is the event recording primitive)* + DERIVABLE-VIEW *(milestone-detect sub-function)*.
- **Reframe:** The **act of writing chronicle entries** is the only truly irreducible part of the entire batch — it is literally how events are recorded. The existing `chronicle.rs:16-96` does only milestone detection (treasury/pop thresholds) and is a mis-named derived-view system; rename that part to `chronicle_milestones` and treat it as a DERIVABLE-VIEW emitter over `state.settlements`. The `chronicle.push(...)` primitive — invoked by *every* action resolver to record what happened — is the essential kernel of the event-sourced architecture.
- **Required NPC actions:** none (chronicle is not an action — every action *emits* a chronicle entry via the resolver).
- **Required derived views:** `treasury_milestones(s) = crossings of TREASURY_THRESHOLDS by s.treasury history`, `population_milestones(s) = crossings of POPULATION_THRESHOLDS by s.population history`.
- **Required event types:** `ChronicleEntry` is the universal event record; specific categories remain (Economy, Narrative, Combat, etc.).
- **One-line:** `chronicle.push` is the kernel primitive; milestone detection is a derived view on top of settlement history.

#### crisis
- **Old:** ESSENTIAL (owns settlement stockpile drain, status effects).
- **New:** DERIVABLE-VIEW + EMERGENT.
- **Reframe:** `crisis.rs:34-174` picks one of 4 crisis types per high-threat region and mutates state. The *pick* is hashed on region+tick — it is not modelling a decision, it is a scripted random effect. Reframe: a region "in crisis" is a **view** (`is_in_crisis(r) = r.threat_level > 70`); the *response* by in-region NPCs is a set of actions (`FleeCrisisRegion`, `ReinforceCrisisRegion`, `RationSupplies`, `RiseAsUnifier`). The damage and commodity drain are framed as deterministic environmental hazards derived from the region's threat level — not a separate system, but a per-tick hazard consultation by NPCs and buildings.
- **Required NPC actions:** `FleeCrisisRegion(destination)`, `ReinforceCrisisRegion(region)` (leader or hero), `RationSupplies(settlement)` (leader), `RiseAsUnifier()` (charismatic NPC).
- **Required derived views:** `is_in_crisis(r) = threat_level(r) > 70`, `crisis_hazard(r, npc_pos) = function of proximity, threat, region_type`.
- **Required event types:** `CrisisBegan {region, type, tick}` (when view tips), `UnifierRose {npc, region}`, `RefugeesFled {region, destination, count}`.
- **One-line:** Crisis is a view; the responses are NPC actions.

#### difficulty_scaling
- **Old:** ESSENTIAL.
- **New:** DERIVABLE-VIEW (meta).
- **Reframe:** `difficulty_scaling.rs:27-165` computes a global power rating and rubber-bands with damage or heals. This is a meta-game controller, not diegetic. Either delete (we don't want rubber-banding in a zero-player world-sim), or keep as a pure view that UI/training loops consult: `player_power_rating = score(...)`. The damage/heal side-effects do not belong in the simulation.
- **Required NPC actions:** none.
- **Required derived views:** `power_rating(state) = f(friendly_count, avg_level, treasury, territory, pop, monster_density)`.
- **Required event types:** none.
- **One-line:** Delete the write-path; keep the scalar as a view for training/UI.

#### charter
- **Old:** DUPLICATIVE.
- **New:** DEAD.
- **Reframe:** `charter.rs:14-46` is a treasury-feedback loop that v1 already flagged as a proxy (comments `:27-29`). The real-world thing it gestures at — a settlement *adopting a charter* — is an EMERGENT leader action `AdoptCharter(template_id)` that should exist in the leader action space. Delete the current system; add the action.
- **Required NPC actions (leader):** `AdoptCharter(template)`, `AmendCharter(clauses)`, `RevokeCharter()`.
- **Required derived views:** `charter(s) = latest AdoptCharter/RevokeCharter for s`, `charter_bonuses(s) = lookup of active charter`.
- **Required event types:** `CharterAdopted {settlement, template, tick}`.
- **One-line:** Delete the noise; the action-space covers it.

#### choices
- **Old:** DUPLICATIVE.
- **New:** DEAD.
- **Reframe:** `choices.rs:15-39` emits random ±gold jitter per settlement. Source comments (`:22-24`) admit it is a placeholder. Under strict event-sourcing, every "choice outcome" is a resolved NPC/leader action — there is no place for system-level random jitter.
- **Required NPC actions:** none new (the whole concept dissolves into the general leader/commoner action space).
- **Required derived views:** none.
- **Required event types:** none.
- **One-line:** Delete.

#### rival_guild
- **Old:** DUPLICATIVE.
- **New:** DEAD.
- **Reframe:** `rival_guild.rs:15-69` auto-fires random sabotage/raids. Source comments (`:35-36`) confirm it is a stub. The rival guild — if it exists — is another faction whose leader chose `Sabotage(target)` or `Raid(target)`. Delete and rely on the action space; if specific rival-guild flavour is wanted, seed a rival faction at worldgen.
- **Required NPC actions:** (covered by `Sabotage`, `Raid` in generic action space.)
- **Required derived views:** none.
- **Required event types:** none new.
- **One-line:** Delete; seed a rival faction at worldgen instead.

#### victory_conditions
- **Old:** DUPLICATIVE.
- **New:** DEAD *(current impl)* + DERIVABLE-VIEW *(as designed)*.
- **Reframe:** `victory_conditions.rs:16-55` damages NPCs in high-threat regions after tick 15000 — source (`:26-28`) calls it an approximation. A true victory check is a pure view: `victory_state(state) = classify({conquest, economic, narrative, survival})`. No write path needed.
- **Required NPC actions:** none.
- **Required derived views:** `victory_state(state, faction_or_player)` covering conquest, economic, cultural, survival axes.
- **Required event types:** `VictoryAchieved {faction, type, tick}` emitted once by the view's tip-crossing.
- **One-line:** Delete write path; compute victory as a view.

#### awakening
- **Old:** EMERGENT-CANDIDATE.
- **New:** EMERGENT (if kept) *or* DEAD (recommended, per v1 passive-buff-clone finding).
- **Reframe:** `awakening.rs:14-96` rolls 1% chance for level-8+ NPCs to become "awakened" with a permanent 1.25× buff. An NPC choosing to "awaken" is coherent — `SeekAwakening()` at a sacred site — but the current impl has no such narrative. If kept: make it an NPC action. If we are being strict: this is one of four near-identical passive-buff systems and should be deleted.
- **Required NPC actions (commoner):** `SeekAwakening(sacred_site)`, `UndergoTrial(type)`.
- **Required derived views:** `is_awakened(npc) = exists AwakeningGranted event for npc`.
- **Required event types:** `AwakeningGranted {npc, trigger}`.
- **One-line:** Make it an NPC action with a location/trial, or delete.

#### visions
- **Old:** DUPLICATIVE.
- **New:** DEAD.
- **Reframe:** `visions.rs:15-78` applies a 1.05× "prophetic" buff; source comments (`:43-45`) admit it is a proxy. Identical shape to awakening/bloodlines/legacy_weapons. If the concept must survive, it is an NPC action `ReceiveVision()` emitted by a prophet — but redundant with `IssueProphecy`. Delete.
- **Required NPC actions:** none.
- **Required derived views:** none.
- **Required event types:** none.
- **One-line:** Delete.

#### bloodlines
- **Old:** DUPLICATIVE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `bloodlines.rs:15-79` applies a 1.10× "bloodline" buff to level-8+ NPCs. Source comments (`:46-48`) admit no actual bloodline tracking. A real bloodline is the **HasChild graph** — purely derivable. The buff model should read from that graph: `bloodline_bonus(npc) = f(ancestral_legend_count_in_lineage(npc))`.
- **Required NPC actions:** none (parent/child are covered by `family`).
- **Required derived views:** `lineage(npc) = chase HasChild(parent, npc) upward via ChildBorn events`, `descendants(npc) = chase downward`, `lineage_prestige(npc) = Σ is_legendary(ancestor) for ancestor in lineage(npc)`.
- **Required event types:** none new — `ChildBorn` already recorded.
- **One-line:** Lineage is a chain query over `ChildBorn`; buff is a view on that chain.

#### legacy_weapons
- **Old:** DUPLICATIVE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `legacy_weapons.rs:15-80` applies a 1.10× "legacy_weapon" buff. There is no weapon entity. Real legacy weapons = items with kill-count history = derivable from an item's `WeaponKill` event history. Bonus is a view on that.
- **Required NPC actions:** `WieldLegacyWeapon(item)`, `PassDownWeapon(heir, item)` (if item entities become part of the world).
- **Required derived views:** `legacy_weapons = items where kill_count(item) ≥ N`, `legacy_wielder_bonus(npc) = bonus if wielded item ∈ legacy_weapons`.
- **Required event types:** `WeaponKill {weapon_id, wielder, victim}`, `WeaponPassedDown {weapon_id, from, to}`.
- **One-line:** Legacy status = derived from kill-count events; no system write needed.

#### artifacts
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `artifacts.rs:18-112` names an "artifact" in the chronicle for high-level multi-class NPCs — **no item entity is created** (comment notes this). Pure chronicle emission. Can be computed on demand: `artifact_of(npc) = derived name if level ≥ 35 AND classes ≥ 4 AND roll < 0.03`. If we want actual artifact items, introduce a `ForgeArtifact` action (leader or grandmaster NPC) and track the item — but the current v1 behaviour is just a view.
- **Required NPC actions (grandmaster):** `ForgeArtifact(material, dedication)` (optional future).
- **Required derived views:** `eligible_artifact_bearers = npcs where level ≥ 35 AND class_count ≥ 4`, `artifact_name(npc) = deterministic hash of (npc_id, tick_at_eligibility)`.
- **Required event types:** `ArtifactForged {smith, item, settlement}` (if item entities exist).
- **One-line:** Current system is a pure derivable view; real artifacts would need a `ForgeArtifact` action.

#### great_works
- **Old:** ESSENTIAL (deducts treasury, commissions named monuments).
- **New:** EMERGENT.
- **Reframe:** `great_works.rs:12-89` auto-commissions monuments on wealthy, populous settlements, deducting treasury. Under strict: commissioning a great work is a **settlement-leader action** `CommissionGreatWork(type, cost)`, and `ContributeToGreatWork(amount)` is a commoner action for volunteer labour/donations. Completion is the reducer emitting `GreatWorkCompleted` when total contributions ≥ cost. Treasury deduct = consequence of the commission action.
- **Required NPC actions (leader):** `CommissionGreatWork(type, budget)`, `CancelGreatWork(project)`.
- **Required NPC actions (commoner):** `ContributeToGreatWork(project, amount)`, `VolunteerLabour(project, hours)`.
- **Required derived views:** `great_works(s) = ongoing + completed projects at s`, `total_works_completed(s) = count(GreatWorkCompleted where settlement==s)`.
- **Required event types:** `GreatWorkCommissioned {settlement, type, budget}`, `GreatWorkContribution {contributor, project, amount}`, `GreatWorkCompleted {settlement, project, type, tick}`.
- **One-line:** Leader commissions, commoners contribute, completion fires from the reducer.

#### culture
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `culture.rs:18-142` reduces per-NPC behaviour tags into four culture axes (martial/mercantile/scholarly/spiritual) and applies settlement-level buffs and morale writes. This is a textbook reduction-over-residents view: `culture_mix(s) = normalized_sum(behaviour_value[axis_tags] over home_settlement_id==s)`. The "effects" (threat_level−, infra+, morale+) are views any consumer can apply — or, if we must cache, they are idempotent derived writes refreshed from the view each tick.
- **Required NPC actions:** none.
- **Required derived views:** `culture(s) = {martial_pct, mercantile_pct, scholarly_pct, spiritual_pct}`, `dominant_culture(s) = argmax of culture(s)`, `culture_modifier_threat(s) = −(martial_pct − 30)*0.01 if martial_pct > 30`, `culture_morale_bonus(s, npc) = +(spiritual_pct − 30)*0.01 if spiritual_pct > 30`.
- **Required event types:** `CultureShifted {settlement, new_dominant, tick}` (optional narrative emission when dominant crosses over).
- **One-line:** Culture is a reduction-over-residents view; every "effect" is that view consulted by another system.

---

### Reduction summary

| System | v1 | v2 | Core primitive |
|---|---|---|---|
| faction_ai | ESSENTIAL | ESSENTIAL *(decision head)* + EMERGENT *(branches)* | Leader action vocabulary |
| diplomacy | ESSENTIAL | EMERGENT | `ProposeTradeAccord` / derived relation |
| espionage | ESSENTIAL | EMERGENT | `Spy(target)` action |
| counter_espionage | ESSENTIAL | EMERGENT + DERIVABLE-VIEW | `ArrestSpy` action + detection view |
| war_exhaustion | STUB/DEAD | DEAD + DERIVABLE-VIEW *(formula)* | `exhaustion(f)` view |
| civil_war | ESSENTIAL | EMERGENT | `DeclareCivilWar`, `JoinFactionSide` |
| council | EMERGENT-CAND. | EMERGENT | `VoteOnIssue` + tally view |
| coup_engine | ESSENTIAL | EMERGENT | `LaunchCoup` + utility view |
| defection_cascade | STUB/DEAD | DEAD + EMERGENT *(when revived)* | `Defect(new_faction)` |
| alliance_blocs | ESSENTIAL | EMERGENT | `FormAlliance`, `DisburseAllianceAid` |
| vassalage | ESSENTIAL | EMERGENT | `SwearVassal`, `RemitTribute`, `Rebel` |
| faction_tech | STUB/DEAD | DERIVABLE-VIEW | `Σ InvestInTech` |
| warfare | ESSENTIAL | EMERGENT | `DeclareWar`, `SignPeace` + grievance view |
| succession | ESSENTIAL | EMERGENT | `VoteForSuccessor`, `Rebel` |
| legends | EMERGENT-CAND. | DERIVABLE-VIEW | `is_legendary(npc)` |
| prophecy | ESSENTIAL | EMERGENT *(issuance)* + DERIVABLE-VIEW *(fulfillment)* | `IssueProphecy` + fulfillment query |
| outlaws | ESSENTIAL | EMERGENT | `BecomeOutlaw`, `Raid`, `SeekRedemption` |
| settlement_founding | ESSENTIAL | ESSENTIAL *(spawn alloc)* + EMERGENT *(expedition)* | `ExpeditionArrived` bookkeeping |
| betrayal | ESSENTIAL | EMERGENT | `Betray(faction)` |
| family | ESSENTIAL | ESSENTIAL *(child alloc)* + EMERGENT *(decisions)* | `ChildBorn` bookkeeping |
| haunted | EMERGENT-CAND. | DERIVABLE-VIEW | `is_haunted(pos)` |
| world_ages | EMERGENT-CAND. | DERIVABLE-VIEW | `current_world_age(tick)` |
| chronicle | EMERGENT-CAND. | ESSENTIAL *(push primitive)* + DERIVABLE-VIEW *(milestones)* | `chronicle.push` |
| crisis | ESSENTIAL | DERIVABLE-VIEW + EMERGENT | `is_in_crisis(r)` + response actions |
| difficulty_scaling | ESSENTIAL | DERIVABLE-VIEW | `power_rating(state)` view only |
| charter | DUPLICATIVE | DEAD | `AdoptCharter` action |
| choices | DUPLICATIVE | DEAD | — |
| rival_guild | DUPLICATIVE | DEAD | covered by generic `Sabotage`/`Raid` |
| victory_conditions | DUPLICATIVE | DEAD + DERIVABLE-VIEW | `victory_state(state)` |
| awakening | EMERGENT-CAND. | EMERGENT *(if kept)* / DEAD | `SeekAwakening` |
| visions | DUPLICATIVE | DEAD | — |
| bloodlines | DUPLICATIVE | DERIVABLE-VIEW | `lineage(npc)` |
| legacy_weapons | DUPLICATIVE | DERIVABLE-VIEW | `kill_count(item)` |
| artifacts | EMERGENT-CAND. | DERIVABLE-VIEW | `artifact_of(npc)` |
| great_works | ESSENTIAL | EMERGENT | `CommissionGreatWork`, `ContributeToGreatWork` |
| culture | EMERGENT-CAND. | DERIVABLE-VIEW | `culture(s)` reduction |

#### Counts

| Class | v1 | v2 |
|---|---:|---:|
| ESSENTIAL | ~14-18 | **4** (chronicle.push, settlement_founding.spawn, family.birth alloc, faction_ai *as decision head*) |
| EMERGENT | 0 (not a class) | **14** (diplomacy, espionage, counter_espionage, civil_war, council, coup_engine, alliance_blocs, vassalage, warfare, succession, prophecy issuance, outlaws, betrayal, great_works, + the action-branches of faction_ai / settlement_founding / family) |
| DERIVABLE-VIEW | ~7 | **11** (faction_tech, legends, haunted, world_ages, culture, crisis, difficulty_scaling, bloodlines, legacy_weapons, artifacts, war_exhaustion formula, victory_conditions, prophecy fulfillment) |
| DEAD | ~3 | **7** (war_exhaustion, defection_cascade, faction_tech *write*, charter, choices, rival_guild, victory_conditions *write*, visions, awakening) |

(Totals sum to >36 because several systems split across buckets: `chronicle` is both ESSENTIAL-kernel and DERIVABLE-VIEW-milestones; `family` is ESSENTIAL-birth + EMERGENT-decisions; `prophecy` is EMERGENT-issuance + DERIVABLE-VIEW-fulfillment; `settlement_founding` is EMERGENT-decision + ESSENTIAL-spawn.)

---

### Required action vocabulary (split by NPC role)

#### Common NPC actions
- `Marry(target)`, `Divorce(spouse)`, `HaveChild(spouse)`, `AdoptChild(candidate)`, `LeaveSpouse(destination)`
- `Betray(faction)`, `StealTreasury(settlement, amount)`, `FleeAfterBetrayal(destination)`
- `BecomeOutlaw()`, `Raid(target_npc)`, `FormBanditCamp(members)`, `JoinBanditCamp(camp_id)`, `SeekRedemption(settlement, offer)`
- `Spy(target_faction)`, `Sabotage(target_settlement)`, `InfiltrateCouncil(target_faction)`
- `JoinFactionSide(side)` (civil war), `FleeCivilWar(destination)`
- `Defect(new_faction)`, `OfferDefectionBribe(target, amount)`
- `FleeCrisisRegion(destination)`, `ReinforceCrisisRegion(region)`, `RiseAsUnifier()`
- `JoinExpedition(leader)`, `RefuseExpedition(leader)`
- `VoteOnIssue(issue_id, choice)`, `AbstainVote(issue_id)`, `VoteForSuccessor(candidate)`, `DeclareCandidacy()`, `AcceptSuccessor(winner)`, `Rebel()`
- `ContributeToGreatWork(project, amount)`, `VolunteerLabour(project, hours)`
- `FormGrudge(target)`, `FormBelief(LocationDangerous(site))`, `PledgeLoyalty(leader)`
- `SeekAwakening(sacred_site)`, `UndergoTrial(type)` *(if awakening survives)*
- `WieldLegacyWeapon(item)`, `PassDownWeapon(heir, item)` *(if item entities)*

#### Leader NPC actions (faction / settlement level)
- `DeclareWar(opponent, casus_belli)`, `SignPeace(enemy, terms)`, `RatifyTreaty(draft)`, `DemandReparations(enemy, amount)`
- `ProposeTradeAccord(target)`, `OpenDiplomaticChannel(target)`, `OfferGift(target, amount)`, `BreakRelations(target)`
- `FormAlliance(partners, terms)`, `BreakAlliance(target)`, `DisburseAllianceAid(target, amount)`, `CallAllianceToWar(enemy)`
- `SwearVassal(lord)`, `AcceptVassal(petitioner)`, `DemandTribute(vassal, amount)`, `ReleaseVassal(vassal)`
- `LaunchCoup(target_leader)`, `BribeGarrison(amount)`, `FleeAfterFailedCoup(destination)`
- `DeclareCivilWar(target_leader)`, `SurrenderCivilWar()`
- `LaunchConquest(target_settlement)`, `ReclaimSettlement(target)`, `RecruitMilitia()`, `RaiseTaxes(rate)`
- `LeadFoundingExpedition(target_region)`, `ReturnFromFailedExpedition()`
- `CommissionGreatWork(type, budget)`, `CancelGreatWork(project)`
- `AdoptCharter(template)`, `AmendCharter(clauses)`, `RevokeCharter()`
- `IssueProphecy(condition, effect)`, `Recant(prophecy_id)`, `InterpretProphecy(id, reading)`
- `Exile(target_npc)`, `AcceptRedemption(outlaw, payment)`, `PostBounty(outlaw, reward)`
- `SetStandingOrder(kind, level)`, `ArrestSpy(agent)`, `ExecuteSpy(agent)`, `ExileSpy(agent)`
- `InvestInTech(axis, amount)`, `PoachScholar(target_faction)`, `FoundAcademy(settlement)`
- `Abdicate()`, `CallCouncil()`, `TableMotion(text)`
- `RationSupplies(settlement)` (crisis response)
- `CharterGuild(name, purpose)`, `OutlawCitizen(target)`

---

### Required event types

Event names follow `PastTenseAction {actor, ..., tick}`. All actions resolve to one or more events; every event lives in the chronicle or `world_events` stream.

**War / conquest**
- `WarDeclared {aggressor, defender, casus_belli, tick}`
- `PeaceSigned {former_combatants, terms, tick}`
- `SettlementConquered {aggressor, defender, settlement, tick}`
- `SettlementReclaimed {faction, settlement, tick}`

**Diplomacy**
- `TradeAccordSigned {a, b, terms, tick}`
- `DiplomaticGiftSent {from, to, amount, tick}`
- `RelationsBroken {a, b, cause, tick}`
- `AllianceFormed {factions, terms, tick}`
- `AllianceBroken {former_partners, breaker, tick}`
- `AllianceAidSent {from, to, amount, reason, tick}`

**Vassalage**
- `VassalageOathSworn {vassal, lord, terms, tick}`
- `TributeRemitted {vassal, lord, amount, tick}`
- `VassalRebelled {rebel, former_lord, tick}`
- `VassalReleased {vassal, lord, tick}`

**Succession / coups / civil wars**
- `LeaderDied {settlement_or_faction, predecessor, tick}`
- `SuccessorVoteCast {voter, candidate, issue, tick}`
- `LeaderSucceeded {scope, predecessor, successor, mechanism, tick}`
- `Abdicated {leader, scope, tick}`
- `CoupAttempted {instigator, target_leader, success, tick}`
- `CoupSuppressed {instigator, defender, tick}`
- `RegimeChanged {faction, new_leader, mechanism, tick}`
- `CivilWarDeclared {faction, instigator, grievance, tick}`
- `CivilWarSideJoined {citizen, side, tick}`
- `CivilWarResolved {victor, mechanism, tick}`

**Espionage**
- `SpyMissionStarted {agent, target_faction, tick}`
- `SpyMissionSucceeded {agent, target, impact, tick}`
- `SpyCaught {agent, defender_settlement, tick}`
- `SpyArrested {agent, settlement, tick}`
- `SpyKilled {agent, settlement, tick}`

**Outlaws / betrayal**
- `BecameOutlaw {npc, trigger, tick}`
- `RaidSucceeded {outlaw, victim, gold, tick}`
- `BanditCampFormed {members, location, tick}`
- `RedemptionAccepted {outlaw, settlement, payment, tick}`
- `BetrayalCommitted {traitor, victim_faction, theft_amount, tick}`
- `GrudgeFormed {holder, target, cause, tick}`
- `NPCDefected {npc, old_faction, new_faction, trigger, tick}`

**Family / lineage**
- `MarriageFormed {spouses, settlement, tick}`
- `MarriageEnded {former_spouses, cause, tick}`
- `ChildBorn {parents, child_id, settlement, tick}`
- `ChildAdopted {parents, child_id, tick}`

**Settlement / founding / charter / works**
- `ExpeditionLaunched {leader, members, target, tick}`
- `ExpeditionArrived {members, new_settlement_id, tick}`
- `ExpeditionFailed {leader, cause, tick}`
- `SettlementFounded {founders, loc, parent_settlement, tick}` (alias / wrapper of ExpeditionArrived)
- `CharterAdopted {settlement, template, tick}`
- `GreatWorkCommissioned {settlement, type, budget, tick}`
- `GreatWorkContribution {contributor, project, amount, tick}`
- `GreatWorkCompleted {settlement, project, type, tick}`

**Crisis / prophecy / legends / culture**
- `ProphecyIssued {prophet, condition, effect, tick}`
- `ProphecyFulfilled {prophecy_id, tick, trigger_event}`
- `CrisisBegan {region, type, tick}`
- `UnifierRose {npc, region, tick}`
- `RefugeesFled {region, destination, count, tick}`
- `LegendAscended {npc, criteria, tick}` *(optional narrative)*
- `LegendMourned {npc, tick}` *(optional narrative, derivable)*
- `CultureShifted {settlement, new_dominant, tick}` *(optional narrative)*

**Tech / items**
- `TechInvested {faction, axis, amount, tick}`
- `TechMilestoneCrossed {faction, axis, tier, tick}` *(optional)*
- `WeaponKill {weapon_id, wielder, victim, tick}`
- `WeaponPassedDown {weapon_id, from, to, tick}`
- `ArtifactForged {smith, item, settlement, tick}` *(optional future)*

**Council / votes**
- `CouncilVoteCast {voter, issue, choice, tick}`
- `CouncilMotionResolved {issue, outcome, tick}`

**Governance**
- `TaxesRaised {faction, rate, tick}`
- `CitizenExiled {exiler, target, reason, tick}`
- `BountyPosted {settlement, target_outlaw, reward, tick}`
- `StandingOrderSet {leader, kind, level, tick}`

---

### Required derived views

All of the following are pure functions of primary state + event streams. None of them require a mutable system to maintain.

**Relations & alliances**
- `is_at_war(a, b, tick) = exists(WarDeclared(a,b,t1)) with no later PeaceSigned(a,b)`
- `bloc(f, tick) = transitive closure of { g | is_ally(f,g,tick) }`
- `is_ally(a, b, tick) = exists(AllianceFormed) with no later AllianceBroken`
- `relation(a, b, tick) = Σ decayed contributions from {TradeAccord, AllianceFormed, WarDeclared, DiplomaticGiftSent, Betrayal, GrudgeFormed}`
- `grievance_matrix(a, b) = Σ over npcs_of(a) Grudge beliefs against npcs_of(b)`
- `vassal_of(f, tick) = lord chain from latest VassalageOath minus later VassalReleased`

**Faction health**
- `military_strength(f, tick) = base + Σ MilitaryRecruited − Σ casualty events` *(or kept as a primary field updated by reducers)*
- `war_exhaustion(f) = duration_at_war(f) × casualty_rate × treasury_drain_rate`
- `coup_conditions(f) = {unrest_avg, treasury_ratio, escalation, leader_legitimacy}` *(utility inputs)*
- `faction_tech(f, axis) = Σ TechInvested(f,axis) with diminishing returns`
- `has_tech_milestone(f, axis, tier) = faction_tech(f,axis) ≥ tier_threshold`
- `civil_war_status(f) = { phase, rebel_support, loyalist_support, duration }`

**Settlements**
- `pop(s, tick) = count(npcs where home_settlement_id==s AND alive)`
- `housing(s) = Σ completed residential buildings capacity`
- `is_overcrowded(s) = pop(s) / housing(s) > 1.5`
- `culture(s) = normalized reduction over resident NPC behaviour tags on {martial, mercantile, scholarly, spiritual}`
- `dominant_culture(s) = argmax of culture(s)`
- `culture_modifier_threat(s) = −(martial_pct − 30)*0.01 if martial_pct > 30 else 0`
- `culture_morale_bonus(s, npc) = +(spiritual_pct − 30)*0.01 if spiritual_pct > 30 else 0`
- `council_composition(s) = histogram(npc.faction_id for home_settlement_id==s)`
- `council_outcome(s, issue) = argmax(votes_by_choice) over CouncilVoteCast`
- `charter(s) = latest AdoptCharter for s minus later RevokeCharter`
- `great_works(s) = open + completed GreatWork* events at s`
- `treasury_milestones(s) = threshold crossings over treasury history`
- `population_milestones(s) = threshold crossings over pop(s, tick) history`

**Regions**
- `is_in_crisis(r) = threat_level(r) > 70`
- `crisis_hazard(r, pos) = f(proximity, threat, terrain)`
- `is_haunted(pos, tick) = ∃ cluster of EntityDied events (window, count≥5) within radius of pos`
- `haunted_sites(tick) = set of qualifying cluster centres`

**NPC status**
- `is_outlaw(npc) = latest between BecameOutlaw / RedemptionAccepted is BecameOutlaw`
- `is_legendary(npc) = mention_count(npc) ≥ 5 AND class_count(npc) ≥ 2 AND friend_deaths(npc) ≥ 3`
- `is_spy(npc) = exists SpyMissionStarted without corresponding SpyMissionEnded`
- `is_awakened(npc) = exists AwakeningGranted for npc`
- `treachery_score(npc) = STEALTH_tag(npc) + DECEPTION_tag(npc) − compassion_penalty`
- `legendary_name(npc) = base + " the Legendary" if is_legendary`
- `legend_halo(s) = +0.06 if any resident is_legendary`

**Lineage & items**
- `lineage(npc) = chase ChildBorn(parent, npc) upward`
- `descendants(npc) = chase ChildBorn(npc, child) downward`
- `lineage_prestige(npc) = Σ is_legendary(ancestor) for ancestor in lineage(npc)`
- `bloodline_bonus(npc) = f(lineage_prestige(npc))`
- `kill_count(item) = count(WeaponKill where weapon_id==item)`
- `is_legacy_weapon(item) = kill_count(item) ≥ N_legacy`
- `legacy_wielder_bonus(npc) = bonus if is_legacy_weapon(wielded_item(npc))`
- `eligible_artifact_bearers = npcs where level ≥ 35 AND class_count ≥ 4`
- `artifact_of(npc) = deterministic name from hash(npc_id, first_eligible_tick)`

**Prophecy & narrative**
- `prophecy_fulfilled(p) = exists event in world_events matching p.condition AND tick > p.issued_tick`
- `active_prophecies = prophecies where !prophecy_fulfilled(p)`
- `current_world_age(tick) = classify(chronicle_window_stats(tick − 2400, tick))`
- `age_history = running label over rolling chronicle windows`
- `reputation(npc, tick) = Σ recency-weighted narrative-category events involving npc`

**Meta**
- `power_rating(state) = f(friendly_count, avg_level, treasury, territory, pop, monster_density)` *(UI / training only)*
- `victory_state(state, faction_or_player) = classify({conquest, economic, cultural, survival})`

---

### Truly essential (irreducible) set in this batch

Under the strictest reading, the politics/narrative batch contains **four** irreducible responsibilities:

1. **`chronicle.push(event)`** — the act of recording any event. This is the kernel of the event-sourced architecture. Every action resolver calls it. There is no derivation from which you could recover an unwritten event.
2. **Entity ID allocation at `ChildBorn`** (`family.rs:process_births`) — when `HaveChild` resolves, a new NPC entity must be allocated, `alive=true`, with blended profile, and pushed into `state.entities`. The *decision* is EMERGENT; the push is not.
3. **Entity / settlement ID allocation at `ExpeditionArrived`** (`settlement_founding.rs:advance_settlement_founding`) — when `LeadFoundingExpedition` + `JoinExpedition` resolve on arrival, a new `SettlementState` is allocated and colonists' `home_settlement_id` is reassigned. The *decision* is EMERGENT; the push is not.
4. **`faction_ai` as the decision function for leader-class NPCs** — parallel to `action_eval.rs` for combat NPCs. This *could* be subsumed by a unified decision function once the leader action space is defined; for now, keep it as the "brain" that selects among `DeclareWar` / `SignPeace` / `LaunchConquest` / `RaiseTaxes` / etc. each tick for each faction leader. It is ESSENTIAL in the same sense that `action_eval` is ESSENTIAL — you need *something* picking what an NPC does; it is EMERGENT *in its effects* (all of which are in the leader action list above).

Notes on edge cases:

- **Team flip** (currently done by `succession`, `betrayal`, `outlaws`, `civil_war`) looks essential because `Entity::team` is a hot field consulted by combat every tick. Under pure event sourcing it is not essential — `team(npc, tick) = derive_from_latest_FactionMembershipChanged(npc, tick)`. For performance it will likely remain cached on `Entity`, but only as a materialization of the latest faction-membership event, not as a writable top-level state.
- **`state.relations[(a,b,WAR_KIND)]`** looks essential in `warfare.rs` but is also purely derived from `WarDeclared` without later `PeaceSigned`. Materialize it as a cache if needed.
- **`state.prophecies[]`** looks essential because fulfillment uses a `fulfilled` flag — but `fulfilled(p)` is a predicate query over subsequent events. The array itself need only store issuances (events), not a mutable flag.

Everything else in the batch — every war declaration, alliance, coup, vassalage oath, succession outcome, civil war ignition, betrayal, outlaw turn, founding expedition, marriage, child, prophecy, great work, charter adoption — is **an NPC action followed by a chronicle entry**. The remaining "systems" in the politics/narrative batch collapse into (a) the action vocabulary above, (b) reducers that apply event consequences, and (c) derived views that other systems and the UI query on demand.

---

## Batch 5 — World / Terrain / Low-level

### Methodology

This document re-classifies the 24 world/terrain/utility systems under a **strict event-sourcing + agentic-action** rubric. The v1 doc (`systems.md`) called 16 of 24 ESSENTIAL. Under the strict rubric, "essential" means **irreducible physics/bookkeeping**: a primitive the DSL cannot desugar away.

Four bins:

- **ESSENTIAL** — world physics primitives: voxel material write, tile write, structural collapse propagation, entity-id allocation, tick++, and the world's own stochastic event emission (the world-as-actor). Nothing an NPC can choose to do lives here.
- **EMERGENT** — anything that can be rephrased as "an NPC (or settlement leader) chose to do X, and X reduces to a sequence of essential writes". Construction, harvest, barter, migration, retirement, recruitment, class assumption, entering buildings, attending gatherings, lighting a signal, holding a festival — all emergent.
- **DERIVABLE-VIEW** — stored-as-cache values that are a pure function of other state: `season = tick/1200`, `weather = f(season, terrain, noise)`, `settlement.population = count(alive NPCs at s)`, `infrastructure_level = count(buildings at s)`, `occupants(b) = NPCs inside b`, `is_dead_zone = threat > 70`, `current_class = argmax(behavior_profile, thresholds)`. These require a view fn, not a system.
- **DEAD/STUB** — empty bodies or zero-work loops per v1 annotations (`class_progression.rs` noop, `guild_rooms.rs`/`guild_tiers.rs` treasury tax stubs, `weather.rs` stand-in). Delete or fold.

**Cited lines** trace back to `src/world_sim/systems/<file>.rs`.

---

### Per-system reanalysis

#### seasons
- **Old:** MIXED (v1 essential)
- **New:** DERIVABLE-VIEW (with one sub-emission kept ESSENTIAL)
- **Reframe:** `current_season(tick) = (tick/1200) % 4` is a pure function (seasons.rs:53-62). Every derived table — `season_modifiers`, `food_production_mult`, `food_consumption_mult`, `price_pressure`, `wilderness_food_for_season` (seasons.rs:65-285) — is a constant lookup. The per-tick side effects are re-expressible: price drift every 50 ticks (seasons.rs:201-218) is a DERIVABLE price = exponential-moving-average toward seasonal pressure (no stored memory needed since it's `settlement.prices` that drifts, but that drift is mechanical, not agentic → move it into a pure "daily market close" view). Morale drift every 10 ticks (seasons.rs:221-233) and winter spoilage / autumn harvest (seasons.rs:169-197) are deterministic functions of (season, settlement). The only non-derivable emission is the `SeasonChanged` world event (seasons.rs:163-167) at transition boundaries — but even that is derivable (`tick % 1200 == 0`) so it's not even an event, just a clock tick downstream readers can pattern-match.
- **Required NPC actions:** none.
- **Required derived views:** `season(tick)`, `season_modifiers(season)`, `food_production_mult(season)`, `price_pressure(season) → prices`, `wilderness_food(terrain, season)`.
- **Required event types:** none (the `SeasonChanged` v1 event is redundant with `season(tick)`).
- **One-line summary:** The entire seasons system is a tick-indexed constant table; delete the system, keep the functions.

---

#### weather
- **Old:** STUB/DEAD
- **New:** DEAD (or promote to a thin DERIVABLE-VIEW)
- **Reframe:** The file admits (weather.rs:62-64) "without active_weather on WorldState, we can only apply season-derived effects". `apply_winter_travel_penalty` is a literal no-op (weather.rs:90-98 "now handled by entity.move_speed_mult"). The only live path is `apply_ambient_threat_damage` (weather.rs:102-154), which is just "high-threat region → 0.1% chance damage nearby NPCs" — that belongs to dead_zones/terrain_events, not a weather system. The `WeatherEvent` struct (weather.rs:36-43) is never stored anywhere.
- **Required NPC actions:** none.
- **Required derived views:** `weather(region, tick) = f(season, region.terrain, hash(region.id, tick/7))` **if** weather becomes a real concept. Currently not a real concept.
- **Required event types:** none.
- **One-line summary:** Delete the file; if weather ever matters, it's `f(season, terrain, noise)`, not a system.

---

#### dead_zones
- **Old:** EMERGENT-CANDIDATE with stub elements
- **New:** DERIVABLE-VIEW
- **Reframe:** Header comment (dead_zones.rs:43-49) openly proxies a missing `extraction_pressure`/`dead_zone_level` field. Constants `PRESSURE_THRESHOLD`, `SPREAD_RATE`, `RECOVERY_PRESSURE_THRESHOLD`, `PRESSURE_DECAY` (dead_zones.rs:18-28) are declared and never referenced in the function body. Functionally: `is_dead_zone(region) = region.threat_level > 70`. Entity damage (dead_zones.rs:72-78) and stockpile drain (dead_zones.rs:82-99) are `f(severity, proximity_hash)` — a reactive derivation, not state.
- **Required NPC actions:** none (the "damage" effect can be folded into a terrain_events-style world-as-actor emission, but v1 already reclassified that properly).
- **Required derived views:** `is_dead_zone(region) = region.threat_level > 70`, `dead_zone_severity(region) = (region.threat_level - 70) / 30`.
- **Required event types:** none.
- **One-line summary:** Delete; the single non-trivial predicate `threat > 70` is a query, not a system.

---

#### terrain_events
- **Old:** ESSENTIAL (world disasters)
- **New:** ESSENTIAL (world-as-actor)
- **Reframe:** This is the canonical world-as-actor system: the world itself is an agent whose "intents" are volcanic eruptions (terrain_events.rs:33-37), floods (terrain_events.rs:38-42), avalanches (terrain_events.rs:43-47), forest fires (terrain_events.rs:48-59), cave collapses (terrain_events.rs:60-64), sandstorms (terrain_events.rs:65-69), corruption pulses (terrain_events.rs:70-74). Each emits `WorldRandomEvent` whose downstream effects (damage, stockpile drain, threat mutation) are mechanically derived from the event kind and region. So strictly speaking, the *emission* is essential (irreducible stochastic decision from the world-agent); the *resolution* is reducible to a derivation once the event is emitted. The `(region.id * 137.5).sin()` synthetic position hack (terrain_events.rs:92-94, 129-130, 184-185, 322-323) is load-bearing but wrong — should be `region.pos`.
- **Required NPC actions:** none — NPCs don't cause terrain events. They observe them.
- **Required derived views:** `terrain_event_damage_radius(event_kind) → f32`, `settlement_affected_by(event, settlement) → bool`.
- **Required event types:** `WorldRandomEvent { kind: Eruption|Flood|Avalanche|ForestFire|CaveCollapse|Sandstorm|CorruptionPulse, region_id, tick }`.
- **One-line summary:** The emission of the event is essential (world-as-actor); downstream damage/stockpile/threat mutations are derivations of the event.

---

#### geography
- **Old:** DUPLICATIVE with terrain_events + seasons
- **New:** DERIVABLE-VIEW (delete the system)
- **Reframe:** The file applies "one-shot effects each tick that approximate the original system's gradual changes" (geography.rs:42-45) — it doesn't store any geography state. Each of the 5 branches (geography.rs:47-157) is conditional on `(region.threat, season, settlement.population)` and emits commodity deltas. ForestGrowth overlaps `seasons.food_production_mult` + `terrain_events.forest_fire`. RiverFlood overlaps `terrain_events.flood`. DesertExpansion overlaps `terrain_events.sandstorm`. RoadDegradation is just `Damage(entity, 1)` for NPCs near high-threat regions — already dead_zones. SettlementGrowth is a small treasury tip that is identical to `population.rs`'s tax income. No unique contribution.
- **Required NPC actions:** none.
- **Required derived views:** all its effects are already covered by `seasons(tick)` and `terrain_events` emissions.
- **Required event types:** none.
- **One-line summary:** Delete; it's a duplicate layer on seasons+terrain_events with synthetic region positions.

---

#### signal_towers
- **Old:** ESSENTIAL (info diffusion)
- **New:** EMERGENT (NPC actions) + DERIVABLE-VIEW (coverage)
- **Reframe:** The tower is a Building entity; the "system" runs every 7 ticks and does two things: (1) battle damage — hostile entities near tower, 30% damage roll (signal_towers.rs:44-73) — this is just the normal combat loop applied to a building, not a tower-specific rule; (2) scouting — operational towers share price reports to friendly NPCs within range (signal_towers.rs:76-123). Price sharing is a derivable view: `price_knowledge(npc) = union of recent SharePriceReport events OR nearest_tower_coverage(npc.pos)`. The decision to *build* the tower is EMERGENT (settlement-leader action `BuildTower(loc)`). The decision to *light* the signal is EMERGENT (`LightSignal(tower)` — not implemented here but obvious extension). The automatic price-share on cadence should be an emission triggered by the tower's own "intent" (world-as-building-agent); or simply a derived view `sees_prices(npc, settlement) = ∃ operational_tower within R of both`.
- **Required NPC actions:** `BuildTower(loc)` (settlement leader), `LightSignal(tower)` (operator NPC).
- **Required derived views:** `tower_coverage(tower) = {(npc, settlement) : npc in R and settlement in R}`, `sees_prices(npc, settlement)` (replaces `SharePriceReport` broadcast).
- **Required event types:** `TowerBuilt { settlement, pos }`, `SignalLit { tower, theme }` (if we keep signaling as an emergent action).
- **One-line summary:** Tower construction is emergent; the price-share behavior is a derivable view over (operational_tower, settlement, npc) proximity.

---

#### timed_events
- **Old:** ESSENTIAL but STUB-ish
- **New:** ESSENTIAL (world-as-actor)
- **Reframe:** World emits positive buffs (trade winds treasury bonus, meteor shower commodity boost, eclipse HoT, harvest moon food, faction summit heal, ancient portal all-commodity boost — timed_events.rs:74-150). Same pattern as `terrain_events`: the *emission* is the world-agent's intent (irreducible stochastic world pulse); the *resolution* is reducible. The `estimate_active_events` function (timed_events.rs:29-45) is a header-acknowledged proxy for missing `timed_events` storage on WorldState — when that storage is added, this becomes a storage-driven emission (still essential).
- **Required NPC actions:** none — NPCs don't cause cosmic events.
- **Required derived views:** `event_selection(tick) = f(tick_hash, season)`, `event_effect_amount(event, settlement) = f(event_kind, settlement.id)`.
- **Required event types:** `WorldRandomEvent { kind: TradeWinds|MeteorShower|Eclipse|HarvestMoon|FactionSummit|AncientPortal, tick }`.
- **One-line summary:** World-as-actor positive pulses; essential emission, derivable effect.

---

#### random_events
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (world-as-actor)
- **Reframe:** Same pattern as `timed_events` but the pulses are mixed-sign: treasure (random_events.rs:35-44), harvest bounty (45-54), bandit raid (55-75), plague (76-100), famine (101-112), faction gift (113-122), equipment breakage (123-138), prophecy of doom (139-156), mercenary band (157-172). All are world-emitted; NPCs don't cause them. The String-allocating `Debuff { stat: "morale".to_string() }` (random_events.rs:148-149) is a perf hack, not a semantic issue.
- **Required NPC actions:** none.
- **Required derived views:** `event_selection(tick) = f(tick_hash)`, `event_target_selection(event, settlements) = f(tick)`.
- **Required event types:** `WorldRandomEvent { kind: Treasure|Bounty|Raid|Plague|Famine|Gift|Breakage|Doom|Mercs, tick, target: Option<SettlementId> }`.
- **One-line summary:** Siblings with timed_events — the two modules could merge into a single `WorldRandomEvent` emitter.

---

#### voxel_construction
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC action) + ESSENTIAL (voxel write)
- **Reframe:** `advance_blueprint_construction` (voxel_construction.rs:40-115) is called from work.rs per worker tick, not on cadence. It's what an NPC *does* when its intent is "build this blueprint voxel": consume one commodity unit (voxel_construction.rs:81-85), write one voxel (voxel_construction.rs:101), flip `BlueprintVoxel.placed = true` (voxel_construction.rs:104-112). The decision to place this voxel belongs to the NPC (it's chosen via goal_stack/work assignment). The **voxel write itself** (`voxel_world.set_voxel`) is ESSENTIAL physics — that's the primitive. Everything else is EMERGENT NPC action. `attach_blueprint` (voxel_construction.rs:118-149) is settlement-leader-level action (deciding to seed a building footprint). `site_clearing_targets` (voxel_construction.rs:155-190) is a query/view.
- **Required NPC actions:** `PlaceVoxel(pos, material)` (the emergent atomic), `AttachBlueprint(building, blueprint)` (settlement leader), `ClearSite(building)` (derived target list).
- **Required derived views:** `site_clearing_targets(blueprint) → [voxels needing removal]`, `next_unplaced_voxel(blueprint) → Option<(idx, mat, offset)>`, `material_commodity(mat) → commodity_idx`.
- **Required event types:** `VoxelChanged { pos, from_mat, to_mat, by: entity_id, cause: Construction }` (so chroniclers can observe who built what).
- **One-line summary:** NPC chooses `PlaceVoxel`; the voxel write is the essential physics atom.

---

#### voxel_harvest
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC action) + ESSENTIAL (voxel write + inventory add)
- **Reframe:** `harvest_tick` (voxel_harvest.rs:17-61) applies damage to the NPC's current `harvest_target`, removes the voxel when broken (voxel_harvest.rs:37), credits the yield to inventory (voxel_harvest.rs:43-45), and chains to the next adjacent voxel (voxel_harvest.rs:48-52). The *decision* to harvest is EMERGENT (NPC chose `HarvestVoxel(pos)` intent; `select_harvest_target` at voxel_harvest.rs:64-73 is the query the NPC uses). The voxel mutation (`voxel_world.mine_voxel`) is ESSENTIAL physics. The inventory write is ESSENTIAL bookkeeping.
- **Required NPC actions:** `HarvestVoxel(pos)` — with internal ticks of damage; under the hood this is a sequence of `MineVoxel(pos, dmg)` essentials.
- **Required derived views:** `find_nearest_harvestable(pos, material, r)`, `required_harvest_material(building_type)`, `yield_for(mat) → Option<(commodity, amount)>`.
- **Required event types:** `VoxelChanged { pos, from_mat, to_mat=Air, by, cause: Harvest }`, `CommodityHarvested { by, commodity, amount }` (optional — can be derived from VoxelChanged).
- **One-line summary:** NPC chooses `HarvestVoxel`; voxel removal + inventory credit are the essential atoms.

---

#### construction
- **Old:** ESSENTIAL (room growth automaton)
- **New:** EMERGENT (NPC actions) + ESSENTIAL (tile writes)
- **Reframe:** Currently a cadence-driven automaton (construction.rs:25-91) that auto-grows rooms from `BuildSeed`s: flood-fills, computes interior/boundary, then inserts Floor/Wall/Door tiles. The **tile writes** (construction.rs:71-76, 125-129, 143-148) are essential physics. The **automaton logic** (when to expand vs close, stall detection, door placement) is exactly the kind of thing the DSL should compile away: NPCs choose `PlaceTile(pos, Floor)`, `PlaceTile(pos, Wall)`, `PlaceTile(pos, Door)` actions via `action_eval`, using the same flood-fill + door-position heuristics as *queries* (views), not as system mutations. `BuildSeed` becomes "this is an NPC's (or settlement's) open room-building project" — state on the agent, not a global list. Stall detection (construction.rs:41-49) becomes an agent giving up on a goal.
- **Required NPC actions:** `PlaceTile(pos, tile_type, material)`, `OpenRoomProject(seed_pos, min_size)`, `AbandonRoomProject(seed)`.
- **Required derived views:** `flood_fill(pos, tiles) → (interior, boundary)`, `is_enclosed(boundary, tiles)`, `has_door(boundary, tiles)`, `find_door_position(boundary, tiles)`, `detect_room_function(interior, tiles) → RoomFunction`.
- **Required event types:** `TilePlaced { pos, tile_type, material, by }`, `RoomClosed { seed, interior_size, function }`.
- **One-line summary:** NPC chooses `PlaceTile` using flood-fill views; the tile write is the essential atom — delete the automaton.

---

#### structural_tick
- **Old:** ESSENTIAL
- **New:** ESSENTIAL (physics)
- **Reframe:** This is irreducible world physics: "unsupported voxels fall". `structural_tick` (structural_tick.rs:23-79) BFS-walks every dirty chunk, anchors from ground/Granite/z<=0, removes any solid voxel not reached (structural_tick.rs:67-69), emits `StructuralEvent::FragmentCollapse` (structural_tick.rs:71-77). No NPC chooses collapse; it's a consequence of the *combined* voxel field. This is the only system in the batch that is genuinely, unambiguously ESSENTIAL under the strict rubric.
- **Required NPC actions:** none.
- **Required derived views:** `is_anchored(voxel, world)` (BFS from ground — the core algorithm).
- **Required event types:** `StructuralCollapse { chunk, affected_voxels, reason: Natural|Induced }`.
- **One-line summary:** The one irreducible physics system in this batch.

---

#### interiors
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC actions) + DERIVABLE-VIEW (occupancy)
- **Reframe:** Two phases: (1) clear stale occupancies (interiors.rs:32-55) — iterates rooms, checks if `room.occupant_id`'s NPC still has `inside_building_id == this_building.id`; if not, clears. This is pure validation of a denormalized cache. (2) NPCs near their target building enter it, claim a room of kind matching their action (interiors.rs:58-177). Both are re-expressible: **occupancy is derivable** — `occupants(b) = [npc for npc in NPCs if inside_building_id == b.id]`; **room occupancy** is derivable — `room_occupant(b, ri) = argmin_distance(npcs_inside_b, room.offset)` or pick-one. The **act of entering** is an NPC action `EnterBuilding(b)` that flips `inside_building_id`. The **act of leaving** is `LeaveBuilding()`. The `(bi, ri) clears` vec at interiors.rs:32-55 is just a consistency sweep on a cache the DSL wouldn't store.
- **Required NPC actions:** `EnterBuilding(b)` (sets `inside_building_id`, `current_room`), `LeaveBuilding()` (clears both).
- **Required derived views:** `occupants(b)`, `room_occupant(b, room_idx)`, `preferred_room_kind(npc.action) → RoomKind`.
- **Required event types:** `EnteredBuilding { npc, building, room }`, `LeftBuilding { npc, building }`.
- **One-line summary:** Entering/leaving is an NPC action; room occupancy is a derived index over NPCs inside.

---

#### social_gathering
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC actions)
- **Reframe:** `advance_social_gatherings` (social_gathering.rs:25-210) collects NPCs wanting to socialize, pair-matches them within 10u, starts a 15-tick `Socializing` action on both, boosts `needs.social +25`, accumulates DIPLOMACY/TEACHING tags, exchanges `HeardStory` beliefs about the most dramatic memory. Every one of these is re-expressible as NPC actions: `AttendGathering(loc)`, `ConverseWith(other)`, `ShareStory(other, story_id)`. The `social_npcs` collection (social_gathering.rs:32-46) is just a query for "NPCs wanting social". The O(N²) pair matching (social_gathering.rs:52-73) is choosing an action target. The `record_npc_event` calls (social_gathering.rs:89-101) are the side-effects of the conversation action. The memory/belief mutations (social_gathering.rs:134-192) are the effects of `ShareStory`.
- **Required NPC actions:** `AttendGathering(loc)`, `ConverseWith(other_npc)`, `ShareStory(target, story)` (action with side-effects on both).
- **Required derived views:** `wants_to_socialize(npc) = npc.needs.social < 40 || npc.goal_stack.has(Socialize)`, `best_story(npc) = argmax_impact(memory.events)`, `has_heard(npc, about) = belief.any(HeardStory{about})`.
- **Required event types:** `ConversationStarted { a, b, ticks }`, `StoryShared { teller, listener, about, impact }`.
- **One-line summary:** Gathering/conversing/storytelling are all NPC actions; the system is a scheduler the DSL can compile away.

---

#### festivals
- **Old:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New:** EMERGENT (settlement-leader action) + DERIVABLE-VIEW
- **Reframe:** The compiled behavior (festivals.rs:49-64) is literally "autumn → +5 FOOD, summer → +10 treasury, spring/winter → nothing". No actual festival state — no attendees, no theme, no duration. Could be deleted and folded into seasons. But the real DF-style festival is emergent: a settlement leader *declares* a festival with theme; NPCs *attend*; attendees gain morale; the settlement pays a cost. That's the DSL target: `HoldFestival(theme)` is a settlement-leader action that posts a `FestivalDeclared` event; NPCs can choose `AttendFestival(settlement)`; `festival_active(settlement) = ∃ FestivalDeclared in last N ticks`.
- **Required NPC actions:** `HoldFestival(theme)` (settlement leader), `AttendFestival(settlement)` (NPC).
- **Required derived views:** `festival_active(s) = ∃ FestivalDeclared{s} in [tick-N, tick]`, `festival_attendees(s, tick) = npcs attending`.
- **Required event types:** `FestivalDeclared { settlement, theme, start_tick, duration }`, `FestivalAttended { npc, festival }`.
- **One-line summary:** Current festivals is dead; real festivals are a settlement-leader emergent action with derivable "active" window.

---

#### guild_rooms
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** 36 lines (guild_rooms.rs:1-36). `compute_guild_rooms_for_settlement` derives `room_count = min(treasury/200, 5)` and emits `UpdateTreasury(-room_count * 0.5)` (guild_rooms.rs:26-35). No guild_room state is stored. This is literally a treasury-indexed negative scalar; it's not guild-rooms, it's upkeep.
- **Required NPC actions:** none.
- **Required derived views:** if we actually want guild upkeep: `upkeep(s) = f(buildings_at_s)`; derive from `count(Building entities where entity.settlement_id == s.id)`.
- **Required event types:** none.
- **One-line summary:** Delete; zero guild-room state exists — it's just negative treasury drift.

---

#### guild_tiers
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** 34 lines (guild_tiers.rs:1-34). Identical pattern to guild_rooms but positive: `tier = min(treasury/500, 5)`, emits `ProduceCommodity(FOOD, tier * 0.02)` (guild_tiers.rs:26-33). Tiny tip (<0.1 FOOD) indexed by treasury. No tier state stored.
- **Required NPC actions:** none.
- **Required derived views:** `guild_tier(s) = min(s.treasury / 500, 5)` — if needed anywhere, it's a view.
- **Required event types:** none.
- **One-line summary:** Delete; same dead-stub pattern as guild_rooms.

---

#### migration
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC action)
- **Reframe:** `compute_migration` (migration.rs:63-168) scans settlements for high threat (>30) or low food (<5), picks the best reachable alternative (migration.rs:84-107), and for each NPC that *knows* the destination (migration.rs:127-133) and rolls <5% (migration.rs:136-139), emits `SetIntent(Travel{destination})` + optional chronicle. Every decision here is the NPC choosing to migrate: `MigrateTo(settlement)` is a pure NPC action. The eligibility ("this settlement has high threat") is an input the NPC observes, not a system rule. The migration chance is an NPC's internal decision threshold. Note: v1 observed that `home_settlement_id` reassignment isn't here — it's presumably handled on arrival; the proper EMERGENT model has `ArriveAtSettlement` set the new home.
- **Required NPC actions:** `MigrateTo(settlement)` (sets EconomicIntent::Travel to dest, later flips home_settlement_id on arrival).
- **Required derived views:** `settlement_attractiveness(s, regions)`, `migration_candidates(npc) = settlements npc.price_knowledge.contains within 100u`, `should_flee(settlement) = threat > 30 || food < 5`.
- **Required event types:** `MigrationEvent { npc, from, to, reason }`.
- **One-line summary:** Migration is a pure NPC action; the system is scheduling dressed up as essential.

---

#### retirement
- **Old:** ESSENTIAL
- **New:** EMERGENT (NPC action) + ESSENTIAL (entity lifecycle)
- **Reframe:** `compute_retirement_for_settlement` (retirement.rs:149-222) scans NPCs eligible (level≥10, hp_ratio≥0.7, not on hostile grid), rolls 10% (retirement.rs:190-193), emits `Die` + `UpdateTreasury` + optional `UpdateStockpile(FOOD)` for Quartermaster (retirement.rs:203-220). The decision to retire is the NPC's (`Retire()`). The *effect* is entity lifecycle change (Die delta) + settlement bonus; Die is ESSENTIAL bookkeeping (entity transitions alive→dead), the treasury/stockpile bonus is just `UpdateTreasury` emission from the retire action itself. The legacy-type classification (retirement.rs:51-125) is a derived view on `class_tags`.
- **Required NPC actions:** `Retire()` (NPC decides); under the hood: `Die(self)` + `GrantLegacy(home_settlement, legacy_type)`.
- **Required derived views:** `legacy_type(npc.class_tags, hash) → LegacyType`, `legacy_bonus(legacy, level) → f32`, `retirement_eligible(npc) = level>=10 && hp_ratio>=0.7 && !on_hostile_grid`.
- **Required event types:** `Retired { npc, settlement, legacy }`, plus the underlying `EntityDied { id, cause: Retirement }`.
- **One-line summary:** NPC chooses `Retire()`; the Die delta is the essential lifecycle primitive.

---

#### progression
- **Old:** ESSENTIAL by action, DERIVATIONAL by design
- **New:** DERIVABLE-VIEW
- **Reframe:** `compute_progression_for_settlement` (progression.rs:54-141) explicitly derives `entity.level = Σ class.level` (progression.rs:68) and back-computes the stat delta since last sync. The comment "XP removed — entity level derived from class levels" (progression.rs:1-10) confirms the derivation story. Stats could be computed on-demand from `npc.classes` every time they're read — no need to store the denormalized `entity.max_hp/attack/armor/move_speed`. Or store them as a cache rebuilt when classes change. Either way, there is no decision being made here; it's a projection from (classes, registry) → stats.
- **Required NPC actions:** none (leveling happens in the class matcher, not here).
- **Required derived views:** `entity_level(npc) = Σ class.level`, `entity_stats(npc, registry) = weighted_sum(class.per_level_stats)`.
- **Required event types:** `LeveledUp { npc, new_level, class }` (optional — observable projection).
- **One-line summary:** Delete; level and stats are pure functions of `npc.classes` — make them views.

---

#### class_progression
- **Old:** STUB/DEAD
- **New:** DEAD
- **Reframe:** `compute_class_progression_for_settlement` (class_progression.rs:27-54) filters NPCs with `behavior_sum >= 10.0` then the function body is literally one comment line — "XP removed — entity level derived from class levels" (class_progression.rs:52). Nothing is emitted. The module header (class_progression.rs:4-9) acknowledges class acquisition is handled by `ClassGenerator` trait elsewhere.
- **Required NPC actions:** if we add agentic class assumption: `AssumeClass(class)` when `behavior_profile` passes threshold.
- **Required derived views:** `current_class(npc) = class such that behavior_profile passes threshold` (if we want it; otherwise class is just stored).
- **Required event types:** `ClassAssumed { npc, class }`.
- **One-line summary:** Delete the file; emergent version is `AssumeClass(class)` NPC action keyed on behavior_profile view.

---

#### recruitment
- **Old:** ESSENTIAL (structural pooling)
- **New:** EMERGENT (settlement-leader action) + ESSENTIAL (entity revive)
- **Reframe:** `compute_recruitment_for_settlement` (recruitment.rs:37-107) is the settlement-leader deciding to birth NPCs from the dead pool when food surplus + pop headroom allow: compute `target_births = min(8*growth_factor, food_surplus/0.3, 8)` (recruitment.rs:61-71), scan dead NPCs at settlement (recruitment.rs:79-85), then unaffiliated pool (recruitment.rs:88-97), revive each via `Heal(200) + SetPos + SetIntent(Produce)` (recruitment.rs:109-130). This is `Recruit(n)` settlement-leader action. The *revival* is ESSENTIAL bookkeeping — flipping a dead slot to alive is an irreducible entity-lifecycle primitive (inverse of Die). The *decision to recruit* (how many, which dead to pick) is EMERGENT.
- **Required NPC actions:** `Recruit(n)` (settlement leader, bounded by food/pop/dead-pool).
- **Required derived views:** `alive_at(s)`, `food_surplus(s)`, `growth_factor(alive, capacity) = max(1 - alive/500, 0.05)`, `dead_pool(s)`, `unaffiliated_dead_pool()`.
- **Required event types:** `RecruitmentEvent { settlement, revived_npc, source: OwnDead|Unaffiliated }`, plus the underlying `EntityRevived { id, pos }`.
- **One-line summary:** Settlement leader chooses `Recruit`; the revive (dead→alive flip) is the essential bookkeeping primitive.

---

#### npc_decisions
- **Old:** ESSENTIAL (barter + trade completion)
- **New:** EMERGENT (NPC actions)
- **Reframe:** Comment at line 130 openly states "NPC action scoring now handled by action_eval.rs. Only trade arrival and barter remain here." The trade-arrival path (npc_decisions.rs:40-128) is the NPC's `CompleteTrade` action: arrive within 3u of dest, dump all inventory at dest prices, earn gold, record trade, set back to Produce. The barter path (npc_decisions.rs:152-265) pairs producers of different commodities at same settlement, swaps at local prices bounded by carry capacity — that's `BarterWith(other_npc, give_commodity, want_commodity, amount)`. Both are pure NPC actions with derivable preconditions. The `producers: [(u32,usize,f32); 64]` stack array (npc_decisions.rs:165-180) is a candidate-list view, not state.
- **Required NPC actions:** `CompleteTrade(dest)` (sell all carried at dest prices → earn gold → return to Produce), `BarterWith(other_npc, give, want, amount)` (commodity swap at local prices).
- **Required derived views:** `arrived_at(npc, settlement) = dist(npc.pos, settlement.pos) < 3`, `carry_capacity(npc) = level*5+10`, `total_carried(npc)`, `barter_candidates(settlement) = NPCs with production[0] at s`, `fair_ratio(a_commodity, b_commodity, prices) in 0.33..3`.
- **Required event types:** `TradeCompleted { npc, home, dest, profit }`, `BarterCompleted { a, b, a_gave, b_gave }`.
- **One-line summary:** Trade completion and barter are NPC actions; delete the system, move the logic into `action_eval` with views.

---

#### population
- **Old:** MIXED (essential per-storage)
- **New:** DERIVABLE-VIEW (the settlement.population scalar)
- **Reframe:** `compute_population_for_settlement` (population.rs:56-193) is pure scalar-drift math on `settlement.population`: food consumption `= pop * 0.001 * cap(stockpile)` (population.rs:77-88), growth rate from `(food_ratio, treasury, threat)` (population.rs:90-128), clamped by `MIN_POPULATION` / `MAX_POPULATION` (population.rs:141-147), tax income `= pop * 0.001` (population.rs:157-163), surplus stockpile on growth (population.rs:167-177), decline stockpile drain (population.rs:181-192). The key insight: **`settlement.population` as a stored scalar is a redundant view over `count(alive NPCs with home_settlement_id == s.id)`**. In a pure-event model, delete the scalar; derive it from the NPC roster. The food-consumption, tax, and stockpile-drift emissions then hang off *NPC lifecycle events* (Born, Died, MigrateTo) rather than a scalar drift system. Birth/death are handled by `recruitment` and combat.
- **Required NPC actions:** none (births/deaths are Recruit/Die).
- **Required derived views:** `population(s) = count(alive NPCs with home_settlement_id == s.id)`, `food_demand(s) = population(s) * FOOD_PER_POP`, `tax_income(s) = population(s) * TAX_PER_POP`, `growth_shape(food_ratio, treasury, threat, pop, capacity) → f32` (only used to decide Recruit rate — so it moves to recruitment).
- **Required event types:** none (all derivable from per-NPC events).
- **One-line summary:** Delete the scalar drift; population is `count(NPCs at s)`; food/tax/stockpile effects attach to NPC lifecycle events instead.

---

### Reduction summary

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

### Required action vocabulary

#### NPC actions (commoner)

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

#### Settlement-leader actions

- `Recruit(n)` — revive n dead NPCs from own pool, then unaffiliated; consume food. (recruitment)
- `HoldFestival(theme)` — post FestivalDeclared event; pay treasury cost; attendees gain morale. (festivals)
- `BuildTower(loc)` — seed a tower Building entity with blueprint at loc. (signal_towers)
- `AttachBlueprint(building, blueprint)` — seed voxel construction. (voxel_construction)
- `GrantLegacy(legacy_type)` — receive side-effect of an NPC's Retire(). (retirement)

---

### Required event types

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

### Required derived views

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

### Truly essential (irreducible) set in this batch

The minimum set of primitives the DSL must provide so everything else desugars cleanly:

1. **Voxel material write** — `set_voxel(pos, material)`. Used by harvest, construction, collapse, terrain-events. (voxel_harvest.rs:37, voxel_construction.rs:101, structural_tick.rs:68)
2. **Tile write** — `set_tile(pos, tile)`. Used by construction. (construction.rs:71-76, 125-129, 143-148)
3. **Structural collapse propagation** — "unsupported solids fall" BFS from anchors. Irreducible combinational physics. (structural_tick.rs:23-79, 101-250)
4. **Entity ID allocation / revive** — flipping `alive=false → alive=true` and `alive=true → alive=false`. (recruitment.rs:109-130 revive; retirement.rs:203 die). Die is a projection of HP<=0 in normal play, but as a lifecycle atom it's irreducible — identity persists, state bit flips.
5. **Tick++** — advancing `state.tick`. All cadences and derivations depend on it.
6. **World-random-event emission** — the world-as-actor's stochastic intents (terrain, timed, random). Irreducible because the choice "does volcano erupt this tick" is not an agent decision and not a pure state function (it's a draw from the RNG gated by state).

Everything else in this batch is either an NPC/settlement-leader emergent action built on these primitives, or a derivable view over state.

---

### Notes on re-classification

- v1 called `voxel_construction`, `voxel_harvest`, `construction` ESSENTIAL at the system level. Strict: the *system* is EMERGENT (NPC choosing what to build/harvest); only the underlying `set_voxel` / `set_tile` *atom* is essential.
- v1 called `npc_decisions`, `migration`, `retirement`, `recruitment`, `interiors`, `social_gathering`, `signal_towers` ESSENTIAL. Strict: each is an NPC or settlement-leader decision + effect, fully emergent, with at most a Die/Revive primitive touching the lifecycle atom.
- v1 had `progression` straddling; strict: it's a pure derivation from class levels — delete the system, add a view.
- v1 had `population` MIXED (scalar is state); strict: delete the scalar, derive from NPC count. This collapses the two-layer population abstraction that v1 flagged as a load-bearing pattern.
- `seasons.rs` in v1 had essential emissions (winter spoilage, autumn harvest, festival morale, morale drift, price drift). Strict: every one of these is `f(season, settlement_id)` and season is `f(tick)` — they're all derivations. The only reason they exist as deltas is because in the current architecture, "settlement state changes" must flow through the delta pipeline. A view-based architecture reads `settlement.food_at_tick(t) = base_food - winter_loss(t) + autumn_bonus(t)` directly.
- Unified `WorldRandomEvent` stream: `terrain_events` + `timed_events` + `random_events` all emit the world-agent's stochastic intents. Their handlers differ but the emission is one primitive event type with a kind discriminator. Three files can fuse into one.
