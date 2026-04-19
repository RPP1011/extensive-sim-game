# Economic Systems — Strict Reframing

## Methodology

Every system is re-evaluated against a much tighter rubric than v1. The default answer is no longer "this owns state, so it's essential" — the default answer is "this could be an NPC action, so it's emergent."

Four classes:

- **ESSENTIAL** — encodes an irreducible physics/bookkeeping rule: combat damage emission, voxel material change, tile placement, entity ID allocation (spawn), `alive=false` (despawn), `pos += force × dt`, `tick++`, RNG advance, or a spatial/group index that provably cannot be derived from event history. If there is any way to reframe the system's effect as (derived view) → (NPC chooses action) → (action emits event) → (physics rule applies), it is NOT essential.
- **EMERGENT** — the system is currently a central planner doing something an NPC could choose to do. Replace the system with an action vocabulary on the agent side. The system's effect re-appears as the sum of NPC choices, mediated by events.
- **DERIVABLE-VIEW** — a pure function of current state + event history. No system needed; the "state" is a query. Examples: treasury is `Σ income − Σ spend`, price is `f(stockpile, pop)`.
- **DEAD** — function body is a stub / no-op / noise, or admitted-approximation that has never owned real state.

The rubric forces us to separate **physics** (irreducible world change) from **policy** (who decides when it happens). Systems that only encode policy disappear in favor of NPC agency; only the event that physics consumes survives.

---

## Per-system reanalysis

### economy
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT (with one DERIVABLE-VIEW half)
- **Reframe:** The three transitions v1 claimed as irreducible are all policy choices that belong on agents. (1) Taxation (`economy.rs:49-89`) is a settlement leader choosing `LevyTax(rate)` and each resident choosing `PayTax(amount)` — the current `tax_eff = 1/(1 + treasury/10000)` is just a decision rule the leader would apply. (2) Admin overhead and maintenance (`economy.rs:106-121` surrounding block) are settlement-scoped spend decisions — `PaySalary(role)`, `PayUpkeep(target)`. (3) Debt repayment (`advance_debt`, `economy.rs` direct-mut) is literally the debtor choosing `Repay(creditor, amount)` — it already encodes a rule ("20 % of income, capped at debt, capped at gold") that NPCs could just follow as a utility. The price formula is DERIVABLE-VIEW: `price[c] = base[c] / (1 + (stockpile[c]/pop)/50)` is a pure function.
- **Required NPC actions:** `LevyTax(rate)`, `PayTax(amount)`, `PaySalary(role, amount)`, `PayUpkeep(target, amount)`, `Repay(creditor_id, amount)`.
- **Required derived views:** `treasury(sid) = init + Σ income − Σ spend`, `price(c, sid) = f(stockpile, pop)`, `outstanding_debt(npc) = Σ borrowed − Σ repaid`.
- **Required event types:** `TaxPayment {from_npc, to_settlement, amount}`, `Upkeep {settlement, target, amount}`, `DebtRepayment {debtor, creditor, amount}`, `MoraleDelta {npc, amount, reason}`, `CreditHistoryDelta {npc, delta}`.
- **Verdict:** The only "system" that remains is the event-application rule (ledger fold). Behavior moves to agents.

### food
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT
- **Reframe:** Production, wage collection, consumption, and starvation are the exact per-NPC workflow v1 describes in its DSL (`food.rs:103`, `food.rs:153-157`). A working NPC chooses `Produce(commodity, rate × level_mult × scale)` once per work cycle; a resident NPC chooses `Eat(food_source)` or suffers a physics-level starvation damage event emitted by a `hunger >= threshold` trigger. The settlement wage is just the employer's `PaySalary(worker, rate × price[c])` — exactly what economy already does. Nothing in this file is an irreducible rule; it is all policy + accounting.
- **Required NPC actions:** `Produce(commodity, amount)`, `Consume(commodity, amount)` (recipe inputs), `Eat(food_source)`, `BeginWorkShift(building)`, `EndWorkShift`.
- **Required derived views:** `stockpile(c, sid) = Σ produce − Σ consume`, `hunger(npc) = last_hunger + tick_delta − Σ eat_amount`, `recipe_feasibility(c, sid) = min(stockpile[inputs]/needed, 1)`.
- **Required event types:** `Production {producer, settlement, commodity, amount}`, `Consumption {consumer, settlement, commodity, amount}`, `Meal {eater, source, hunger_restore}`, `Starvation {npc, damage, morale_delta}`, `WageEarned {worker, commodity, gold}`.
- **Verdict:** Dissolved into agent actions; only the starvation physics rule (`hp -=` on a hunger threshold event) is an event-driven damage application.

### crafting
- **Old classification:** DUPLICATIVE / STUB
- **New classification:** DEAD (for the regen half) + EMERGENT (for the ore+wood→luxury half)
- **Reframe:** The natural regen (`crafting.rs:68-76`) is noise that doesn't correspond to any agent action — delete. The luxury recipe is a duplicate of what `food.rs` recipes already encode; an NPC with a "craftsman" profile chooses `Produce(Luxury, 1)` with `Consume(Ore, 2) + Consume(Wood, 2)`. No system needed; this is one row in a recipe table.
- **Required NPC actions:** covered by `Produce` + `Consume` above.
- **Required derived views:** stockpile fold.
- **Required event types:** covered by `Production` / `Consumption`.
- **Verdict:** Delete. Fold the luxury recipe into the shared recipe registry.

### buildings
- **Old classification:** ESSENTIAL (spawn + voxel stamp + nav rebake)
- **New classification:** mostly EMERGENT with a thin ESSENTIAL core
  - ESSENTIAL: entity spawn (`state.entities.push` for a new Building ID), voxel stamp (`state.voxel_world.set_voxel`), nav grid rebake. These are physics: allocating an ID, changing a voxel material, invalidating a spatial cache.
  - EMERGENT: `advance_construction` is an NPC `WorkOnBuilding(id, amount)` loop; `assign_npcs_to_buildings` is an NPC `ClaimWorkSlot(building)` or `ClaimResidence(building)` choice; `update_building_specializations` is a DERIVABLE-VIEW over `worker_class_ticks`; `compute_buildings` treasury upgrade cost is a leader's `PayUpgrade(building)` choice.
- **Reframe:** The file's 1014 lines collapse to: a BlueprintPlacer rule (voxel stamp when a `PlaceTile(pos, material)` event arrives), an ID allocator (when a `StartConstruction(blueprint)` event arrives), and a nav rebake trigger (when `PlaceTile` or `StartConstruction` events affect nav). Everything else is agent policy. The zone-score matching, blueprint attaching, and worker class tallying are all downstream of individual NPC actions + straightforward reductions.
- **Required NPC actions:** `StartConstruction(blueprint, pos)`, `PlaceTile(pos, material)`, `WorkOnBuilding(building_id, effort)`, `ClaimWorkSlot(building_id)`, `ClaimResidence(building_id)`, `PayUpgrade(building_id, amount)`.
- **Required derived views:** `specialization(building) = argmax(worker_class_ticks)/sum(worker_class_ticks)`, `construction_progress(building) = Σ WorkOnBuilding.effort / blueprint.cost`, `is_built(building) = construction_progress >= 1`.
- **Required event types:** `ConstructionStarted {initiator, building_id, blueprint}`, `TilePlaced {placer, pos, material}` (ESSENTIAL emitter), `ConstructionProgress {worker, building, effort}`, `WorkSlotClaimed {npc, building}`, `ResidenceClaimed {npc, building}`, `BuildingUpgraded {building, tier}`, `NavGridDirty {region, aabb}` (ESSENTIAL trigger).
- **Verdict:** Keep only the physics kernel (tile placement, spawn, nav). The orchestration melts into agent actions and their aggregates.

### work
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT
- **Reframe:** The work.rs state machine (`work.rs` WorkState: Idle→Traveling→Working→Carrying→Idle) is exactly the "agent plan" v1 admits in its own DSL. Every transition is an agent choice: `GoToWorkplace` (Idle→Traveling), `BeginProduction` (Traveling→Working), `FinishProduction` emits a `Production` event (Working→Carrying), `DepositAtSite` emits `Deposit` (Carrying→Idle). Wage and stockpile mutations are agent `Deposit` + settlement `PaySalary` actions. Forge spawning an item is an NPC action `ForgeItem(recipe)` that emits `ItemCrafted {item_id, owner_id}` which is ESSENTIAL physics (spawn). Eating is `Eat(source)`. Voxel harvest is an NPC `HarvestTile(pos)` action that emits `TilePlaced(pos, Air)` + `ResourceHarvested(node, amount)` — the voxel change is ESSENTIAL but the choice to harvest is not. Even the price-belief update is an agent learning rule: `UpdatePriceBelief(c, observed)`.
- **Required NPC actions:** `GoToWorkplace(building_id)`, `BeginProduction(building_id)`, `FinishProduction`, `DepositAtSite(target_id, commodity, amount)`, `ForgeItem(recipe)`, `HarvestTile(pos)`, `HarvestNode(node_id)`, `Eat(source)`, `PurchaseFromStockpile(settlement, commodity, amount)`, `UpdatePriceBelief(c, observed)`.
- **Required derived views:** `inventory(npc)`, `stockpile(c, sid)`, `hunger(npc)`, `income_rate(npc) = EMA over WageEarned`, `price_beliefs(npc, c) = fold over observations`.
- **Required event types:** `ProductionStarted`, `ProductionFinished`, `Deposit {from, to, commodity, amount}`, `ItemCrafted {crafter, item_entity_id, quality, slot}` (ESSENTIAL: new entity ID), `TileHarvested {harvester, pos, material_before}` (ESSENTIAL: voxel change), `ResourceHarvested {harvester, node_id, amount}`, `WageEarned`, `MealEaten`, `PriceBeliefUpdated`.
- **Verdict:** The file becomes a Vec of agent utilities; the simulation's only job is to apply the events they emit.

### gathering
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT (with physics event for harvest + despawn)
- **Reframe:** `gathering.rs:advance_plans` is a hard-coded plan interpreter: `Gather, MoveTo, Perform, PlaceBuilding, PayGold, Wait`. Each variant is already a named action (the grammar is right there). Promote them to agent-level actions and delete the plan-step-bookkeeping layer — an agent doesn't need a "plan index" in central state if its own utility function picks the next action each tick. `resource.remaining -=` and depletion's `alive=false` are ESSENTIAL physics on a `HarvestNode` event. Full-entity scans for nearest-resource become a pure spatial query, not state owned by this system.
- **Required NPC actions:** `HarvestNode(node_id, amount)`, `MoveTo(pos)`, `PerformTimed(ticks)`, `PlaceBlueprint(pos, type)`, `PayGold(target, amount)`, `Wait(ticks)`.
- **Required derived views:** `nearest_resource(pos, rtype)` is a spatial-index query; `node.remaining` is a state field on the resource entity; plan progress is just the agent's own memory.
- **Required event types:** `ResourceHarvested {harvester, node_id, amount}`, `NodeDepleted {node_id}` (ESSENTIAL: emits despawn), `Movement {entity, from, to}`, `GoldPaid {from, to, amount}`.
- **Verdict:** The plan-executor vanishes; agents pick actions directly. The event-apply layer handles `remaining -=` and the depletion → `alive=false` transition.

### trade_goods
- **Old classification:** EMERGENT-CANDIDATE + DUPLICATIVE
- **New classification:** DERIVABLE-VIEW (price) + EMERGENT (caravan)
- **Reframe:** The price formula (`trade_goods.rs:42-45`) is literally `demand/max(stockpile, 0.1)` — pure function. Remove, replace with `fn price(sid, c)`. The caravan arbitrage (`trade_goods.rs:57-119`) is a central planner doing an NPC's job: discover max-min price gap, pick goods, move them, collect profit. This is an NPC `RunCaravan(src, dst, commodity, amount)` action; the settlement's treasury gain is a `GoldReceived` event. The regen duplicates `crafting.rs`; delete.
- **Required NPC actions:** `RunCaravan(src, dst, commodity, amount)` (or decomposed: `BuyFromSettlement`, `Travel`, `SellToSettlement`).
- **Required derived views:** `price(c, sid)`, `stockpile(c, sid)`, `arbitrage_opportunities() = pairs where price_dst/price_src > threshold`.
- **Required event types:** `CaravanDeparted {trader, src, dst, commodity, amount}`, `CaravanArrived {trader, dst, commodity, delivered_amount, transit_loss}`, `GoldReceived {recipient, amount, source}`.
- **Verdict:** Price formula becomes a query; caravan becomes an agent action.

### trade_routes
- **Old classification:** ESSENTIAL
- **New classification:** DERIVABLE-VIEW (mostly) + EMERGENT (agent history side)
- **Reframe:** v1 claimed `Vec<TradeRoute>` was irreducible because it tracks establishment tick + strength. But every field is derivable: `strength(src, dst) = f(time since last trade, Σ profit)` is a pure fold over `CaravanArrived` events with a decay kernel. `trade_count(src, dst) = count(CaravanArrived)`. The "establish when count >= 3" rule becomes a query that returns which pairs qualify. The chronicle entry on establishment/abandonment is an ESSENTIAL narrative event but can be emitted by a trigger on the view crossing a threshold. `npc.trade_history` is agent-side memory (already path-dependent on agent, but that's per-NPC state naturally).
- **Required NPC actions:** implicit via `RunCaravan` (feeds the fold).
- **Required derived views:** `route_strength(a, b) = Σ(CaravanArrived(a→b).profit × exp(-(now−tick)/tau))`, `established_routes() = pairs where trade_count >= 3`, `route_utility_bonus(a, b) = f(strength)`.
- **Required event types:** `CaravanArrived` (above) is sufficient input. Narrative triggers: `RouteEstablished {a, b}`, `RouteAbandoned {a, b}` emitted when the derived view crosses ±threshold.
- **Verdict:** Persistent Vec goes away; read-time queries replace it. Chronicle entries are threshold-triggered events.

### trade_guilds
- **Old classification:** EMERGENT-CANDIDATE (price half) + ESSENTIAL (gold half)
- **New classification:** EMERGENT
- **Reframe:** "Settlement with ≥3 traders forms a guild" is a derivable predicate. The price floor `max(0.7, f(state))` is a read-time clamp on the derived price. The guild-funds-best-merchant-for-15 gold is simply an NPC (or guild-leader NPC) taking action `FundMerchant(target, amount)` when they qualify — there is no guild entity required, just a decision rule on the merchants who satisfy the membership predicate. Rival undercutting is a merchant's `AdjustLocalPrice(c, delta)` choice (or via a posted contract to suppress rivals); it should not be a centrally-applied price mutation. NPC tag accumulation is downstream of choice (agent `AccumulateTags` on successful action).
- **Required NPC actions:** `FundMerchant(target, amount)`, `AdjustLocalPrice(commodity, delta)` (or we drop price-writing-as-action and keep price purely derivable), `CoordinateUndercut(rival_settlement, commodity)`.
- **Required derived views:** `is_guild_member(npc)`, `guild_members(sid) = list where trade+neg > 30`, `has_guild(sid) = guild_members.len >= 3`.
- **Required event types:** `GuildFunded {funder, recipient, amount}`, `GuildFormed {sid}` (threshold trigger), `PriceAdjustment {settlement, c, delta, reason}`.
- **Verdict:** Dissolves into agent actions + a threshold-triggered narrative event.

### contracts
- **Old classification:** ESSENTIAL (advance) + STUB (compute)
- **New classification:** EMERGENT
- **Reframe:** `compute_contracts` is a random treasury nudge (`contracts.rs:53`) — DEAD. `advance_contracts` is the full bidding lifecycle — but every transition is an NPC choice. The contract requester chooses `PostContract(spec)`; eligible bidders choose `BidOnContract(contract_id, bid_amount)`; the requester chooses `AcceptBid(bid_id)` (or a scoring rule auto-accepts the best); payment on completion is `PayContract(provider, amount)`. The `service_contracts: Vec<ServiceContract>` becomes a query over open `ContractPosted` events that haven't received `ContractCompleted`.
- **Required NPC actions:** `PostContract(spec)`, `BidOnContract(contract_id, bid_amount)`, `AcceptBid(bid_id)`, `WithdrawBid(bid_id)`, `MarkContractFulfilled(contract_id)`, `PayContract(provider, amount)`.
- **Required derived views:** `open_contracts(sid)`, `bids_on(contract_id)`, `eligible_bidders(contract_id) = idle npc matching service type`.
- **Required event types:** `ContractPosted {poster, service, max_payment, deadline}`, `BidPlaced {bidder, contract, amount}`, `BidAccepted {contract, bid}`, `ContractCompleted {contract, provider}`, `ContractPayment {from, to, amount}`, `ContractExpired {contract}`.
- **Verdict:** Replace entire 403-line lifecycle with action vocabulary + event-sourced queries.

### contract_negotiation
- **Old classification:** STUB/DEAD
- **New classification:** DEAD
- **Reframe:** Pure random treasury bonus (`contract_negotiation.rs:8` admits placeholder). Delete. If real negotiation is desired later, it's `NegotiateTerms(contract, counter_offer)` — an agent action that modifies a pending `ContractPosted` before `BidAccepted` fires.
- **Required NPC actions:** (future) `CounterOffer(contract, terms)`.
- **Required event types:** (future) `ContractAmended`.
- **Verdict:** Delete.

### auction
- **Old classification:** STUB
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** The current file (`auction.rs:30`) is a slow treasury drain — delete. A real auction is just `PostAuction(item, reserve)`, `PlaceAuctionBid(auction, amount)`, `CloseAuction(auction)` — three agent actions mediated by events. No system needed.
- **Required NPC actions:** `PostAuction(item_id, reserve_price, deadline)`, `PlaceAuctionBid(auction_id, amount)`, `CloseAuction(auction_id)`.
- **Required derived views:** `open_auctions(sid)`, `high_bid(auction_id)`.
- **Required event types:** `AuctionPosted`, `AuctionBid`, `AuctionClosed {winner, final_price}`, `ItemTransferred {from, to, item}`, `GoldPaid`.
- **Verdict:** Delete current. Reframe future version as actions.

### black_market
- **Old classification:** STUB / EMERGENT-CANDIDATE
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Header (`black_market.rs:8-10`) admits approximation; the output is a price-triggered random gold shower — delete. A real black market is `SmugglingDeal(buyer, seller, commodity, amount, price)` with a detection roll against an authority NPC's `InspectMarket` action. Heat, reputation, and discovery are derivable views over those events.
- **Required NPC actions:** `OfferIllicitGoods(commodity, amount, price)`, `PurchaseIllicit(offer_id)`, `InspectMarket(sid)`, `Bribe(target, amount)`.
- **Required derived views:** `heat(sid) = Σ IllicitTrade.amount × weight − Σ InspectMarket × decay`, `reputation(npc)`.
- **Required event types:** `IllicitTrade {buyer, seller, commodity, amount, price}`, `IllicitDetected {inspector, seller}`, `Bribe {from, to, amount}`.
- **Verdict:** Delete current; reframe as agent actions + derived heat.

### commodity_futures
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Current file is a tick-parity treasury oscillator — delete. Futures contracts are agent actions: `WriteFuture(commodity, amount, strike_price, expiry)`, `BuyFuture(future_id)`, `SettleFuture(future_id)` at expiry.
- **Required NPC actions:** `WriteFuture`, `BuyFuture`, `SettleFuture`.
- **Required derived views:** `open_futures(sid)`, `fair_value(future) = f(price_expectation)`.
- **Required event types:** `FutureWritten`, `FuturePurchased`, `FutureSettled {future, cash_settlement}`.
- **Verdict:** Delete.

### price_discovery
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT (with persistent agent memory)
- **Reframe:** v1 called `npc.price_knowledge: Vec<PriceReport>` irreducible. But this is *per-agent memory* that the agent naturally owns — it should not be written by a central system. Replace with: agents choose `ObserveLocalPrice(sid)` when at a settlement (emits `PriceObserved`) and `Gossip(target_npc, topic)` when paired up (emits `PriceGossiped`). Each agent's `price_knowledge` is a derivable fold over its own observed+gossiped events, or a direct agent-owned ring buffer updated by those actions. The stochastic pair selection currently in the system is pushed into agents themselves.
- **Required NPC actions:** `ObserveLocalPrice`, `Gossip(partner, topic)`, `RecordPriceObservation(c, sid, value)`.
- **Required derived views:** `agent.price_knowledge = fold over PriceObserved + PriceGossiped`.
- **Required event types:** `PriceObserved {observer, settlement, c, value, tick}`, `PriceGossiped {from, to, report}`.
- **Verdict:** Promote to agent actions; remove central scheduler.

### price_controls
- **Old classification:** EMERGENT-CANDIDATE (clamp) + ESSENTIAL (shortage/subsidy)
- **New classification:** EMERGENT
- **Reframe:** The ceiling/floor clamp is DERIVABLE (clip the derived price at read time). The stockpile drain on ceiling ("shortage") is the residents choosing to `HoardCommodity` or `BuyUpShortage` when the announced price is below equilibrium — an NPC action. The treasury subsidy on floor is the settlement leader's `PaySubsidy(commodity, amount)` — already fits LevyTax/PayUpkeep style. The act of imposing a control is the leader's `SetPriceCeiling(c, value)` or `SetPriceFloor(c, value)` action.
- **Required NPC actions:** `SetPriceCeiling(c, v)`, `SetPriceFloor(c, v)`, `RemovePriceControl(c)`, `PaySubsidy(c, amount)`, `HoardCommodity(c, amount)`.
- **Required derived views:** `effective_price(c, sid) = clamp(derived_price, ceiling, floor)`, `active_controls(sid)`.
- **Required event types:** `PriceCeilingSet`, `PriceFloorSet`, `SubsidyPaid`, `HoardingEvent`.
- **Verdict:** Actions + a read-time clamp.

### currency_debasement
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** `currency_debasement.rs` admits no faction currency state and uses a nonsense `(rid - sid).abs() <= 1` proximity heuristic. Delete. Real debasement is a faction leader's `DebaseCurrency(rate)` action, with an event `CurrencyDebased {faction, rate}` that an agent-side price-update rule applies on next observation.
- **Required NPC actions:** `DebaseCurrency(faction, rate)`, `MintCoinage(faction, amount)`.
- **Required derived views:** `currency_purity(faction) = initial × Π(1 − debasement_rate)`.
- **Required event types:** `CurrencyDebased`, `CoinageMinted`.
- **Verdict:** Delete. Reframe as a faction-leader decision.

### smuggling
- **Old classification:** DUPLICATIVE
- **New classification:** EMERGENT (duplicates the black_market + caravan action set)
- **Reframe:** `smuggling.rs` differs from `trade_goods` caravan only in constants (`smuggling.rs:57-119` vs `trade_goods.rs:57-119`). Both collapse into the same `RunCaravan` action, optionally flagged `illicit`, with detection rolls from the black-market vocabulary. Delete; goods moved via agent actions only.
- **Required NPC actions:** covered by `RunCaravan` + `OfferIllicitGoods` + `InspectMarket`.
- **Required derived views:** arbitrage + heat views above.
- **Required event types:** `CaravanDeparted/Arrived` + `IllicitTrade/Detected`.
- **Verdict:** Delete.

### economic_competition
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Treasury/price nudge on `treasury/avg_treasury` is pure function of state (`economic_competition.rs:98-119`) and the region-proximity block computes distance-from-origin, which is a bug. Delete. Real trade wars are `DeclareTradeWar(target_faction)`, `ImposeTariff(commodity, rate)`, `ImposeEmbargo(target_faction, commodity)` — agent actions that attach to caravan/trade events as gating or bonus.
- **Required NPC actions:** `DeclareTradeWar`, `ImposeTariff`, `ImposeEmbargo`, `LiftEmbargo`.
- **Required derived views:** `active_embargoes(sid)`, `tariff_rate(sid, c)`, `trade_war_state(a, b)`.
- **Required event types:** `TradeWarDeclared`, `TariffImposed`, `EmbargoImposed`, `TradePenaltyAssessed {trader, settlement, amount}`.
- **Verdict:** Delete. Reframe as faction-leader policy actions.

### bankruptcy_cascade
- **Old classification:** EMERGENT-CANDIDATE
- **New classification:** DERIVABLE-VIEW
- **Reframe:** The entire effect is `total_loss = Σ|deficit| × 0.15; healthy.treasury -= total_loss / n_healthy` — a pure function of current treasuries. Compute at read time or during the periodic settlement book-close pass. No system needed.
- **Required NPC actions:** none inherent; if contagion is desired via creditor relationships, use `loans`-style `Repay` / `Default` actions.
- **Required derived views:** `insolvent_settlements()`, `systemic_risk_multiplier() = f(insolvent_count/total)`, `correction_transfer(sid) = view`.
- **Required event types:** `SettlementInsolvent {sid}` (threshold trigger), `BankruptcyCorrection {sid, delta}` (if realized at book-close).
- **Verdict:** Becomes a query. No tick-scheduled mutation.

### corruption
- **Old classification:** STUB/DEAD
- **New classification:** EMERGENT (as-intended) + DEAD (as-written random rolls)
- **Reframe:** Random hashed treasury drain is noise. Real corruption is an official NPC choosing `Embezzle(amount)`, emitting `Embezzlement {actor, settlement, amount}`, with desertion modeled as NPC `Desert(target_settlement)` actions. Morale and inspection are agent-side.
- **Required NPC actions:** `Embezzle(amount)`, `Bribe(target, amount)`, `Investigate(suspect)`, `Desert(reason)`.
- **Required derived views:** `corruption_index(sid) = Σ Embezzlement.amount / treasury × decay`, `integrity(npc) = Σ uncovered_acts`.
- **Required event types:** `Embezzlement`, `InvestigationStarted`, `CorruptionUncovered`, `Desertion`.
- **Verdict:** Delete current. Reframe as agent choices.

### loans
- **Old classification:** STUB/DEAD / DUPLICATIVE
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Current implementation is a second tax (`loans.rs` flat 5% of gold) unrelated to `npc.debt`. Delete. Real loans are `RequestLoan(lender, amount, terms)`, `GrantLoan(request_id)`, `Repay(loan_id, amount)`, `Default(loan_id)`. The `advance_debt` logic from economy.rs already belongs here as `Repay`.
- **Required NPC actions:** `RequestLoan(amount, terms)`, `GrantLoan(request_id)`, `Repay(loan_id, amount)`, `Default(loan_id)`, `CallDebt(loan_id)`.
- **Required derived views:** `outstanding(npc) = Σ GrantLoan.amount − Σ Repay.amount`, `credit_history(npc) = Σ Repay − Σ Default`, `debt_capacity(npc) = income_rate × horizon`.
- **Required event types:** `LoanRequested`, `LoanGranted {lender, borrower, amount, terms}`, `LoanRepayment {loan, amount}`, `LoanDefault {loan}`.
- **Verdict:** Delete. Reframe. A loan is just a pair of events + a fold.

### insurance
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Current flat-premium / hp-triggered payout is a tax + reactive gold injection with no policy records. Delete. Real insurance is `BuyPolicy(underwriter, terms)`, `PayPremium(policy, amount)`, `FileClaim(policy, damage)`, `ApproveClaim(claim, amount)`.
- **Required NPC actions:** `BuyPolicy`, `PayPremium`, `FileClaim`, `ApproveClaim`, `DenyClaim`, `CancelPolicy`.
- **Required derived views:** `active_policies(npc)`, `claim_history(npc)`, `reserve(underwriter)`.
- **Required event types:** `PolicyIssued`, `PremiumPaid`, `ClaimFiled`, `ClaimApproved`, `ClaimPaid`, `ClaimDenied`.
- **Verdict:** Delete. Agent actions only.

### caravans
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT
- **Reframe:** v1 said the on-arrival dump + commission is path-dependent and not reconstructible. But the dump is literally "the trader NPC, upon arriving, chooses `DepositAtSettlement(commodity, amount)` and `TipCommission(settlement, amount)`." These are discrete agent actions at arrival; the tick-scheduled proximity check becomes an agent-side arrival trigger. Raid damage is already `Damage` (combat event) — agents choose `Attack(trader)`, physics applies the damage.
- **Required NPC actions:** `DepositAtSettlement(sid, commodity, amount)`, `TipCommission(sid, amount)`, `Attack(target)` (existing).
- **Required derived views:** `at_destination(npc) = dist(npc.pos, npc.dest.pos) < arrival_radius`.
- **Required event types:** `Deposit`, `GoldPaid`, `Damage` (existing).
- **Verdict:** Dissolved into agent actions on arrival.

### traveling_merchants
- **Old classification:** DUPLICATIVE
- **New classification:** EMERGENT (same vocabulary as caravans)
- **Reframe:** Identical pattern to `caravans` — merchant arrives somewhere, offloads inventory, takes payment. Merge both into the same `DepositAtSettlement` + `SellToSettlement` action set. Merchant also shares price reports: `Gossip(sid, price_report)`.
- **Required NPC actions:** covered by `DepositAtSettlement`, `SellToSettlement(sid, c, amount)`, `Gossip`.
- **Required derived views:** nearest settlement query.
- **Required event types:** covered above.
- **Verdict:** Delete; unified with caravans.

### mercenaries
- **Old classification:** STUB/DEAD
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Current: threat-gated treasury drain or random betrayal damage — delete. Real mercenaries are NPCs with `ContractForHire(employer, terms, duration)`, `Desert(employer)`, `Betray(employer)` actions. Wages are `PaySalary`. No central system.
- **Required NPC actions:** `ContractForHire`, `Desert`, `Betray`, `PaySalary` (existing).
- **Required derived views:** `loyalty(mercenary) = f(wages_paid, treatment_events)`.
- **Required event types:** `MercenaryHired`, `MercenaryDeserted`, `MercenaryBetrayed`.
- **Verdict:** Delete. Reframe as agent choices on existing contract vocabulary.

### supply
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT + ESSENTIAL (damage physics)
- **Reframe:** Traveler food drain is an agent `ConsumeTravelRations` choice each tick (or an automatic agent rule). Starvation damage on empty inventory is a physics rule on a `StarvationThreshold` event (just like food.rs — same damage kernel). The self-to-self `TransferCommodity` hack goes away — use `Consume`.
- **Required NPC actions:** `ConsumeTravelRations`, `Eat(source)` (shared with food).
- **Required derived views:** `travel_hunger(npc) = f(last_eat_tick, distance_traveled)`, `inv.food(npc)`.
- **Required event types:** `Consumption` (shared), `Starvation {npc, damage}` (ESSENTIAL damage application).
- **Verdict:** Merge into the shared `food` event/action set; one physics rule for starvation damage.

### supply_lines
- **Old classification:** ESSENTIAL (but overlaps caravans raid)
- **New classification:** EMERGENT (duplicates caravans raid + a patrol check)
- **Reframe:** Interdiction of trader cargo is `RaiderAttackCaravan(trader)` — a hostile NPC choice with a `CargoSeized` event. Patrol protection is just a spatial predicate that modulates the hostile's decision (or causes them to avoid/abort the attack). The `ConsumeCommodity { settlement_id: entity.id }` type pun (`supply_lines.rs:119`) is a bug that disappears when the action is `SeizeCargo(trader, c, amount)`.
- **Required NPC actions:** `RaidCaravan(trader)`, `SeizeCargo(trader, c, amount)`, `Patrol(area)` (patrol's only effect is its own presence in the spatial index — no write).
- **Required derived views:** `hostile_nearby(trader)`, `patrol_nearby(trader)` (spatial queries).
- **Required event types:** `RaidAttempted`, `CargoSeized {victim, attacker, c, amount}`, `RaidDeterredByPatrol`.
- **Verdict:** Delete; same action vocabulary as caravans + combat.

### equipment_durability
- **Old classification:** ESSENTIAL (advance) + STUB (compute)
- **New classification:** ESSENTIAL (physics: durability decay + despawn on break) — but narrower than v1
- **Reframe:** The compute half is confirmed DEAD (no emission). The advance half has a real physics role: `item.durability -= loss` is a physical-integration step (like HP decay), and `item.alive = false` on break is a despawn. These remain ESSENTIAL. HOWEVER, the stat-rollback on unequip (`owner.attack_damage -= eq`, etc.) is NOT physics — it is the recomputation of a derived view. NPC effective stats should be a DERIVABLE view over `equipped_items`, not a mutated mirror, so equip/unequip just changes the slot and stats re-derive on read. The decision to unequip a broken item is the agent's `UnequipItem(item)` choice triggered by a `durability==0` event.
- **Required NPC actions:** `UnequipItem(item_id)` (triggered by break event or on demand).
- **Required derived views:** `effective_stats(npc) = base + Σ equipped_items.modifiers × durability_frac`.
- **Required event types:** `DurabilityDecay {item, delta, cause}`, `ItemBroken {item}` (ESSENTIAL: emits despawn `alive=false`), `ItemUnequipped {npc, item, slot}`.
- **Verdict:** Keep the physics kernel (durability integration + break→despawn). Delete the stat-mirror write. Add an agent action for unequip.

### equipping
- **Old classification:** ESSENTIAL
- **New classification:** EMERGENT
- **Reframe:** The matching/upgrade logic is exactly the "NPC shops for better gear" agent choice. Each NPC chooses `ClaimItem(item_id, slot)` when they see better gear at their settlement — the central nested loop over items×slots×npcs becomes an agent utility. Emitting Equip/Unequip deltas becomes the two events `ItemEquipped {npc, item, slot}` / `ItemUnequipped`. No central assignment system needed.
- **Required NPC actions:** `ClaimItem(item_id, slot)`, `EquipItem(item_id)`, `UnequipItem(item_id)`, `DropItem(item_id)`.
- **Required derived views:** `unclaimed_items(sid)`, `effective_quality(item) = base_quality × durability_frac`, `upgrade_opportunity(npc) = best unclaimed − currently equipped`.
- **Required event types:** `ItemClaimed`, `ItemEquipped {npc, item, slot}`, `ItemUnequipped`.
- **Verdict:** Promote to agent actions.

### resource_nodes
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

### infrastructure
- **Old classification:** EMERGENT-CANDIDATE / DUPLICATIVE
- **New classification:** DEAD (as-written) → EMERGENT (as-intended)
- **Reframe:** Output is pure function of `(treasury, population)` with no infrastructure state (`infrastructure.rs` header admits this). Delete. Real infrastructure is `BuildRoad(from, to)`, `RepairBridge(id)`, `PayMaintenance(asset_id, amount)` — agent actions that spawn Infrastructure entities (same shape as buildings: `StartConstruction` + `WorkOnBuilding` + `PlaceTile`).
- **Required NPC actions:** `BuildRoad(from, to)`, `RepairBridge(id)`, `PayMaintenance(asset_id, amount)`.
- **Required derived views:** `road_network()`, `settlement_connectivity(a, b) = BFS over road graph`, `maintenance_deficit(asset)`.
- **Required event types:** `RoadBuilt`, `BridgeRepaired`, `MaintenancePaid`, `AssetDeteriorated`.
- **Verdict:** Delete. Share construction vocabulary with `buildings`.

---

## Reduction summary

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

## Required action vocabulary (deduplicated, batch-level)

### Taxes / settlement finance
- `LevyTax(rate)` — economy
- `PayTax(amount)` — economy
- `PaySalary(role_or_npc, amount)` — economy, food, work
- `PayUpkeep(target, amount)` — economy
- `PaySubsidy(commodity, amount)` — price_controls
- `PayMaintenance(asset_id, amount)` — infrastructure

### Production / consumption
- `Produce(commodity, amount)` — food, work, crafting
- `Consume(commodity, amount)` — food, work, crafting
- `Eat(food_source)` — food, supply
- `ConsumeTravelRations` — supply
- `ForgeItem(recipe)` — work (emits ESSENTIAL spawn)
- `HarvestTile(pos)` — work (emits ESSENTIAL voxel change)
- `HarvestNode(node_id, amount)` — work, gathering (consumes `remaining`)

### Movement / work state
- `GoToWorkplace(building_id)` — work
- `MoveTo(pos)` — gathering, work
- `BeginProduction(building_id)` — work
- `FinishProduction` — work
- `Deposit(target_id, commodity, amount)` / `DepositAtSettlement(sid, c, amount)` — work, caravans, traveling_merchants
- `PerformTimed(ticks)` — gathering
- `Wait(ticks)` — gathering

### Construction
- `StartConstruction(blueprint, pos)` — buildings (emits ESSENTIAL spawn)
- `PlaceBlueprint(pos, type)` — gathering, buildings
- `PlaceTile(pos, material)` — buildings (emits ESSENTIAL voxel change)
- `WorkOnBuilding(building_id, effort)` — buildings
- `ClaimWorkSlot(building_id)` — buildings
- `ClaimResidence(building_id)` — buildings
- `PayUpgrade(building_id, amount)` — buildings
- `BuildRoad(from, to)` / `RepairBridge(id)` — infrastructure

### Trade / markets
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

### Contracts / obligations
- `PostContract(spec)` — contracts
- `BidOnContract(contract_id, bid)` — contracts
- `AcceptBid(bid_id)` — contracts
- `WithdrawBid(bid_id)` — contracts
- `MarkContractFulfilled(contract_id)` — contracts
- `PayContract(provider, amount)` — contracts
- `CounterOffer(contract, terms)` — contract_negotiation (future)

### Auctions / futures (future EMERGENT)
- `PostAuction(item_id, reserve, deadline)`, `PlaceAuctionBid(auction_id, amount)`, `CloseAuction(auction_id)` — auction
- `WriteFuture(c, amount, strike, expiry)`, `BuyFuture(future_id)`, `SettleFuture(future_id)` — commodity_futures

### Lending / insurance (future EMERGENT)
- `RequestLoan(amount, terms)`, `GrantLoan(request_id)`, `Repay(loan_id, amount)`, `Default(loan_id)`, `CallDebt(loan_id)` — loans
- `BuyPolicy(underwriter, terms)`, `PayPremium(policy, amount)`, `FileClaim(policy, damage)`, `ApproveClaim(claim, amount)`, `DenyClaim`, `CancelPolicy` — insurance

### Faction-level policy (future EMERGENT)
- `DeclareTradeWar(target)`, `ImposeTariff(c, rate)`, `ImposeEmbargo(faction, c)`, `LiftEmbargo` — economic_competition
- `DebaseCurrency(faction, rate)`, `MintCoinage(faction, amount)` — currency_debasement

### Illicit (future EMERGENT)
- `OfferIllicitGoods(c, amount, price)`, `PurchaseIllicit(offer_id)`, `InspectMarket(sid)`, `Bribe(target, amount)` — black_market, smuggling
- `Embezzle(amount)`, `Investigate(suspect)`, `Desert(reason)` — corruption
- `ContractForHire(employer, terms, duration)`, `Desert(employer)`, `Betray(employer)` — mercenaries
- `RaidCaravan(trader)`, `SeizeCargo(trader, c, amount)`, `Patrol(area)` — supply_lines

### Items / equipment
- `ClaimItem(item_id, slot)` — equipping
- `EquipItem(item_id)` — equipping
- `UnequipItem(item_id)` — equipping, equipment_durability (break trigger)
- `DropItem(item_id)` — equipping

---

## Required event types

### Ledger / finance events
- `TaxPayment { from_npc, to_settlement, amount }`
- `Upkeep { settlement, target, amount }`
- `SubsidyPaid { settlement, c, amount }`
- `WageEarned { worker, commodity_or_gold, amount }`
- `GoldPaid { from, to, amount, reason }`
- `GoldReceived { recipient, amount, source }`
- `MoraleDelta { npc, amount, reason }`
- `CreditHistoryDelta { npc, delta }`
- `DebtRepayment { debtor, creditor, amount }`

### Production / consumption events
- `Production { producer, settlement, commodity, amount }`
- `Consumption { consumer, settlement, commodity, amount }` (recipe inputs)
- `Meal { eater, source, hunger_restore, heal }`
- `Starvation { npc, damage, morale_delta }` (ESSENTIAL damage application)
- `Deposit { from, to, commodity, amount }`

### Voxel / construction / world-mutation events (ESSENTIAL)
- `TilePlaced { placer, pos, material }`
- `TileHarvested { harvester, pos, material_before }`
- `ConstructionStarted { initiator, building_id, blueprint }` (ESSENTIAL: allocates new ID)
- `ConstructionProgress { worker, building, effort }`
- `BuildingUpgraded { building, tier }`
- `NavGridDirty { region, aabb }` (ESSENTIAL: spatial cache invalidation trigger)
- `WorkSlotClaimed { npc, building }`
- `ResidenceClaimed { npc, building }`

### Resource / item / entity lifecycle (ESSENTIAL spawn/despawn)
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

### Trade / market events
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

### Contract / auction / loan / insurance events
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

### Narrative / threshold triggers
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

## Required derived views (cached or lazy)

### Ledger folds
- `treasury(sid) = initial + Σ income − Σ spend` — over `TaxPayment`, `GoldPaid`, `Upkeep`, `SubsidyPaid`, `ContractPayment`, `BankruptcyCorrection`
- `gold(npc) = initial + Σ WageEarned + Σ received − Σ paid` — agent-side
- `outstanding_debt(npc) = Σ LoanGranted.amount − Σ LoanRepayment.amount`
- `credit_history(npc) = Σ Repay − Σ Default`
- `debt_capacity(npc) = income_rate × horizon`
- `income_rate(npc) = EMA over WageEarned`

### Stockpile / price / demand
- `stockpile(c, sid) = Σ Production − Σ Consumption − Σ Deposit_out + Σ Deposit_in`
- `price(c, sid) = base[c] / (1 + (stockpile[c]/pop)/50)` — pure function, canonical
- `effective_price(c, sid) = clamp(price(c,sid), floor, ceiling)` — when controls active
- `currency_purity(faction) = initial × Π(1 − debasement_rate)`
- `arbitrage_opportunities() = { (src,dst,c) | price(c,dst)/price(c,src) > threshold ∧ stockpile(c,src) > min }`

### Agent state
- `hunger(npc) = last_hunger + tick_delta − Σ Meal.hunger_restore`
- `travel_hunger(npc) = f(last_eat_tick, distance_traveled)`
- `inventory(npc) = Σ pickups − Σ drops/transfers`
- `effective_stats(npc) = base + Σ equipped_items.modifiers × durability_frac` — replaces the stat-mirror write in equipment_durability
- `price_beliefs(npc, c) = fold over PriceObserved + PriceGossiped (with recency window)`

### Social / relational
- `route_strength(a, b) = Σ CaravanArrived(a→b).profit × exp(-(now−tick)/τ)`
- `established_routes() = pairs where count(CaravanArrived) ≥ 3`
- `is_guild_member(npc) = home=sid ∧ trade+negotiation > 30`
- `has_guild(sid) = |guild_members| ≥ 3`
- `loyalty(mercenary) = f(wages_paid, treatment_events)`
- `reputation(npc) = f(uncovered_acts, public_deeds)`
- `heat(sid) = Σ IllicitTrade.amount × weight − Σ InspectMarket × decay`
- `corruption_index(sid) = Σ Embezzlement.amount / treasury × decay`

### Structural
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

## Truly essential (irreducible) set in this batch

Only three survive the strict test:

1. **resource_nodes** — `spawn_initial_resources` (entity ID allocation + pos), `tick_resource_regrowth` (`remaining += rate × dt`, clamped at capacity; `alive = false` on non-renewable depletion). Everything here is time-integration on a physical quantity or a lifecycle transition.

2. **equipment_durability** (advance kernel only) — durability integration (`durability -= loss` per tick of wear) plus break despawn (`alive = false` when `durability <= 0`). The stat-rollback is NOT physics and disappears into a derived view.

3. **buildings** (thin kernel only) — when a `StartConstruction` event arrives, allocate a new entity ID; when a `PlaceTile` event arrives, call `voxel_world.set_voxel`; when either happens inside a nav-tracked region, emit `NavGridDirty`. The 1014-line orchestration above that kernel is entirely EMERGENT (agent choices + reductions).

Everything else in the batch — all taxation, production scheduling, trade routing, contract lifecycles, market pricing, guild formation, equipping policy, bankruptcy propagation, and the entire "without X state" family — is either a policy that belongs on NPC agents (EMERGENT), a pure function of state (DERIVABLE-VIEW), or dead code (DEAD).

The consequence: **30 of 32 systems collapse into an action vocabulary + an event-fold index.** The ECS DSL needs maybe 60-80 agent action verbs and ~50 event types to cover this entire batch, with a dozen derived views defined as named queries. The remaining irreducible kernel is small enough to live inside a single ~200-line physics module covering resource integration, durability integration, spawn/despawn, voxel writes, and nav invalidation.
