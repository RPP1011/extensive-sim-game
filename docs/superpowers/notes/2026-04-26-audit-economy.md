# Economy Spec Audit (2026-04-26)

> Audit of `docs/spec/economy.md` (~1400 lines, 3 phases) against codebase.
> Companion spec references: `spec/ability.md`, `spec/language.md`, `spec/state.md`.
> Prior work: `npc_economy_plan.md` (2026-03-28, world-sim delta-arch branch — now separate from engine crate).

---

## Summary

| Phase | Title | Status | Deliverables |
|---|---|---|---|
| Phase 1 | Foundation (Dim 2, 9, 6, 1) | ❌ overall — thin substrate only | 0 of ~40 spec deliverables implemented in engine crate |
| Phase 2 | Behavioral depth (Dim 3, 5, 4) | ❌ overall — nothing started | 0 of ~25 spec deliverables |
| Phase 3 | Emergent macro (Dim 7, 8, 10) | ❌ overall — depends entirely on Phase 1+2 | 0 of ~15 spec deliverables |

**What does exist** is a thin economic substrate in the engine crate:
- `Inventory { gold: i32, commodities: [u16; 8] }` — 8 anonymous commodity slots, signed gold
- `EffectOp::TransferGold { amount: i32 }` (variant 5) and `EffectOp::ModifyStanding { delta: i16 }` (variant 6) — GPU-accelerated
- `QuestPosted / QuestAccepted / BidPlaced` events + `Quest` aggregate with `QuestCategory::Economic`
- `MacroAction::{PostQuest, AcceptQuest, Bid}` — policy surface
- `standing(a, b) → i32` materialized view — symmetric pair, K=8
- `AuctionId`, `SettlementId`, `ItemId` ID stubs (reserved but no backing structs)
- `Creditor { creditor, amount }` stub and `MentorLink` stub in `agent_types.rs`
- `GroupRole::Apprentice` in membership enum
- `can_trade: bool` in `Capabilities`

The prior world-sim implementation (`npc_economy_plan.md` commit `915dae2e`, 8-commodity production/settlement/gossip) lives in the **old world-sim branch** (src/world_sim/), which is not present in the current codebase tree (only `src/lib.rs` + `src/rendering.rs` remain). That work is not integrated into the engine crate and represents pre-spec architecture that the spec §21.3–21.4 intentionally reshapes.

---

## Top gaps (ranked by impact)

### Gap 1 — `CommodityRegistry` + named commodities (Phase 1, Dim 2 blocker)
The spec's entire production-chain model requires named commodities declared in `.sim` via `CommodityRegistry`. Currently `Inventory.commodities` is `[u16; 8]` — anonymous, fixed-size, no DSL declaration. All recipe, trade, market, and preference logic hangs on named commodity IDs. **No plan targets this.** Blocking for all three phases.

### Gap 2 — `EffectOp::Recipe` (variant 17) + `RecipeRegistry` (Phase 1, Dim 2 core)
The "recipes as abilities" anchor (§3) — the central architectural bet of the entire spec — is not implemented. No `RecipeEntry`, no `RecipeRegistry`, no `require_skill`/`require_tool` gate predicates, no quality formula system. Until this lands, the production graph, ingredient chains, component assembly, and all three runtime-composition stages are inexpressible. **Highest leverage primitive**: it unblocks Dim 3 (skills), Dim 5 (preferences scoring over recipe outputs), Dim 4 (counterfeit recipes), and the training curriculum.

### Gap 3 — `entity Obligation` + `EffectOp::{CreateObligation, DischargeObligation, DefaultObligation}` (Phase 1, Dim 1)
The contract/debt/futures/insurance/retainer model needs a `AggregatePool<Obligation>` + three new `EffectOp` variants. The `Creditor` stub exists but is opaque (`creditor: AgentId, amount: u32`). The full obligation lifecycle (creation, collateral, default cascade → property seizure, standing penalty) is not present. This is the foundation for Phase 3 banking, insurance pools, and credit crunches.

### Gap 4 — `entity Tool / Component / ResourceNode / Property` (Phase 1, multiple dims)
Five new entity types required by Phase 1 (`Tool`, `Component`, `ResourceNode`, `Property`, `Obligation`) have no backing structs. `ItemId` is a stub ID only. Without `ResourceNode` there is no depletion; without `Tool` there is no `require_tool` gate or wear; without `Property` there is no inheritance. Each requires both an `AggregatePool` and DSL `.sim` declaration support in the compiler. The DSL emitter (`dsl_compiler/src/emit_entity.rs`) currently only handles `Agent` subtypes.

### Gap 5 — `BeliefState` economic extensions + `cold_price_beliefs` / `cold_merchant_beliefs` / `cold_route_beliefs` (Phase 1/2)
`BeliefState` currently tracks `last_known_pos`, `last_known_hp`, `last_known_creature_type`, `confidence`. The spec extends it with `believed_wealth: f32, wealth_confidence: f32` (Phase 1) and adds three new per-agent cold maps (`cold_price_beliefs`, `cold_merchant_beliefs`, `cold_route_beliefs`). These are Phase 2 deliverables but phase-1 wealth-visibility also needs the extension. The gossip propagation for price beliefs reuses existing `Communicate`/`Announce` machinery — so the transport is free — but the storage struct doesn't exist.

---

## Pre-existing primitives that map to the spec

| Spec concept | Existing impl name/location | Status |
|---|---|---|
| Agent gold balance | `Inventory.gold: i32` in `agent_types.rs` | Thin — signed i32, GPU-compatible, no debt model |
| Commodity slots | `Inventory.commodities: [u16; 8]` | Thin — fixed 8 anonymous slots, not `CommodityRegistry` |
| Gold transfer | `EffectOp::TransferGold { amount: i32 }` in `ability/program.rs` | ✅ Works, GPU-accelerated |
| Standing (bilateral trust) | `EffectOp::ModifyStanding` + `view standing(a,b) → i32` | ✅ Works, GPU-resident |
| Quest posting/auction | `MacroAction::{PostQuest, AcceptQuest, Bid}` + `Quest` aggregate | ⚠️ Frame exists; no bid-resolution logic, no payment terms |
| Quest category (Economic) | `QuestCategory::Economic` enum variant | ✅ Declared |
| BidPlaced event | `BidPlaced { bidder, auction_id, amount }` | ✅ Emitted; `amount: f32` only — no commodity/item payment |
| Group kinds | `Group { kind_tag: u32, members, leader }` | ⚠️ Generic; `kind_tag` not typed (spec needs `Guild`, `Caravan` etc.) |
| Group roles | `GroupRole::{Member, Officer, Leader, Founder, Apprentice, Outcast}` | ✅ `Apprentice` is present |
| Creditor stub | `Creditor { creditor: AgentId, amount: u32 }` in `agent_types.rs` | Stub only; `Obligation` entity doesn't exist |
| Mentor link stub | `MentorLink { mentor: AgentId, discipline: u8 }` in `agent_types.rs` | Stub only; no `Apprenticeship` obligation |
| Chronicle events | `ChronicleEntry { template_id, agent, target }` | ⚠️ Shell; template library for economic events not authored |
| Item IDs | `ItemId`, `AuctionId`, `SettlementId` ID newtypes | Reserved stubs only |
| `can_trade` capability | `Capabilities.can_trade: bool` on `Human = true` | ✅ Flag present |
| Price beliefs (old arch) | Per-NPC `price_beliefs` in old world-sim (`src/world_sim/`, not in codebase) | ❌ Old arch; spec §21.3 replaces with `cold_price_beliefs` |
| 8-commodity system (old arch) | `Inventory.commodities: [u16; 8]` shape from old world-sim | ⚠️ Shape preserved, semantics gutted — commodity names/types not carried |

---

## Per-phase findings

### Phase 1 — Foundation (Dim 2, 9, 6, 1)

**Overall: ❌ — engine crate has substrate only; spec deliverables not implemented.**

#### Dim 2 — Supply-chain depth (§4)

- **RecipeEntry / RecipeRegistry** ❌ — No struct, no DSL syntax, no registry. The `// GENERATED` schema hash in `schema.rs` doesn't include recipe types. The DSL IR (`ir.rs`) has no `EffectOp::Recipe` variant.
- **EffectOp::Recipe (variant 17)** ❌ — `EffectOp` in `ability/program.rs` has 8 variants (0–7); economic variants 17–26 absent. Schema hash locked at variants 0–7.
- **EffectOp::WearTool (variant 18)** ❌ — absent.
- **Quality grades `[0.0, 1.0]`** ❌ — No quality field on any item/component type.
- **entity Tool : Item** ❌ — `ItemId` stub only; no `Tool` struct.
- **entity Component : Item** ❌ — absent.
- **entity ResourceNode** ❌ — absent.
- **CommodityRegistry** ❌ — `commodities: [u16; 8]` is anonymous slots, not a named registry.
- **require_skill / require_tool gate predicates** ❌ — `Gate` struct in `ability/program.rs` has no `require_skill` or `require_tool` fields.
- **Perishables / decay_rate** ❌ — `Inventory` has no `created_at_tick` per stack; no decay view.
- **Production DAG / price-feedback** ❌ — No `producers_of(c)` / `consumers_of(c)` adjacency; no `commodity_shortage` view.
- **Component-based assembly** ❌ — absent.
- **Events: RecipeCast, RecipeCompleted, ToolWornOut, ProductionStalled, ResourceDepleted, ResourceRegenerated** ❌ — none in `events.sim`.
- **AbilityHint::Economic** ❌ — `AbilityHint` has {Damage, Defense, CrowdControl, Utility}; Economic/Financial variants absent.

#### Dim 9 — Geography & transport (§5)

- **Per-commodity weight** ❌ — No weight field in commodity catalog.
- **inventory_weight_capacity** ❌ — No capacity field on agent.
- **terrain speed_modifier** ❌ — `TerrainQuery` extension not present.
- **RouteRegistry + RouteData** ❌ — No struct; `SettlementId` stub exists but no settlement-cluster view.
- **Caravan as Group{kind: Caravan}** ❌ — `Group.kind_tag` is untyped `u32`; no Caravan semantic.
- **EffectOp::EstablishRoute (variant 24)** ❌ — absent.
- **EffectOp::JoinCaravan (variant 25)** ❌ — absent.
- **Gate predicates: require_terrain, require_in_settlement_cluster, require_on_route** ❌ — absent.
- **Passive triggers: on_arrived_at, on_route_traversed, on_bandit_encounter** ❌ — passive trigger system not present in DSL.
- **Settlement cluster materialized view** ❌ — `SettlementId` stub only; spec §16 calls for a `cluster_agents_and_structures()` materialized view. Not in `views.sim`.
- **Bandit creature_type** ❌ — `entities.sim` has {Human, Wolf, Deer, Dragon} only.

#### Dim 6 — Wealth & property (§6)

- **entity Property { kind, owner, location }** ❌ — absent.
- **PropertyKindRegistry** ❌ — absent.
- **EffectOp::TransferProperty (variant 19)** ❌ — absent.
- **EffectOp::ForcibleTransfer (variant 20)** ❌ — absent; `ContestKind` enum absent.
- **Will per-agent side state** ❌ — absent (non-replayable side state slot concept noted in spec).
- **Inheritance cascade rule (on AgentDied → will read)** ❌ — `AgentDied` cascade exists in `physics.sim` (kill_agent) but no will/intestacy handler.
- **BeliefState.believed_wealth / wealth_confidence** ❌ — `BeliefState` has {pos, hp, max_hp, creature_type, tick, confidence} only.
- **Default reactive passives: ReactToBeingDemanded, WitnessAllyRobbed** ❌ — passive ability blocks not in DSL.
- **power_estimate materialized view** ❌ — absent from `views.sim`.
- **authority_in materialized view** ❌ — absent.
- **Events: EffectPropertyTransferred, EffectStolen, EffectAttemptedTheft, EffectBequest, EffectInheritanceResolved** ❌ — none in `events.sim`.

#### Dim 1 — Contracts & obligations (§7)

- **entity Obligation { kind, parties, terms, status, created_tick }** ❌ — `Creditor` stub holds only `(creditor: AgentId, amount: u32)`.
- **ObligationKind enum** ❌ — {Debt, Future, Insurance, Retainer, Service} absent.
- **EffectOp::CreateObligation (variant 21)** ❌ — absent.
- **EffectOp::DischargeObligation (variant 22)** ❌ — absent.
- **EffectOp::DefaultObligation (variant 23)** ❌ — absent.
- **Per-agent obligation indices** ❌ — `obligations_owed_by_me` / `obligations_owed_to_me` not in `SimState`.
- **Default cascade: due_tick → check gold → fire ContractDefaulted** ❌ — no due_tick handler in `physics.sim`.
- **Risk-adjusted bid pricing for loans** ❌ — `BidPlaced` carries `amount: f32` only; no obligation terms.
- **AbilityHint::Financial** ❌ — absent.
- **Events: ObligationCreated, ObligationDischarged, ObligationDefaulted, ContractInsuranceClaimed** ❌ — none in `events.sim`.

**Phase 1 notes:**
- The `Quest` + `Bid` + `MacroAction::PostQuest/AcceptQuest/Bid` frame is the closest running piece: posting a quest with `category: Economic` and bids are emitted. However, quest resolution is not implemented — `Quest.acceptors` is populated but there is no accept-then-pay-then-discharge cycle.
- `EffectOp::TransferGold` and `EffectOp::ModifyStanding` are the only economic effect primitives that actually run.
- The shape of `Inventory.commodities: [u16; 8]` matches the "8 commodities" legacy system from `npc_economy_plan.md`, but the commodity names/meanings are stripped; the spec requires named `CommodityId`s via a registry.

---

### Phase 2 — Behavioral depth (Dim 3, 5, 4)

**Overall: ❌ — all Phase 2 deliverables depend on Phase 1 primitives that don't exist yet.**

#### Dim 3 — Labor & skills (§8)

- **Agent.skills: SortedVec<(RecipeId, f32), 16>** ❌ — `RecipeId` doesn't exist yet; skills field absent.
- **RecipeTransferMatrix (compile-time)** ❌ — absent.
- **Skill-update cascade rules (practice, observation, book, decay)** ❌ — no `SkillRaised` event, no cascade rule.
- **Apprenticeship as (InviteToGroup{Apprentice} + Obligation{Service})** ❌ — `GroupRole::Apprentice` is present but the composite pattern with an obligation is not wired.
- **Wage formation via PostQuest{Service, min_payment}** ❌ — no `Service` `QuestCategory` variant (only Economic exists as a category, no sub-kind); no `min_payment` field in `Quest`.
- **Events: SkillRaised, ApprenticeshipBegun, ApprenticeshipCompleted, MassQuit** ❌ — none in `events.sim`.

#### Dim 5 — Heterogeneous preferences (§9)

- **AgentPreferences struct** ❌ — absent.
- **EthicalConstraint enum** ❌ — absent.
- **commodity_affinity, status_threshold, merchant_loyalties, ethical_bans, addictive_attachments** ❌ — no per-agent preference storage.
- **Mask gate: not_banned(commodity, source)** ❌ — absent.
- **Events: PreferenceShifted, AddictionDeveloped, WithdrawalOnset** ❌ — none in `events.sim`.

#### Dim 4 — Information asymmetry & trust (§10)

- **cold_price_beliefs, cold_merchant_beliefs, cold_route_beliefs, cold_quest_beliefs per-agent** ❌ — `BeliefState` is per-agent/target combat belief only; no economic belief maps.
- **apparent_* vs actual_* item attributes** ❌ — no item attribute system at all.
- **appraisal as SkillId** ❌ — no SkillId type.
- **Counterfeiting recipe family** ❌ — depends on Recipe + apparent attributes.
- **Condition atoms: believes_genuine, apparent_quality, actual_quality, merchant_belief.fair_dealer** ❌ — absent.

---

### Phase 3 — Emergent macro (Dim 7, 8, 10)

**Overall: ❌ — entirely dependent on Phase 1+2. No Phase 3 primitives present.**

#### Dim 7 — Market structure (§11)

- The spec claims "mostly emergent" — all Dim 7 phenomena (cartels, monopolies, sumptuary laws) emerge from Phase 1+2 primitives.
- **Passive trigger: on_observe_item(commodity_pred, target_pred, range)** ❌ — the only new Phase 3 primitive; passive trigger system not in DSL.
- **Verification harness scenarios** ❌ — none authored.
- **Condition atoms: surplus_from_one_source, dominant_supplier** ❌ — no commodity-level supply views.

#### Dim 8 — Financial instruments (§12)

- **Obligation.transferable: bool field** ❌ — `Obligation` itself doesn't exist.
- **EffectOp::TransferObligation (variant 26)** ❌ — absent.
- Banking/equity/insurance emergence: blocked by Phase 1 obligations and Phase 2 preferences.
- **Verification harness: bank formation, mutual-aid pool, futures speculation** ❌ — none.

#### Dim 10 — Macro dynamics (§13)

- Spec claims "entirely emergent" — inflation, boom/bust, credit crunches, resource shocks, war economy, famine cascade are all emergent from Phase 1+2 primitives.
- **Verification harness scenarios** ❌ — none.
- **Chronicle templates: IronHillsDepleted, BakersCartelCollapsed, BankFoundationByX, etc.** ❌ — `ChronicleEntry` struct exists but template library for economic events is not authored. `chronicle.rs::templates` has no economic template IDs.
- **Macro metrics dashboard (price drift, default rate, gini, migration rate)** ❌ — absent.

#### Runtime ability composition (§14)

- **Slot-fill generation model** ❌ — no generation-trigger cascade rules, no slot-fill schema, no grammar/economic/novelty gates.
- **Runtime `AbilityRegistry` partition** ❌ — `AbilityRegistry` exists for authored abilities only; no runtime-generated partition.
- **GOAP planner integration** ❌ — not referenced in any current plan.

#### Settlement cluster view (§16)

- **`settlement_clusters()` materialized view** ❌ — `SettlementId` stub only. `views.sim` has no clustering view. The spec replaces the `Settlement` entity with this view.
- **Settlement-conditional atoms: at_settlement, in_any_settlement, settlement_size_above** ❌ — absent from scoring/condition primitives.

---

## Cross-cutting observations

### 1. Engine crate is the right target; world-sim branch is archaeology

The old `npc_economy_plan.md` work (8-commodity system, settlement founding, price beliefs, merchant trade) was implemented in `src/world_sim/` which is not present in the current codebase. The spec §21.3–21.4 explicitly reshapes that design: settlements become emergent views, production becomes recipe-as-ability, wages become Service auctions. The engine crate is the correct target; nothing from the old branch should be ported wholesale — only the `Inventory.[gold, commodities]` shape carries forward.

### 2. DSL compiler is the rate-limiting dependency

Nearly every Phase 1 deliverable requires DSL-level support: new entity declarations (`Tool`, `Component`, `ResourceNode`, `Property`, `Obligation`), new `EffectOp` variants, new gate predicates, new condition atoms, new passive triggers, and new events. The `dsl_compiler` currently supports 4 entity subtypes (`Human`, `Wolf`, `Deer`, `Dragon`) and 8 `EffectOp` variants. Each new construct requires both AST nodes in `ast.rs`, resolution in `resolve.rs`, and Rust+WGSL emit paths. The compiler work is a prerequisite multiplier for all of Phase 1.

### 3. Three phases are correctly sequenced; plan decomposition should mirror them

Phase 1 dimensions have few circular dependencies (Dim 2 → Dim 9, Dim 6 mostly independent, Dim 1 depends on Dim 6's `Property` for collateral). Recommended plan structure:
- **Plan A** (Phase 1a): `CommodityRegistry` + `EffectOp::Recipe` + `RecipeRegistry` + named commodity slots — unblocks all downstream dims
- **Plan B** (Phase 1b): `entity Tool/Component/ResourceNode` + `EffectOp::WearTool` + quality + depletion
- **Plan C** (Phase 1c): `entity Property` + `entity Obligation` + `EffectOp::{TransferProperty, CreateObligation, DischargeObligation, DefaultObligation}` + inheritance cascade
- **Plan D** (Phase 1d): Geography: terrain weight, `RouteRegistry`, caravan Group kind, settlement-cluster view, bandit entity
- **Plans E–G** (Phase 2): Skills (depends on RecipeId from A), Preferences (depends on CommodityId from A), Information beliefs (depends on price beliefs schema)
- **Plans H–I** (Phase 3): Verification harnesses + Chronicle templates (purely emergent; no new primitives except `on_observe_item` passive trigger and `TransferObligation`)

### 4. The Quest/Bid frame is a partial scaffold — not yet wired

`QuestPosted → BidPlaced → QuestAccepted` events exist and fire through the cascade. But `Quest` carries no payment terms (no commodity amounts, no obligation terms), and `QuestAccepted` fires no `EffectGoldTransfer`. The economic action loop (post → bid with terms → accept → payment → discharge) is structurally sketched but semantically empty. Plan A should include wiring payment terms into `Quest`/`BidPlaced` — this is lower effort than any new entity and unlocks the Service auction model (wage formation, §8.3).

### 5. `EffectOp` budget constraint

`EffectOp` is `#[repr(u8)]` with 8 current variants; the spec adds 10 more (variants 17–26). The budget test `effect_op_size_budget()` asserts ≤ 16 bytes per variant. Each new variant needs to fit that budget. `Recipe { recipe: RecipeId, target_tool_sel: ToolSel }` at two `u32`s = 8 bytes — fits. `ForcibleTransfer { ..., contest_kind, detection_threshold }` at ~16 bytes — tight. The schema hash in `schema.rs` is hand-maintained and must be updated on every addition.
