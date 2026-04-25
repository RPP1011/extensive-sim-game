# Economic Depth Design — Beyond X4

> **Status:** Design spec (2026-04-24). Companion to `spec/ability.md`
> (the action grammar). This doc owns the *system* design — supply chains,
> contracts, labor, market structure, macro dynamics. The ability DSL spec owns
> the verbs and IR. Cross-references are explicit throughout.
>
> Companion to: `spec/ability.md` (ability DSL grammar +
> IR); `docs/spec/language.md` (world-sim DSL + auctions/quests); existing memory
> doc `npc_economy_plan.md` (initial NPC economy state, 2026-03-28).

---

## §1 Scope & non-goals

### 1.1 Scope

Design a deeply emergent NPC-autonomous economy that meaningfully exceeds X4's
depth across ten dimensions, organized into three implementation phases. The
spec owns:

- The dimensional analysis: ten depth axes, each with sub-topic decisions.
- The phase ordering and per-phase MVP criterion.
- The integration with the ability DSL (recipes as abilities; economic acts
  composable from `EffectOp` primitives).
- The status markers per construct: `runs-today` / `planned` / `reserved`.
- The IR additions cross-referenced into the ability DSL spec.
- The verification + Chronicle surface for emergent macro phenomena.
- The training curriculum at a *referenced* (not designed) level.

### 1.2 Non-goals

- **Action grammar.** Owned by `spec/ability.md`.
- **Cascade rules / event handlers.** Owned by `assets/sim/physics.sim` once
  emitted.
- **Training pipeline implementation.** This doc references the curriculum
  shape (§14) but does not specify gradients, datasets, or hyperparameters.
- **Hand-authored balance numbers.** Tuning happens via scenarios.

### 1.3 Design principles

1. **Emergent if possible, primitive only when forced.** A new entity / event /
   `EffectOp` requires demonstrating that the phenomenon cannot arise from
   existing primitives. Most Phase 3 macro behavior is pure emergence.
2. **Recipes are abilities.** Economic actions compose from the ability DSL's
   `EffectOp` vocabulary (§22 of the ability DSL spec). No parallel action
   system.
3. **Heterogeneity over homogeneity.** Two same-class agents must be able to
   diverge on preferences, skills, beliefs. No global behavior.
4. **Beliefs over ground truth.** Agents act on what they believe; deception,
   reputation, and information cost are first-class.
5. **Multi-agent fixed points.** Macro phenomena (cartels, banking, inflation)
   emerge from local choices, are not designed-in.
6. **Status markers.** Every construct is `runs-today` / `planned` / `reserved`.

### 1.4 Player-facing target

DF-grade emergent narrative for a pure NPC simulation. No player intervention.
Each economic decision should be capable of producing Chronicle entries.

### 1.5 Fidelity

Single fidelity tier — full simulation for every agent. Multi-fidelity
collapse (the `npc_economy_plan.md` Background/Low/Medium/High proposal) is
out of scope; this spec assumes the simulation runs at the equivalent of "High"
everywhere.

---

## §2 Phase ordering & dependency graph

Three implementation phases; each phase's MVP criterion gates the next.

**Phase 1 — Foundation.** Substrate dimensions everything else references.
- Dim 2 — Supply-chain depth (multi-stage production, capital goods, depletion, perishables).
- Dim 9 — Geography & transport.
- Dim 6 — Wealth & property.
- Dim 1 — Contracts & obligations.

*Phase 1 MVP:* an NPC can own tools + stockpile commodities + trade
cross-region + inherit wealth + enter binding contracts. The action surface
makes Phase 2 expressible.

**Phase 2 — Behavioral depth.** Heterogeneous-agent dimensions.
- Dim 3 — Labor & skills.
- Dim 5 — Heterogeneous preferences.
- Dim 4 — Information asymmetry & trust.

*Phase 2 MVP:* two equally-wealthy NPCs make different purchases based on
class/skill/preferences; a "legendary blacksmith" emerges from skill +
reputation; a counterfeit economy exists alongside the legitimate one.

**Phase 3 — Emergent macro.** Cartels, banking, inflation, war economy. All
emergent from Phase 1+2 primitives; this phase is mostly verification.
- Dim 7 — Market structure.
- Dim 8 — Financial instruments.
- Dim 10 — Macro dynamics.

*Phase 3 MVP:* a cartel emerges; a central bank emerges; a mine collapse
triples local prices and induces labor migration; all from local decisions
without explicit scripting.

---

## §3 The recipe-as-ability anchor

Recipes are abilities. Every production / contract / trade act is an
`AbilityProgram` composed of `EffectOp` primitives via the ability DSL.

```
ability ForgeSword {
    target: self
    cast: 400ms
    hint: economic
    require_skill blacksmithing >= 0.4
    require_tool forge

    consume iron_ingot 2
    consume leather 1
    produce sword quality(0.4 * inputs_quality + 0.5 * skill + 0.1 * tool_quality)

    wear_tool forge 0.05
}
```

Compiles to:

```
AbilityProgram {
    delivery: Instant
    area:     SingleTarget { range: 0.0 }
    gate: {
        cooldown_ticks:  0,
        cast_ticks:      4,
        require_skill:   Some((blacksmithing, 0.4)),
        require_tool:    Some(ToolKind::Forge),
    }
    effects: [
        Recipe { recipe: 42, target_tool_sel: ToolSel::OwnedNearest },
        WearTool { tool_kind: Forge, amount: 0.05 },
    ]
    hint: Some(Economic)
    tags: [(Utility, 1.0)]
}
```

The `Recipe` `EffectOp` (variant 17, see §10) carries inputs, outputs, and
the quality formula; runtime resolution looks up the recipe registry and
applies the canonical effect (consume listed inputs, produce listed outputs at
computed quality, fire `RecipeCompleted`).

Recipes can be:
- **Authored** at compile time in `.ability` files.
- **Generated at runtime** by the ability composition model (§13–§14). The
  agent's invented `WeavePaperArmor` becomes a new entry in the registry.
- **Learned** by other agents via memory / testimony / apprenticeship.

The grammar surface, IR shape, and registry mechanism are shared across all
three sources; only the source of the entry differs.

---

## §4 Phase 1: Dim 2 — Supply-chain depth

Multi-stage production with quality, capital goods, depletion, perishables,
and component-based assembly.

### 4.1 Recipe shape (2A)

**Decision:** single `EffectOp::Recipe` variant + recipe registry indirection.

A `Recipe` carries inputs, outputs, quality formula, skill, tool, and
duration in one entry. Mirrors how voxel ops use a mask registry. Keeps
`MAX_EFFECTS_PER_PROGRAM = 4` budget honest — a single ability can pair a
recipe with combat side-effects ("Brew of Healing" produces a potion *and*
heals self).

```rust
struct RecipeEntry {
    inputs:         SmallVec<[(CommodityId, u16); 4]>,
    outputs:        SmallVec<[(CommodityOrComponent, u16); 4]>,
    duration_ticks: u32,
    skill_id:       SkillId,
    tool_kind:      Option<ToolKindId>,
    quality_formula: QualityFormulaId,
}
```

`RecipeRegistry` is partitioned: `[1..N_authored]` for compile-time entries
(part of schema hash), `[N_authored+1..MAX]` for runtime-generated (recorded
in the trace).

### 4.2 Quality grades (2B)

**Decision:** continuous `f32` quality `[0.0, 1.0]` per produced item.
Per-recipe quality formula over inputs / skill / tool; default formula:
`0.4 * inputs_quality.avg + 0.5 * skill + 0.1 * tool_quality`.

Quality formulas are expression trees in the same arithmetic sublanguage
as ability DSL §15.3 (compile-time evaluation, bounded operators). The
runtime-composition model (§13) emits formulas by selecting from a small
set of named templates and parameterizing weights.

### 4.3 Capital goods (2C)

**Decision:** tools with continuous quality + depreciation + repair.

- New entity: `entity Tool : Item { kind: ToolKindId, quality: f32, wear: f32, durability: f32, owner: OwnerRef }`.
- New gate predicate: `require_tool <ToolKindId>` checked at cast attempt.
- New `EffectOp::WearTool { tool_kind, amount }` increments wear. At `wear >= durability` the tool is broken; recipes gating on it fail.
- Repair is itself a recipe: `RepairForge { consume: [iron_ingot 1, stone 2], target_tool: forge, reduce_wear: 0.5, require_skill: blacksmithing }`.

Tools are tradable items; high-quality tools command premiums. Tool ownership
is per-`OwnerRef` (Agent / Group / Settlement-as-emergent-cluster).

### 4.4 Resource depletion (2D)

**Decision:** depletable nodes + per-node regeneration rate.

- New entity: `entity ResourceNode : { kind: ResourceKind, location: Vec3, remaining: f32, capacity: f32, regen_rate: f32 }`.
- Harvesting decrements `remaining`; per-tick `regen` cascade rule replenishes by `regen_rate` up to `capacity`.
- Renewable (wood, herbs, fish): `regen_rate > 0`. Non-renewable (iron, crystal): `regen_rate = 0`.
- New events: `ResourceDepleted { node_id, tick }`, `ResourceRegenerated { node_id, amount, tick }`.

Discovery (initially-hidden deposits) deferred — no fog-of-resources system
in current scope.

### 4.5 Perishables & spoilage (2E)

**Decision:** decay + storage modifiers + transit spoilage as a single lazy view.

- Each commodity stack carries `created_at_tick: u32`.
- Effective quantity is a lazy view: `effective(stack) = stack.amount * decay_curve(now - stack.created_at_tick, storage_modifier)`.
- Storage entities (granaries, salt-cellars) carry `decay_modifier: f32 < 1.0`; commodity stacks inside inherit the modifier.
- Transit decay: an agent carrying a stack uses an open-air modifier (typically `1.0` = no protection).
- Material catalog (ability DSL §17) gains a `decay_rate: f32` property.

### 4.6 Production graph + price-feedback (2F)

**Decision:** compiled DAG + price-belief-driven recipe selection.

- Recipe registry's input/output edges form a DAG at world init. Adjacency
  lists `producers_of(c)` and `consumers_of(c)` are stored.
- Shortage detection: per-tile view `commodity_shortage(c)` becomes true
  when `local_stockpile < threshold`. Downstream consumer abilities see
  `shortage_blocking(c)` as a condition atom.
- Price feedback: agent's existing `price_beliefs[c]` couple back into
  recipe scoring. Scoring computes expected profit `output_value - input_cost`
  using current local beliefs; the agent's policy picks the highest-scoring
  applicable recipe.
- New events: `ProductionStalled { recipe, reason: ShortageOf(c) }`,
  `RecipeCast { recipe, agent, tick }`, `RecipeCompleted { recipe, agent, output_quality, tick }`.
- New passives can trigger on these events (`on_production_stalled`).

### 4.7 Component-based assembly (2G)

**Decision:** quality-bearing component items + assembly recipes.

- New entity: `entity Component : Item { kind: ComponentKindId, quality: f32, wear: f32, durability: f32 }`.
- `ComponentKindId` declared in `.sim`, sibling to `ToolKindId` (e.g.,
  `sword_blade`, `sword_hilt`, `forge_anvil`, `forge_furnace`).
- Component-producing recipes output components rather than finished items.
  Assembly recipes consume components and produce finished items.
- Per-recipe component cap: 8 components per assembly recipe (matches
  `MAX_ABILITIES = 8` precedent).
- Quality propagation: assembly's quality formula reads constituent component
  quality. A finished sword with a flawed hilt has `quality_min < quality_avg`.
- Identity: components carry per-instance state. A sword references its blade
  and hilt by `ComponentId`; lineage stories work ("this blade has been on
  five swords").
- Component decay: same wear/durability rules as tools.
- Salvage / disassembly: an inverse-recipe consumes a finished item and
  produces some fraction of its components at reduced quality.

Tools are themselves assemblies. Building a forge is an assembly recipe over
`forge_anvil + forge_furnace + forge_bellows`. Specialist component-makers
emerge naturally as the supply graph rewards specialization.

### 4.8 IR additions for Dim 2

| Construct | Type | Status |
|---|---|---|
| `EffectOp::Recipe { recipe, target_tool_sel }` (variant 17) | new | planned |
| `EffectOp::WearTool { tool_kind, amount }` (variant 18) | new | planned |
| `Gate.require_skill: Option<(SkillId, f32)>` | extension | planned |
| `Gate.require_tool: Option<ToolKindId>` | extension | planned |
| `entity Tool : Item { kind, quality, wear, durability, owner }` | new | planned |
| `entity Component : Item { kind, quality, wear, durability }` | new | planned |
| `entity ResourceNode { kind, location, remaining, capacity, regen_rate }` | new | planned |
| `RecipeRegistry`, `ToolKindRegistry`, `CommodityRegistry`, `SkillRegistry`, `ComponentKindRegistry` | new | planned |
| Material `decay_rate` property | extension | planned |
| Events: `RecipeCast`, `RecipeCompleted`, `ToolWornOut`, `ProductionStalled`, `ResourceDepleted`, `ResourceRegenerated` | new | planned |
| Condition atoms: `has_inputs(recipe)`, `has_tool(kind)`, `has_skill(skill, threshold)`, `shortage_blocking(commodity)`, `commodity_demand_above(c, threshold)`, `component_quality_below(item, kind, threshold)` | new | planned |

---

## §5 Phase 1: Dim 9 — Geography & transport

Spatial economy with regional price gradients, transport friction, route
formation, and the bandit risk that emerges once goods move.

### 5.1 Regional price formation (9A)

**Decision:** per-settlement-cluster local prices + gossip-propagated neighbor
beliefs.

- Per-agent `price_beliefs[commodity]` already exists. The "regional price"
  is the population-distributed view over local agent beliefs — no separate
  regional aggregator.
- Gossip: agents traveling between clusters carry price information. On
  arrival, they update local beliefs of observers. Existing `Communicate` /
  `Announce` machinery covers it.
- Arbitrage emerges: an agent who learns of a price gradient between two
  clusters has a profitable `BuyHere → MoveTo → SellThere` plan available
  in their action scoring.

### 5.2 Transport cost (9B)

**Decision:** time + carrying weight + terrain-speed modifier + transit decay.

- Per-commodity `weight: f32` declared in `.sim`.
- Per-agent `inventory_weight_capacity: f32` (typically tied to strength stat).
- Per-terrain `speed_modifier: f32` extends `TerrainQuery` (forest 0.7,
  paved-road 1.5, mountain 0.4).
- Transit decay reuses 2E.δ — perishables decay during movement.

No new abilities or `EffectOp`s; existing `MoveToward` micro carries the
modified speed and weight gating. Mask layer reads `inventory_weight_below(cap)`.

### 5.3 Trade routes (9C)

**Decision:** emergent routes from recurring traversals.

- New world-level `RouteRegistry: Vec<RouteData>`. Auto-populates: when a
  per-pair traversal-count crosses threshold, a `RouteData` entry is created.
- `RouteData` carries `endpoints, avg_traversal_ticks, bandit_encounters,
  safety_belief, last_used_tick`. Updates per traversal.
- Routes that go unused are GC'd after N ticks.
- Authors can pre-seed routes by writing `RouteData` rows in `.sim` (matches
  the "prebuilt road network" scenario use case without forcing all routes
  to be authored).

### 5.4 Caravans (9D)

**Decision:** caravans as `Group{kind: Caravan}` with shared inventory + profit
splits + service-auction escorts.

- Caravan formation is `InviteToGroup{kind: Caravan}` — existing macro action.
- Shared inventory: caravan-level `caravan_inventory: Inventory` shared by
  members during the caravan's lifetime. On termination (arrival /
  dissolution), profits split per pre-agreed `profit_split: SmallVec<[(AgentId, f32); 8]>`.
- Escorts: caravan posts `PostQuest{kind: Service, terms: Guard(duration=route_estimate)}`
  via existing auction system. Guards earn `Payment::ServicePledge`.

### 5.5 Bandit risk (9E)

**Decision:** real bandit NPCs with standing-conditioned hostility.

- Bandits are first-class `creature_type` agents declared in `.sim`. They
  have hostility predicates against trader-class agents.
- Bandit encounters are real combat. Goods captured become bandits' inventory
  → bandit wealth → bandit-favored fence-towns → emergent economic centers.
- Standing-conditioning: a faction's bandits ignore allies, predate enemies.
- Routes that traverse bandit-rich tiles accumulate bad safety records;
  traders avoid them; predator-prey dynamics shift bandit migration.

### 5.6 Geography-aware ability extensions (9F)

| Construct | Status |
|---|---|
| Gate predicates: `require_terrain <kind>`, `require_in_settlement_cluster`, `require_on_route(<id>)` | planned |
| Condition atoms: `at_settlement_cluster(<sid>)`, `distance_to_cluster(<sid>) <op> N`, `on_route(<rid>)`, `route_safety_above(<rid>, X)`, `region_price_above(<commodity>, X)`, `inventory_weight_above(X)` | planned |
| `EffectOp::EstablishRoute { from, to }` (variant 24) | planned |
| `EffectOp::JoinCaravan { caravan: GroupId }` (variant 25) | planned |
| Passive triggers: `on_arrived_at(<sid>)`, `on_route_traversed(<rid>)`, `on_bandit_encounter` | planned |

---

## §6 Phase 1: Dim 6 — Wealth & property

Property registry + inheritance chain + theft model + wealth-as-belief.

### 6.1 Property registration (6A)

**Decision:** property registry separate from inventory.

- New entity: `entity Property { kind: PropertyKindId, owner: OwnerRef, location: Vec3 }`.
- `PropertyKindId` declared in `.sim`: land tile, mine, field, building,
  business, etc.
- `OwnerRef = Agent(AgentId) | Group(GroupId) | None`.
- Property is transferable via `EffectOp::TransferProperty { property_id, target_sel }` (variant 19).
- Property registry is a sortable aggregate pool, queryable by owner /
  location / kind.

### 6.2 Inheritance / death transfer (6B)

**Decision:** wills + intestacy default chain + contested via existing auction
system.

- Per-agent non-replayable side state: `Will: SmallVec<[(AgentRef, ItemRef | PropertyRef); 4]>`.
  Default empty.
- New ability: `bequest <heir> of <items>` writes into the will. Status: `planned`.
- On `AgentDied`, cascade rule reads the will + applies intestacy fallback:
  spouse → eldest child → group leader → settlement cluster → world (loot).
- Intestacy fallback uses the agent's relationship graph to compute the next
  heir at death-time.
- Contested: heirs can dispute via `PostQuest{kind: Inheritance}` resolved
  via `MutualAgreement` / `Coalition` / `Majority`.
- New events: `EffectBequest`, `EffectInheritanceResolved`.

### 6.3 Theft & illegitimate transfer (6C)

**Decision:** theft / forcible transfer is a *contested* ability with detection
model and provoked response cascade. Outcomes scale with the power balance
between aggressor and target; victims and witnesses respond automatically
via reactive passives.

#### Two ability shapes

The forcible-transfer surface splits into two abilities by detectability:

- **Stealthy:** `pickpocket <target> of <commodity>` — skill-checked, intended
  to avoid detection. Contest is `caster.stealth_skill` vs `target.perception_skill`.
- **Open:** `demand <target> for <commodity>` — observable confrontation. Contest
  is power-balance: `caster.combat_power + ally_support` vs `target.combat_power + ally_support`. Used by sumptuary-law enforcers, mafia shakedowns, and bandits.

Both lower to `EffectOp::ForcibleTransfer` (variant 20), which now carries
a *contest type* discriminating skill-vs-skill from power-vs-power:

```rust
EffectOp::ForcibleTransfer {
    commodity_or_item:    SubjectRef,
    target_sel:           AgentSel,
    contest_kind:         ContestKind,         // Stealth | OpenPower | Authority
    detection_threshold:  f32,                 // for Stealth contests
}

enum ContestKind {
    Stealth,        // caster stealth vs target perception (pickpocket)
    OpenPower,      // caster combat power vs target combat power (demand / mug)
    Authority,      // caster's standing in target's group hierarchy (legal seizure)
}
```

#### Contest resolution

Resolution is a single roll at cast-resolve, deterministic over the world RNG:

- **Stealth:** `caster.stealth + nearby_ally_distraction - target.perception - witness_count` ≥ threshold → success (item transfers, no detection). Below threshold but above zero → success with detection. Below zero → caught in the act, no transfer.
- **OpenPower:** `caster.power_estimate(allies_in_range) - target.power_estimate(allies_in_range)` determines outcome. Margin > C: transfer succeeds (target submits). Margin in [-C, C]: combat engages instead. Margin < -C: caster backs down, no transfer, standing impact for the attempt.
- **Authority:** `caster.authority_in(target.group)` ≥ threshold → succeeds without combat. Below threshold → falls through to OpenPower.

`combat_power` is the existing scoring-relevant power summary the engine
uses for combat (HP, attack stats, ability portfolio). `power_estimate(allies)`
sums the caster + allies in `engagement_range` who would defend (those with
sufficient standing toward caster).

#### Detection & event emission

The detection rule fires *regardless* of contest outcome — an observed
attempt is itself a Chronicle-worthy event:

- Success-undetected → `EffectStolen { thief, victim, item, observed: false }`.
- Success-detected → `EffectStolen { thief, victim, item, observed: true }` + observers' memory entries.
- Combat-escalated → `EngagementCommitted { actor: target, engaged_with: caster }` (existing combat machinery takes over; whoever wins claims the spoils).
- Backed-down → `EffectAttemptedTheft { thief, victim, target, observed }` (no transfer; standing impact only).

#### Reactive response cascade

Every agent carries a default `on_demanded_of` reactive passive (engine-supplied,
not authored — agents always have it; class / personality differentiates the
chosen response). Triggered by `ForcibleTransfer` cast targeting them:

```
passive ReactToBeingDemanded {
    trigger: on_demanded_of(by: any)

    # Comply if dominated and personality is submissive
    submit when and(
        contest_outcome == "caster_wins",
        personality.submissiveness > 0.6,
    )

    # Engage if dominant or proud
    engage_in_combat when or(
        contest_outcome == "caster_loses",
        personality.pride > 0.5,
    )

    # Default: register grievance + flee if outmatched
    record_memory(grievance) + flee when default
}
```

The branch logic is per the agent's personality vector. Strong / proud agents
fight back even at low power; submissive agents comply even when they could
fight. Heterogeneity in 5A reaches into reaction policies.

Witnesses similarly carry `on_witnessed_demand_of_ally` and
`on_witnessed_demand_of_enemy` passives:

```
passive WitnessAllyRobbed {
    trigger: on_observed(EffectStolen | EffectAttemptedTheft, by: any, victim: ally)

    # Loyal allies intervene
    engage_in_combat when standing(self, victim) > 500

    # Concerned allies record
    record_memory(grievance) when standing(self, victim) > 100

    # Strangers stay out
    do_nothing when default
}
```

#### Authority-mode and emergent law

`ContestKind::Authority` is what gives sumptuary-law-style enforcement
(§11.1) its asymmetry: a noble's `forcible_transfer` against a peasant
in noble-ruled territory uses Authority contest, where the noble's
authority-in-the-group greatly exceeds the peasant's, so the contest
resolves toward success without combat. Outside the noble's authority
zone, the same ability falls through to OpenPower, where the noble
might lose to a stronger commoner.

This makes legal enforcement geographically conditional (works in the
city, breaks down in the wilds) and class-conditional (works against
peasants, fails against rival nobles) — both of which are the right
DF-grade dynamics.

#### IR additions

| Construct | Status |
|---|---|
| `ContestKind` enum (`Stealth | OpenPower | Authority`) | planned |
| `EffectOp::ForcibleTransfer { ..., contest_kind, ... }` (variant 20, revised) | planned |
| `EffectAttemptedTheft` event | planned |
| Default reactive passives `ReactToBeingDemanded`, `WitnessAllyRobbed`, `WitnessEnemyAttacked` | planned |
| Power-estimation views: `power_estimate(agent, ally_range)` materialized | planned |
| Authority view: `authority_in(agent, group)` materialized from standing + role | planned |
| Condition atoms: `contest_outcome`, `target_outmatched`, `target_overmatched` | planned |

### 6.4 Wealth as belief (6D)

**Decision:** `BeliefState` extended with `believed_wealth: f32, wealth_confidence: f32`.

- The Phase 1 TOM `BeliefState` (snapshot model from
  `2026-04-22-theory-of-mind-design.md`) gains two new fields.
- Wealth-belief updates fire on observable purchases / inventory glimpses /
  conspicuous-consumption events.
- Conspicuous consumption emerges: an agent buying status goods (5C)
  raises observers' wealth-belief.
- New condition atoms: `believes(target).wealth_above(X)`,
  `believes(target).wealth_below(X)`.
- Status: `planned` (depends on Phase 1 TOM landing in trunk).

### 6.5 IR additions for Dim 6

| Construct | Status |
|---|---|
| `EffectOp::TransferProperty { property_id, target_sel }` (variant 19) | planned |
| `EffectOp::ForcibleTransfer { subject, target_sel, contest_kind, detection_threshold }` (variant 20) | planned |
| `ContestKind` enum (Stealth, OpenPower, Authority) | planned |
| `entity Property { kind, owner, location }` | planned |
| `Will` per-agent side state | planned |
| `BeliefState.believed_wealth: f32, wealth_confidence: f32` | planned |
| Default reactive passives: `ReactToBeingDemanded`, `WitnessAllyRobbed`, `WitnessEnemyAttacked`, `OnAttempted`-family | planned |
| Power-estimation view: `power_estimate(agent, ally_range)` materialized | planned |
| Authority view: `authority_in(agent, group)` materialized from standing + role | planned |
| Events: `EffectPropertyTransferred`, `EffectStolen`, `EffectAttemptedTheft`, `EffectBequest`, `EffectInheritanceResolved` | planned |
| Gate predicates: `owns(property)`, `is_heir_of(target)`, `has_will` | planned |
| Condition atoms: `believes(target).wealth_above(X)`, `is_owned_by(<entity>, <agent>)`, `contest_outcome`, `target_outmatched`, `target_overmatched`, `would_provoke_response(victim)` | planned |

---

## §7 Phase 1: Dim 1 — Contracts & obligations

Persistent agent-to-agent obligations: debt, futures, insurance, retainer,
collateralized variants.

### 7.1 Obligation kinds (1A)

**Decision:** all four kinds + collateral variant in one entity, kind-specific
terms.

```rust
entity Obligation {
    kind: ObligationKind,
    parties: SmallVec<[AgentRef; 4]>,
    terms: ObligationTerms,
    status: ObligationStatus,
    created_tick: u32,
}

enum ObligationKind {
    Debt        { principal: u32, interest_q8: u8, due_tick: u32, collateral: Option<PropertyRef> },
    Future      { commodity: CommodityId, amount: u16, delivery_tick: u32, prepaid: u32 },
    Insurance   { premium: u32, payout: u32, peril: PerilKind, expires_tick: u32 },
    Retainer    { stipend_per_tick: u16, duration_ticks: u32, services_owed: SmallVec<[RecipeId; 4]> },
    Service     { recipe: RecipeId, payment: u32 },  // existing-style pledge
}

enum ObligationStatus { Active, Discharged, Defaulted, Disputed, Cancelled }
```

### 7.2 Enforcement (1B)

**Decision:** standing penalty on default + automatic property seizure for
collateralized debts.

- On `due_tick` arrival, cascade rule checks debtor's `gold` against
  `principal + interest`. If insufficient, fires `ContractDefaulted` event.
- If `collateral` is `Some`, fires `EffectPropertyTransferred(collateral, debtor → creditor)` (the seizure is enacting pre-pledged contract terms, not external enforcement).
- Standing impact: `EffectStandingDelta(debtor, creditor, -300)` and
  population-wide `EffectStandingDelta(observer, debtor, -30)` for observers.
- Court-style adjudication (a third-party authority enforcing) is emergent
  from the existing quest system — a creditor can post `PostQuest{kind: BringToJustice, target: debtor}`.

### 7.3 Storage / IR (1C)

**Decision:** obligation entities + per-agent debtor/creditor indices.

- World-level `AggregatePool<Obligation>` (matches existing Quest /
  Auction pool pattern).
- Per-agent indices: `obligations_owed_by_me: SortedVec<ObligationId, 8>`,
  `obligations_owed_to_me: SortedVec<ObligationId, 8>`. Updated on
  obligation creation / discharge.

### 7.4 Pricing (1D)

**Decision:** risk-adjusted market pricing.

- Borrowers post `PostQuest{kind: BorrowGold, principal, term, max_rate, collateral?}`.
- Lenders read the post + their belief about borrower's wealth + standing.
  Bid with `Payment::Reciprocal{...obligation terms}` carrying their proposed rate.
- Risk-adjusted: lender's bid rate scales with their belief about default
  probability (a function of borrower's `believed_wealth` and `standing`).
- Auction resolution: lowest-rate bid wins (existing `AuctionResolution::HighestBid`
  with bid = inverse-of-rate).

### 7.5 IR additions for Dim 1

| Construct | Status |
|---|---|
| `entity Obligation { kind, parties, terms, status, created_tick }` | planned |
| `EffectOp::CreateObligation { kind, parties, terms }` (variant 21) | planned |
| `EffectOp::DischargeObligation { obligation_id }` (variant 22) | planned |
| `EffectOp::DefaultObligation { obligation_id }` (variant 23) | planned |
| Per-agent obligation indices | planned |
| Events: `ObligationCreated`, `ObligationDischarged`, `ObligationDefaulted`, `ContractInsuranceClaimed` | planned |
| Condition atoms: `has_obligation_to(target)`, `owes_total_above(X)`, `obligation_due_within(<dur>)`, `creditor_count`, `debtor_count` | planned |
| New ability hint: `AbilityHint::Financial` | planned |

---

## §8 Phase 2: Dim 3 — Labor & skills

Per-recipe continuous skill, full acquisition pipeline, market wages,
contractual apprenticeship, emergent migration. **Guilds are plain groups
with no system-enforced licensing; monopolies emerge from social pressure.**

### 8.1 Skill model (3A)

**Decision:** per-recipe continuous skill `[0.0, 1.0]` with cross-recipe
transfer matrix.

- Per-agent `skills: SortedVec<(RecipeId, f32), 16>`.
- Compile-time `transfer_matrix: SparseMatrix<RecipeId, RecipeId, f32>` — practicing
  one recipe partially raises related ones (forge_sword → forge_dagger ≈ 0.6).
- Recipe gates read per-recipe skill (`require_skill <recipe> >= 0.4`).
- Status: `planned`.

### 8.2 Skill acquisition (3B)

**Decision:** practice + observation + apprenticeship + books + disuse-decay.

- **Practice:** each successful cast updates `skills[recipe] += 0.01 * (1 - skills[recipe])` (asymptotic to 1.0).
- **Observation:** observers within range of a cast accumulate `skills[recipe] += 0.001 * (1 - skills[recipe])` per observation. Slower than practice.
- **Apprenticeship:** mentor-apprentice relationship multiplies observation rate by 5×.
- **Books:** reading a `Document` carrying recipe data confers a one-shot bump (`+0.05` capped).
- **Disuse-decay:** unused skills decay slowly (`skills[recipe] *= 0.999` per 100 ticks since last use).

### 8.3 Wage formation (3C)

**Decision:** market wages with skill-conditioned reserves.

- Worker availability: `PostQuest{kind: Service, terms: { recipe, min_payment_per_cast }}`.
- `min_payment_per_cast` scales with `skills[recipe]`.
- Employers' `Bid`s respect the reserve; matching at auction-resolve.

### 8.4 Apprenticeship (3D)

**Decision:** time-bounded contract via `Service` obligation.

- Mentor-apprentice relationship: `InviteToGroup{kind: Family, sub-role: Apprentice}`.
- The apprenticeship is also an `Obligation { kind: Service }` — the apprentice owes the master a fixed number of recipe-casts at reduced wage during the term.
- On term completion, the obligation is `Discharged`; apprentice's membership tier upgrades.

### 8.5 Guild structure (3E)

**Decision:** guilds as plain groups; no system-enforced licensing; monopolies emerge.

- Guilds use generic `Group{kind: Guild}` machinery. No special economic privileges.
- Recipe casting is open to anyone with inputs + skill + tools.
- Guild monopolies emerge from social mechanisms (see §11.1):
  - Pricing collusion via standing-conditioned trade refusal.
  - Reputation moat via TOM-distributed merchant reputation.
  - Information moat via apprenticeship-only-in-guild-family + book scarcity.
  - Active enforcement via vigilante combat against scabs (passive abilities shared by guild members).

### 8.6 Migration (3F)

**Decision:** migration as emergent scoring outcome.

- No explicit "satisfaction" variable.
- Agent's per-tick scoring already considers `MoveToward(better_job)` vs `Stay(current_job)`. If higher-paying / better-conditions opportunities outscore current, agent migrates.
- Inputs to scoring: wage, standing with employer, working conditions (tool quality), guild access, distance.

### 8.7 Strikes (3G)

**Decision:** coordinated quit emerges from individual scoring.

- A wage cut affecting all workers in a shop produces same-direction quit decisions in all of them simultaneously.
- Chronicle entries fire when N agents quit the same employer in the same tick.

### 8.8 IR additions for Dim 3

| Construct | Status |
|---|---|
| `Agent.skills: SortedVec<(RecipeId, f32), 16>` | planned |
| Compile-time `RecipeTransferMatrix` | planned |
| Gate predicate: `require_skill <recipe, threshold>` | planned (already in §4.8) |
| Skill-update cascade rules: practice / observation / book / decay | planned |
| `Apprenticeship` as `(InviteToGroup{kind: Family, role: Apprentice}, Obligation{kind: Service})` composite | planned |
| Events: `SkillRaised`, `ApprenticeshipBegun`, `ApprenticeshipCompleted`, `MassQuit` | planned |
| Condition atoms: `target_skill_above(target, recipe, threshold)`, `is_apprentice_of(target)`, `is_master_of(target)` | planned |

---

## §9 Phase 2: Dim 5 — Heterogeneous preferences

Per-agent commodity preferences, status goods, addiction, brand loyalty,
ethical bans.

### 9.1 Preference storage (5A)

**Decision:** per-agent preferences with class-derived defaults.

```rust
struct AgentPreferences {
    commodity_affinity:    SortedVec<(CommodityId, f32), 16>,    // [-1, 1]
    status_threshold:      f32,
    merchant_loyalties:    SortedVec<(AgentId, f32), 8>,
    ethical_bans:          SortedVec<EthicalConstraint, 8>,
    addictive_attachments: SortedVec<(CommodityId, f32), 4>,
}
```

Class tags supply defaults at agent spawn; agents deviate via experience.

### 9.2 Preference dimensions (5B)

**Decision:** full set — commodity affinity, status, merchant, ethical, addictive.

Each dimension reads at decision time during scoring; updates are event-driven:
- Trade with merchant X raises `merchant_loyalties[X]`.
- Consumption of addictive commodity raises `addictive_attachments[c]`.
- Cultural / faith events register `ethical_bans` adjustments.
- Status threshold rises with social rank.

### 9.3 Status goods (5C)

**Decision:** status as social signal via wealth-belief.

- Status goods are visible-in-inventory items (apparent quality / origin).
- Buying a status good is observable; observers update their wealth-belief
  about the buyer (6D.γ).
- The buyer's preference for status goods scales with their `status_threshold`,
  which is class-conditioned.
- Veblen-good behavior emerges: as a status good's price rises, its
  signaling power rises, and demand from status-seekers rises.

### 9.4 Addiction (5D)

**Decision:** linear preference accumulation with withdrawal.

- Each consumption of an addictive commodity raises `addictive_attachments[c] += delta`.
- Without consumption for N ticks, attachment decays slowly but craving rises:
  the agent's scoring weights for that commodity rise.
- Without consumption past threshold: agent suffers debuff (`hot_withdrawal_active = true`,
  attached to existing buff/debuff machinery).

### 9.5 Ethical bans (5E)

**Decision:** per-agent constraints with class-derived defaults.

- `EthicalConstraint = { commodity: Option<CommodityId>, source_predicate: SourcePredicate }`.
- Mask-level: `Bid` is illegal when the bidder's bans match the auction's
  commodity / source.
- Runtime composition: a generated ability that violates the agent's bans is filtered at the agent's applicability gate.

### 9.6 IR additions for Dim 5

| Construct | Status |
|---|---|
| `AgentPreferences` per-agent state | planned |
| `EthicalConstraint` enum | planned |
| Mask gate: `not_banned(commodity, source)` | planned |
| Condition atoms: `prefers(target, commodity)`, `is_loyal_to(target, merchant)`, `addicted_to(target, commodity)`, `addiction_severity(target, commodity) > X` | planned |
| Events: `PreferenceShifted`, `AddictionDeveloped`, `WithdrawalOnset`, `EthicalBanAdded`, `EthicalBanLifted` | planned |

---

## §10 Phase 2: Dim 4 — Information asymmetry & trust

Counterfeit goods, distributed merchant reputation, brokerage, search cost.

### 10.1 Belief surface (4A)

**Decision:** TOM-style beliefs over commodities, merchants, routes, quests.

All four belief domains share storage shape (`BeliefState`-like snapshots
with confidence + decay) but live in separate per-agent maps:
- `cold_price_beliefs: SortedVec<(CommodityId × SettlementCluster, PriceBelief), 16>`.
- `cold_merchant_beliefs: SortedVec<(AgentId, MerchantBelief), 8>`.
- `cold_route_beliefs: SortedVec<(RouteId, RouteBelief), 8>`.
- `cold_quest_beliefs: SortedVec<(QuestId, QuestBelief), 8>` (already in ability DSL §20.5).

### 10.2 Counterfeit goods (4B)

**Decision:** apparent-vs-actual attributes; appraisal-skill-gated detection.

- Every observable item attribute has an apparent and an actual: `apparent_quality, actual_quality`, `apparent_material, actual_material`, etc.
- Buyers without sufficient `appraisal` skill read apparent values; experts read actual values.
- Counterfeiting is itself a recipe: `ForgeAppearance { consume: papercraft + dyeing + makeup, target: item, output: same_item_with_fake_attributes }`.
- The runtime composition model can generate counterfeit recipes for any item.

### 10.3 Merchant reputation (4D)

**Decision:** distributed reputation via TOM/gossip.

- A merchant's reputation isn't stored centrally — it's the population-distributed
  view over individual `cold_merchant_beliefs`.
- Each trade fires a memory event for both parties; observers nearby register
  the trade outcome (was item authentic, was price fair).
- Beliefs gossip via existing Communicate / Announce machinery.
- The "merchant has a reputation" emerges from the average belief among agents who care.

### 10.4 Brokerage (4E)

**Decision:** brokerage as service recipe + broker reputation.

- A `inquire_about_market` ability is itself a recipe: caster pays gold,
  receives belief updates from broker's known prices.
- Broker carries their own merchant-reputation belief; brokers with poor
  accuracy track records earn lower fees.

### 10.5 Search cost (4F)

**Decision:** search cost is implicit movement cost.

- Searching for prices requires moving (which takes time + risks decay /
  bandits / inventory weight). No explicit search-cost mechanism.
- Poor agents face higher relative search cost (can't afford to travel).

### 10.6 IR additions for Dim 4

| Construct | Status |
|---|---|
| `cold_price_beliefs`, `cold_merchant_beliefs`, `cold_route_beliefs`, `cold_quest_beliefs` per-agent maps | planned |
| `apparent_*` fields on observable item attributes | planned |
| `appraisal` as a `SkillId` | planned |
| Counterfeiting as a recipe family | planned |
| Condition atoms: `believes_genuine(item)`, `apparent_quality(item) > threshold`, `actual_quality(item) > threshold`, `merchant_belief(target).fair_dealer > X` | planned |

---

## §11 Phase 3: Dim 7 — Market structure (mostly emergent)

Cartels, monopolies, predatory pricing, embargoes — all emergent from Phase 1+2 primitives. Sumptuary laws are emergent too via shared passives.

### 11.1 How each phenomenon emerges

**Cartels** — a `Group{kind: Guild}` whose members have aligned price-belief
gossip + standing-conditioned trade refusals against non-members. No new
state.

**Monopolies** — emerge when one agent or group's recipe output saturates a
commodity's supply. Detectable via the per-tile commodity views: `surplus_from_one_source(commodity, threshold)`. No new state.

**Predatory pricing** — emerges when a wealthy agent posts `Bid`s at
ruinous-low prices to drive competitors to default; their wealth-reserve
absorbs the temporary loss. Wealth-belief (6D) + price-beliefs + endurance
(gold reserves) cover it.

**Embargoes** — emerge from `SetStanding{group, Hostile}` translating into
"members refuse to trade with members of hostile group" via 5E.β cultural
defaults.

**Sumptuary laws** — emerge from shared passives that fire *contested*
forcible-transfer attempts. The noble class has a passive whose trigger is
`on_observe_item(commodity == silk, target.class != noble, target.distance
< observation_range)` and whose effect is
`forcible_transfer(target, silk, contest_kind: Authority) +
modify_standing(target, -200)`. Every noble carries this passive.

Critically, the action is *contested*, not unconditional (per §6.3):

- A weak noble confronting a powerful commoner in noble territory: Authority
  contest succeeds → silk is taken. Noble enforcement holds.
- A weak noble confronting a powerful commoner in the wilds: Authority
  contest falls back to OpenPower → contest loses → no transfer + standing
  impact for the failed attempt. Outside their power base, nobles can't
  enforce.
- A peasant warrior with allies witnessing the attempt: their `WitnessAllyRobbed`
  passive may fire `engage_in_combat` if standing-with-victim is high enough.
  Mob justice can override sumptuary enforcement when the crowd sympathizes
  with the violator.
- A wealthy commoner who's gathered loyal retainers: their retainers'
  `WitnessAllyRobbed` engages with the noble. The noble loses. Sumptuary
  law breaks down for the rich-but-common.

This is the right emergent shape — laws work where the enforcing group has
enough collective power, and break down at the edges. The same generalizes
to: prohibition (guards confiscating drink — succeeds in the city, fails
in the bandit camp), embargoes (foreign-faction goods burned by patrols),
drug laws, weapons restrictions.

All these are `passive` blocks shared across enforcing groups; level-scaling
is automatic via the contest mechanics from §6.3; provoked responses fire
through the default reactive passives carried by every agent.

**Predatory eviction / forced sales** — emerge from theft (6C), default
seizure (1B), and combat. No new primitive.

### 11.2 Phase 3 / Dim 7 IR additions

| Construct | Status |
|---|---|
| Passive trigger: `on_observe_item(commodity_pred, target_pred, range)` | planned |
| Detection cascade for visible-inventory items | planned |
| Condition atoms: `surplus_from_one_source(commodity, threshold)`, `dominant_supplier(commodity)` | planned |
| Verification harness scenarios: cartel formation, monopoly capture, predatory-pricing siege | planned |

The only new primitive is the `on_observe_item` passive trigger, which
generalizes the existing observation cascade.

---

## §12 Phase 3: Dim 8 — Financial instruments

Banking, interest, equity, speculation, pawnbroking — almost all emergent.
Single new primitive: transferable obligations.

### 12.1 How each instrument emerges

**Banking** — emerges from one agent issuing many `Debt` obligations. The agent who holds many obligations is a *bank* by behavior, not by type.

**Interest rates** — already locked in 1D.γ as risk-adjusted market pricing.

**Bills of exchange / transferable obligations** — **new primitive needed.** A holder of a `Debt` may transfer it to a third party at a discount. Requires:
- New field: `Obligation.transferable: bool`.
- New `EffectOp::TransferObligation { obligation_id, target_sel }` (variant 26).
- Status: planned. Single small extension.

**Equity shares in groups** — existing membership records carry
`member_share: f32`. Transferable via the same mechanism as transferable obligations, treating share as a kind of obligation.

**Speculation markets** — futures contracts (Dim 1A) traded between non-original parties. Same primitive as bills of exchange.

**Pawnbroking** — already covered: it's a `Debt + collateral` with short term.

**Insurance pools / mutual aid** — emerge from a group whose members each contribute a small premium and the pool pays out on member loss. Pure composition of obligations + group treasury.

### 12.2 Phase 3 / Dim 8 IR additions

| Construct | Status |
|---|---|
| `Obligation.transferable: bool` field | planned |
| `EffectOp::TransferObligation { obligation_id, target_sel }` (variant 26) | planned |
| Verification harness: bank formation, mutual-aid pool, futures speculation | planned |

---

## §13 Phase 3: Dim 10 — Macro dynamics (entirely emergent)

Inflation, boom/bust, credit crunches, resource shocks, war economy,
famine→migration→labor — all emergent from Phase 1+2 primitives. No new
primitives. Phase 3 work for Dim 10 is verification + Chronicle templates.

### 13.1 How each macro phenomenon emerges

| Phenomenon | Emerges from |
|---|---|
| Inflation/deflation | Currency supply (gold mining via 2D.γ) vs goods produced (Dim 2). Net price-belief drift across population is the metric. |
| Boom/bust cycles | Belief contagion + lagged supply response. Optimistic belief about commodity X → agents borrow to invest → glut → prices crash → defaults cascade → pessimism spreads. |
| Credit crunches | Default cascade (1B): wave of defaults raises perceived default-risk → interest rates rise (1D.γ) → fewer loans → economic contraction. |
| Resource shocks | Depletion (2D.γ) + shortage cascade (2F.γ): mine collapses, iron prices triple, downstream chains stall, smiths migrate (3F.δ). |
| War economy | Group standing → embargo-by-default + combat redirection of labor. |
| Famine → migration → labor oversupply | Food shortage (2F.γ + 2D.γ) + migration (3F.δ) + labor wage formation (3C.γ). |

### 13.2 Phase 3 / Dim 10 work

| Construct | Status |
|---|---|
| Verification harness scenarios for each macro phenomenon | planned |
| Chronicle entry templates: `IronHillsDepleted`, `BakersCartelCollapsed`, `BankFoundationByX`, `FamineSpread`, `WarEconomyDeclared`, etc. | planned |
| Macro metrics dashboard (price drift, default rate, gini coefficient, labor migration rate) | planned |

---

## §14 Cross-cutting: runtime ability composition

A small model composes new abilities — including new recipes — at runtime
based on agent state. Tractability constraints:

### 14.1 Generation triggers (rare)

Generation fires on stuck-state signals, not per-tick. Triggers declared
in `.sim`:
- *Idle-with-resources:* agent has commodities + tools but no applicable
  recipe scored above ε.
- *Persistent income gap:* `income_rate < survival_threshold` for N ticks.
- *Sustained market signal:* commodity demand-above-supply for N ticks
  with agent holding prerequisites.
- *Imitation impulse:* agent observed (memory event / testimony) an ability
  they don't know but have inputs for.

Default trigger sensitivity: ~1 invocation per agent per several thousand
ticks.

### 14.2 Slot-fill structure

The model emits values for a fixed schema, not free tokens:
```
Slots:
  template:  one of N seed templates (~30 patterns: forge, mill, brew, weave, contract, …)
  inputs:    subset of agent's inventory (constrained)
  outputs:   subset of demanded commodities (constrained)
  skill:     one of agent's known skills (constrained)
  tool:      one of available tools (constrained)
  quality:   one of K named quality formulas with parameterized weights
```

Combinatorial space is bounded; most slot combinations fail downstream gates.

### 14.3 Three hard gates

- **Grammar gate:** well-formed `AbilityProgram` (compile-checks).
- **Economic gate:** value-conservation predicate. `Σ expected_output_value > Σ expected_input_value × (1 - skill_factor)`. Anti-money-pump.
- **Novelty gate:** cosine similarity against agent's known abilities; reject if `> 0.9` (duplicate).

### 14.4 Population-level discovery

Inventions propagate via existing information channels. Generation is rare
(~50 inventions per game-year per 10K agents); learning is cheap (~5K
instances). 100× cost amortization vs. per-agent generation.

### 14.5 Bounded registries

- Per-agent: `K_authored + K_generated ≤ MAX_ABILITIES = 8` slots, of which
  `K_generated ≤ 4`.
- World-level `AbilityRegistry`: ~4096 entries cap; LRU GC for unused.
- Schema-hash partition: `[1..N_authored]` covered by hash; `[N_authored+1..MAX]`
  recorded in trace, not hash.

### 14.6 Generated abilities as goals

A generated ability that fails its applicability gate isn't useless — it
becomes a goal for the GOAP planner. "I invented PaperArmor but lack
papercraft skill" → goal: acquire papercraft skill → find a teacher →
apprentice → cast.

The reactive evaluator handles cast-able abilities; the GOAP planner
handles goal-able abilities. Generation feeds both surfaces.

---

## §15 Cross-cutting: training curriculum (referenced)

Detailed curriculum belongs in a separate plan. This spec records the
*shape* so the action surface is designed against it.

| Stage | Objective | Cost | Notes |
|---|---|---|---|
| 0 — Vocabulary | Masked-token prediction on authored abilities | low | Existing pretrain extended to economic abilities |
| 1 — Slot-fill imitation | Predict slot values for authored applicable abilities given context | low | Trains state → recipe-shape mapping |
| 2 — Constrained variation | Generate slots passing grammar + economic gates | low | No game runs yet |
| 3 — Short-horizon profitability | Microsim rollouts, reward = wealth(t+100) − wealth(t) | medium | Filters unprofitable inventions |
| 4 — Adoption signal | Multi-agent games (50 agents, 50K ticks); reward = adoption count | high | Multi-agent fixed-point territory |
| 5 — Long-horizon market | Full-scale games; reward shaped from agent wealth percentile | very high | Convergence not guaranteed |

Early-validation checkpoint: after Stage 0–1, given a held-out blacksmith-with-paper state, model should slot-fill into something paper-armor-shaped. If yes, architecture works. If no, no amount of Stage 4–5 saves it.

---

## §16 Cross-cutting: emergent settlement view

Settlements are not declared; they materialize from agent + structure
clustering.

### 16.1 Settlement view

```
@materialized view settlement_clusters() → [SettlementCluster]
  = cluster_agents_and_structures(
      agents:        all alive agents,
      structures:    all owned property of kind Structure,
      threshold:     min 5 agents within 50m radius + at least 1 structure,
    )
```

Each cluster carries:
- `cluster_id: SettlementId` (assigned deterministically by sorted-min-agent-id).
- `members: SortedVec<AgentId, K>`.
- `structures: SortedVec<PropertyId, K>`.
- `centroid: Vec3`.

Settlements form, dissolve, and merge as agents move and structures change
ownership. No `FoundSettlement` action; a sufficient cluster simply *is* a
settlement.

### 16.2 Settlement-conditional reads

Condition atoms reading the settlement view: `at_settlement(<sid>)`,
`in_any_settlement`, `settlement_size_above(<n>)`. These read the
materialized view at scoring time.

### 16.3 No settlement-as-entity

The previous `Settlement` entity (per `npc_economy_plan.md`) is replaced
by this materialized view. Any state that previously lived on `Settlement`
(treasury, stockpile, leadership) becomes either:
- Group state (settlement leadership = group leader of cluster's primary group).
- Aggregate over cluster-member state (treasury = sum of member contributions
  to mutual-aid pool).

This makes settlements emergent in the same sense as cartels and routes.

---

## §17 IR additions consolidated

Net new constructs across all dimensions, organized by IR layer.

### 17.1 New `EffectOp` variants

Continuing from ability DSL spec §22 (which ended at variant 16). Economic
ops occupy variants 17+:

| Ord | Variant | Phase | Status |
|---|---|---|---|
| 17 | `Recipe { recipe, target_tool_sel }` | 1 | planned |
| 18 | `WearTool { tool_kind, amount }` | 1 | planned |
| 19 | `TransferProperty { property_id, target_sel }` | 1 | planned |
| 20 | `ForcibleTransfer { commodity_or_item, target_sel, detection_threshold }` | 1 | planned |
| 21 | `CreateObligation { kind, parties, terms }` | 1 | planned |
| 22 | `DischargeObligation { obligation_id }` | 1 | planned |
| 23 | `DefaultObligation { obligation_id }` | 1 | planned |
| 24 | `EstablishRoute { from, to }` | 1 | planned |
| 25 | `JoinCaravan { caravan: GroupId }` | 1 | planned |
| 26 | `TransferObligation { obligation_id, target_sel }` | 3 | planned |

### 17.2 New entities

| Entity | Phase | Status |
|---|---|---|
| `Tool : Item { kind, quality, wear, durability, owner }` | 1 | planned |
| `Component : Item { kind, quality, wear, durability }` | 1 | planned |
| `ResourceNode { kind, location, remaining, capacity, regen_rate }` | 1 | planned |
| `Property { kind, owner, location }` | 1 | planned |
| `Obligation { kind, parties, terms, status, created_tick }` | 1 | planned |

### 17.3 New registries (in `.sim`)

| Registry | Phase |
|---|---|
| `RecipeRegistry` | 1 |
| `ToolKindRegistry` | 1 |
| `CommodityRegistry` | 1 |
| `SkillRegistry` | 1 |
| `ComponentKindRegistry` | 1 |
| `RouteRegistry` (auto-populated) | 1 |
| `RecipeTransferMatrix` (compile-time) | 2 |

### 17.4 Per-agent state extensions

| Field | Phase | Status |
|---|---|---|
| `skills: SortedVec<(RecipeId, f32), 16>` | 2 | planned |
| `preferences: AgentPreferences` | 2 | planned |
| `cold_price_beliefs` | 2 | planned |
| `cold_merchant_beliefs` | 2 | planned |
| `cold_route_beliefs` | 1/2 | planned |
| `cold_quest_beliefs` | 2/L | planned |
| `obligations_owed_by_me`, `obligations_owed_to_me` | 1 | planned |
| `Will` (non-replayable side state) | 1 | planned |
| `BeliefState.believed_wealth: f32, wealth_confidence: f32` | 1 | planned |

### 17.5 New events

| Event | Phase | Replayable |
|---|---|---|
| `RecipeCast { recipe, agent, tick }` | 1 | yes |
| `RecipeCompleted { recipe, agent, output_quality, tick }` | 1 | yes |
| `ToolWornOut { tool, owner, tick }` | 1 | yes |
| `ProductionStalled { recipe, agent, reason, tick }` | 1 | yes |
| `ResourceDepleted { node_id, tick }` | 1 | yes |
| `ResourceRegenerated { node_id, amount, tick }` | 1 | yes |
| `EffectPropertyTransferred { property, from, to, tick }` | 1 | yes |
| `EffectStolen { thief, victim, item, tick, observed }` | 1 | yes |
| `EffectBequest { from, heir, item, tick }` | 1 | yes |
| `EffectInheritanceResolved { decedent, items, tick }` | 1 | yes |
| `ObligationCreated { kind, parties, terms, tick }` | 1 | yes |
| `ObligationDischarged { obligation, tick }` | 1 | yes |
| `ObligationDefaulted { obligation, tick }` | 1 | yes |
| `ContractInsuranceClaimed { obligation, tick }` | 1 | yes |
| `SkillRaised { agent, recipe, new_value, tick }` | 2 | yes (sampled) |
| `ApprenticeshipBegun { master, apprentice, term, tick }` | 2 | yes |
| `ApprenticeshipCompleted { master, apprentice, tick }` | 2 | yes |
| `MassQuit { employer, n_workers, tick }` | 2 | yes |
| `PreferenceShifted { agent, dim, delta, tick }` | 2 | yes (sampled) |
| `AddictionDeveloped { agent, commodity, severity, tick }` | 2 | yes |
| `WithdrawalOnset { agent, commodity, tick }` | 2 | yes |
| `ChronicleMacro { template, args, tick }` | 3 | non-replayable |

### 17.6 New gate predicates

| Gate | Phase | Status |
|---|---|---|
| `require_skill <recipe, threshold>` | 1 | planned |
| `require_tool <tool_kind>` | 1 | planned |
| `require_in_settlement_cluster` | 1 | planned |
| `require_terrain <kind>` | 1 | planned |
| `require_on_route(<route_id>)` | 1 | planned |
| `not_banned(commodity, source)` | 2 | planned |

### 17.7 New condition atoms

(Catalogued per dimension above; consolidated catalog ~40 atoms.)

### 17.8 New `AbilityHint` variants

| Variant | Status |
|---|---|
| `Economic` | planned |
| `Financial` | planned |

### 17.9 New passive triggers

| Trigger | Phase |
|---|---|
| `on_arrived_at(<sid>)` | 1 |
| `on_route_traversed(<rid>)` | 1 |
| `on_bandit_encounter` | 1 |
| `on_production_stalled(<recipe>)` | 1 |
| `on_resource_depleted(<node_id>)` | 1 |
| `on_obligation_due(<kind>)` | 1 |
| `on_obligation_defaulted(<creditor>)` | 1 |
| `on_observe_item(commodity_pred, target_pred, range)` | 3 |

---

## §18 Capability status matrix (consolidated)

By dimension and phase. Mostly `planned` (engine work); a few `runs-today`.

| Dim | Phase | Status |
|---|---|---|
| 2 — Supply chains | 1 | planned (extends existing 8-commodity system) |
| 9 — Geography | 1 | planned |
| 6 — Wealth & property | 1 | planned (existing gold + items; new property registry) |
| 1 — Contracts | 1 | planned |
| 3 — Labor & skills | 2 | planned |
| 5 — Preferences | 2 | planned |
| 4 — Information & trust | 2 | planned |
| 7 — Market structure | 3 | mostly emergent + new `on_observe_item` trigger |
| 8 — Financial instruments | 3 | mostly emergent + `TransferObligation` + `transferable` flag |
| 10 — Macro dynamics | 3 | entirely emergent |

---

## §19 Verification harness

Every claimed emergent phenomenon needs a regression test that demonstrates
it under controlled initial conditions. Tests are property-style assertions
over simulation runs.

### 19.1 Phase 1 verification

| Test | Setup | Assertion |
|---|---|---|
| Multi-stage chain | Iron ore present, smith with skills, smelter | After N ticks, swords are produced; intermediates trade |
| Tool wear | Smith forging continuously | After M ticks, tool wear ≥ threshold; smith repairs or buys new |
| Resource depletion | Mine with bounded `remaining` | Depletion event fires; smiths migrate or shift recipes |
| Perishable transit | Food shipped over distance | Arrived quantity < dispatched quantity at expected decay rate |
| Inheritance | Wealthy agent dies with will | Listed heirs receive items; intestacy fallback for unlisted |
| Theft detection | Pickpocket near observers | Detected if appraisal > stealth; standing impact fires |
| Debt default | Debtor without funds at due_tick | Collateral seized; standing falls |

### 19.2 Phase 2 verification

| Test | Setup | Assertion |
|---|---|---|
| Skill specialization | Two smiths, different practice patterns | Skill profiles diverge; recipe selection differs |
| Apprenticeship | Master + apprentice, 1000-tick term | Apprentice's skill rises faster than baseline observation |
| Wage migration | Two settlements, one drops wages | Workers migrate to higher-wage location |
| Strike | Wage cut applied to N workers | All N quit in same tick (or close) |
| Status goods | Noble vs peasant in shop | Noble buys silk preferentially; peasant ignores |
| Counterfeit | Forged "masterwork" sword | Buyers without appraisal pay full price; experts spot fake |
| Merchant reputation | Repeated trades with one merchant | Reputation belief converges; loyal customers form |

### 19.3 Phase 3 verification

| Test | Setup | Assertion |
|---|---|---|
| Cartel | Three guild members + standing-conditioned refusals | Outsider price-bidding excluded over time |
| Monopoly | One agent dominates iron supply | Surplus-from-one-source view fires |
| Bank formation | Wealthy agent issues many debts | Reputation as "bank" emerges in population belief |
| Sumptuary law | Nobles share `on_observe_item` passive | Peasants' silk gets confiscated repeatedly |
| Inflation | Increased gold-mine output | Average price-belief drift detectable |
| Resource shock | Mine collapse | Downstream prices triple; smiths migrate |
| Famine cascade | Crop failure | Food shortage → migration → labor surplus elsewhere |

---

## §20 Chronicle templates

Macro-event narration templates for game-history readability. Each template
fires from a cascade rule when its conditions detect an emergent macro
phenomenon.

| Template | Detected by | Example output |
|---|---|---|
| `IronHillsDepleted` | `ResourceDepleted` event on a major node | "In the year 247, the Iron Hills ran dry." |
| `BakersCartelCollapsed` | Cartel-formed-then-broken pattern | "The Bakers' Guild lost their grip on flour after the Salt Riots." |
| `BankFoundationByX` | Agent crosses N obligations-held threshold | "Madame Verlaine's loan-house grew so vast that her name became coin." |
| `FamineSpread` | Cascading food shortage across N clusters | "Three winters of failed harvest emptied the eastern hamlets." |
| `WarEconomyDeclared` | Group standing → Hostile + production redirection | "The Court redirected all forges to swords; the price of plowshares tripled." |
| `LegendaryCraftsperson` | Agent reaches skill threshold + reputation threshold | "Old Tomek's blades were said to never break." |
| `MarketCornerSuccess` / `…Failure` | Market dominance attempt outcome | "The Ironmonger's gambit collapsed when the second mine reopened." |
| `MassMigration` | N agents migrate same direction within window | "The roads filled with weavers leaving the mill towns." |
| `InventionSpread` | Generated ability adopted by N agents | "Paper armor, born of Hadrian's poverty, became the new fashion of the militia." |

---

## §21 Appendix: delta vs `npc_economy_plan.md`

The 2026-03-28 plan implemented a working but shallow economy (8 commodities,
single-stage production, settlement founding, gold + tax). This spec
extends and reshapes:

### 21.1 Carries forward

- 8 commodities expanded to a `CommodityRegistry` declared in `.sim`.
- Per-NPC gold + inventory.
- Per-NPC price beliefs (now part of broader belief surface in §10).
- Settlement treasury (now emergent group treasury per §16).
- Production from class tags (now per-recipe skill in §8).

### 21.2 Newly introduced

- All of §3 — recipes-as-abilities anchor; runtime composition.
- Multi-stage production chains (§4).
- Component-based assembly (§4.7).
- Tool quality + depreciation (§4.3).
- Resource depletion + regeneration (§4.4).
- Perishables (§4.5).
- Geography & transport (§5).
- Property registry, inheritance, theft (§6).
- Contracts & obligations (§7).
- Per-recipe skill + apprenticeship (§8).
- Heterogeneous preferences (§9).
- Information asymmetry (§10).
- Emergent cartels / monopolies / sumptuary laws (§11).
- Transferable obligations / banking emergence (§12).
- Macro dynamics framework (§13).

### 21.3 Reshaped

- Settlements: from declared entities to materialized clusters (§16).
- Production: from class-tag-driven engine logic to recipe-as-ability (§3).
- Wages: from implicit class-tag-driven to market-driven via Service auctions (§8.3).
- Pricing: from hyperbolic-saturation single-formula to per-agent belief
  with TOM-grade gossip propagation (§10).
- Wealth visibility: from public to belief-based (§6.4).

### 21.4 Removed

- `Settlement` as primary entity (replaced by §16 view).
- 5% settlement tax as automatic — replaced by emergent group-treasury contributions.
- "Class consolidation" patches — class drift becomes a tunable scenario property.

---

*End of spec.*
