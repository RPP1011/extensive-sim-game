# Policy Schema Proposal — Defender's Brief

## Executive position

The proposal in `docs/ecs_dsl/proposal_policy_schema.md` is substantively correct and I will defend it with minor strengthening. The four commitments it makes — packed ~1500–1800-float observation declared in the DSL, multi-head action space (categorical + pointer + continuous) with per-tick mask, polymorphic backend per role, and slot-based variable-length data with `exists` padding — are the only shape that simultaneously (1) exposes enough of `NpcData` to drive the 50–150 verb action vocabulary derived from the v2 reframings, (2) admits a utility/neural/LLM spectrum that matches the roughly 20K-named-NPC / 10-named-leader split we actually ship with, and (3) compiles to batched GPU dispatch. Every serious objection — "the observation is too big", "the action vocabulary is too wide", "backend polymorphism fragments batches", "slot observations lose graph structure" — is answerable from the state catalogs and the v2 audits, without the proposal moving from its core shape. The right extensions are around *versioning*, *training data emission*, and *lazy slot evaluation* — not reshaping the primitives.

## Why the rich observation is correct

### The simulation's state *is* the decision signal

`state_npc.md:209-329` enumerates the NpcData container as roughly 60 decision-relevant fields. Every one of them is read by a named system the v2 audits either preserve (as essential physics) or promote to an agent-chosen action. A policy that does not see these fields cannot reproduce the behavior those systems currently encode.

Concretely, here is a non-cherry-picked sample of decisions an NPC must be able to make after the v2 reframing, and the observation features required for each — traced back to `state_npc.md` line ranges and v2 analyses:

1. **`Repay(creditor_id, amount)` vs `Default(loan_id)`** (`systems_economic.md:210-217` loans) — requires: `gold_log`, `debt_log`, `income_rate_log`, `credit_history`, `creditor_id`, `memory beliefs of EntityDangerous/EntityTrustworthy toward creditor_id`. Without `credit_history` the policy can't model reputation consequences; without `memory.beliefs` it can't model the creditor as an *entity* with a trust relationship; without `income_rate_log` it can't estimate whether `20% of income` is actually payable. All three are right there on `NpcData` (state_npc.md:222-231, 277-293). Omitting any one of them is not a minor loss — it collapses the decision to a random 50/50.

2. **`ProposeMarriage(target)` vs `RebuffAdvances(suitor)`** (`systems_social.md:116-123, 196-200`) — requires: `personality` (`state_npc.md:80-93`), `relationships[target].trust` + `familiarity` + `perceived_personality` (`state_npc.md:156-175`), `spouse_id`, `aspiration.need_vector[social]` (`state_npc.md:115-123`), `needs.social`, `mood` view, `behavior_profile` cosine similarity to the target. A 30-feature combat-style observation *cannot* carry any of this. The `Court` utility in v2 is explicitly `compatibility(a,b) = cosine(behavior_profile_a, behavior_profile_b)` — which is an atom over an arbitrary-length sparse vector on each side, exactly the kind of thing the DSL's `block` + `view::relationship(self, other)` primitives are designed to package.

3. **`DeclareWar(opponent, casus_belli)`** (`systems_politics.md:126-133` warfare) — the v2 reframe makes explicit that "leader chooses; grievances are *inputs* to that choice". The inputs named in the reframe are: `grievance_matrix(a, b)` (derived view over memory beliefs of type `Grudge`), faction treasury, faction `military_strength`, `war_exhaustion` view (derived per `systems_politics.md:54-61`), opponent `diplomatic_stance`, `n_at_war_with`. These are exactly the fields the proposal names in its **Faction context block** (`proposal_policy_schema.md:51-52`): `strength_log, n_at_war_with, n_allies, n_vassals, has_suzerain, war_exhaustion (view), military_strength, diplomatic_stance per neighbor (top-K=4)`. If you believe the v2 politics analysis, you cannot believe in a narrower observation.

4. **`RunCaravan(src, dst, commodity, amount)`** (`systems_economic.md:76-83` trade_goods) — requires the agent's `price_beliefs[c]` per commodity (`state_npc.md:292`), observed nearby settlement inventories, `trade_route_id` / `trade_history`, current `gold`, and its own `inventory` bitmap. The proposal's self-contextual section (~100 floats at `proposal_policy_schema.md:33-34`) carries *exactly* this: `price_beliefs(8)`, `inventory(8 commodities)`, `n_active_contracts`, etc. An NPC without price beliefs cannot do arbitrage; the entire agent-emergent trade economy (v2 economic reduced 30+ systems to `RunCaravan` + folds) collapses.

5. **`PracticeHobby(domain)` selection** (`systems_social.md:79-87`) — requires `behavior_profile` top-K slots (because the DSL's reframe says `hobby_domain(npc) = argmax_domain(behavior_profile)`), `emotions.joy` for hedonic pull, `morale` view, `aspiration.need_vector[purpose]`. The proposal carries all of these.

6. **`Betray(faction)` / `FormGrudge(target)`** (`systems_politics.md:182-189`) — requires `personality.compassion` (betraying is easier for low-compassion NPCs, per the reframe's `treachery_score` utility), behavior tags for `STEALTH` + `DECEPTION`, `faction_loyalty`, `wanted_level`, memory beliefs of witnessed betrayals. The proposal carries `personality(5)`, `behavior_profile top-K (8 slots × 16-bit tag_id_one_hot)`, `faction_id_embed(8), faction_loyalty, role_one_hot(8), wanted_level`. No ad-hoc cruft; each field has a provable consumer.

7. **`SeekRedemption(settlement, amount)`** (`systems_politics.md:162-170` outlaws) — requires `wanted_level`, `gold_log`, `reputation_at_settlement`, memory belief of `SettlementProsperous/Poor` toward target, emotions.grief (the utility is shaped by remorse). Every one of these is in the proposal.

### The alternative is knowingly lossy

The combat-sim observation is ~30 features per unit (see `src/ai/core/ability_eval/game_state.rs`'s `summarize_abilities` — which scans vitals, positions, combat state, abilities, CC, healing). It works because the combat sim has ~8–12 units with 4–6 abilities each and the decision is "which of 14 discrete actions to take". That's an AGGREGATE state size of ~300 floats on 8 units × 30. Scaling to 20K NPCs in a persistent world with 50+ verb types would either force brutal feature compression (losing most of the NpcData signal) or build a separate bespoke encoder per domain. The proposal's answer — declare once in DSL, pack to ~1800 floats, let the backend pick which subset to use — is strictly cheaper and less error-prone than maintaining 5 bespoke Rust encoders in parallel.

### The "144 MB per tick" bogeyman

Yes, `1800 × 4 bytes × 20K NPCs = 144 MB`. At contemporary GPU memory bandwidth (~1 TB/s), that's 144 μs of pure bandwidth — well below the 100 ms fixed-tick budget. On CPU with AVX-2 gather / rayon, packing 20K × 1800 float64s from entity struct-of-arrays has been measured at ~50 ns per NPC for similar workloads (roughly the cost of touching the fields); total ~1 ms. This is negligible. The risk isn't the tensor size; it's that most NPCs won't use most fields on most ticks. That's a real concern — see the **Extensions** section below where I recommend lazy slot evaluation — but it does not justify shrinking the declared schema. The correct answer is "declare wide, materialize lazy, batch by role".

### Supporting both lean and rich

The proposal's DSL is already expressive enough to declare a *lean* observation (a subset block), e.g.:

```
policy CommonerLean {
  observation {
    block self.vitals { from self.hp_pct as f32; from self.hunger as f32; ... }
    slots nearby_npcs[K=4] from query::nearby_entities(self, NPC, 20) { ... }
  }
  default_backend Utility { ... }
}
```

A separate `policy LeaderRich { observation { ... all 1800 floats ... } }`. Same contract, different dimensions. The DSL should make this explicit — **this is an extension, see below** — but the *shape* is right.

## Why multi-head action is correct

### The action vocabulary is not negotiable

Totaling the EMERGENT verbs named across the four v2 docs:

- Economic (`systems_economic.md:362-445`): **~45 verbs** — `LevyTax, PayTax, PaySalary, PayUpkeep, PaySubsidy, PayMaintenance, Produce, Consume, Eat, ConsumeTravelRations, ForgeItem, HarvestTile, HarvestNode, GoToWorkplace, MoveTo, BeginProduction, FinishProduction, Deposit, DepositAtSettlement, PerformTimed, Wait, StartConstruction, PlaceBlueprint, PlaceTile, WorkOnBuilding, ClaimWorkSlot, ClaimResidence, PayUpgrade, BuildRoad, RepairBridge, RunCaravan, TipCommission, SellToSettlement, PurchaseFromStockpile, Gossip, ObserveLocalPrice, UpdatePriceBelief, FundMerchant, SetPriceCeiling, HoardCommodity, PostContract, BidOnContract, AcceptBid, MarkContractFulfilled, PayContract`
- Social: `PursueGoal, PracticeHobby, Court, FlirtAt, RebuffAdvances, ProposeMarriage, AcceptMarriage, RefuseMarriage, ConsumePotion, Train, Teach, OfferApprenticeship, CrownFolkHero, ClaimTrophy, BefriendNpc, DeclareRival, ShareStory, AttendFestival, ...`
- Politics (`systems_politics.md` exhaustively): `DeclareWar, SignPeace, OpenDiplomaticChannel, ProposeTradeAccord, OfferGift, BreakRelations, Spy, Sabotage, InfiltrateCouncil, ArrestSpy, ExecuteSpy, ExileSpy, SetStandingOrder, DeclareCivilWar, JoinFactionSide, FleeCivilWar, SurrenderCivilWar, VoteOnIssue, AbstainVote, TableMotion, LaunchCoup, PledgeLoyalty, BribeGarrison, FormAlliance, BreakAlliance, DisburseAllianceAid, SwearVassal, AcceptVassal, DemandTribute, RemitTribute, Rebel, InvestInTech, PoachScholar, FoundAcademy, IssueProphecy, Recant, BecomeOutlaw, Raid, FormBanditCamp, SeekRedemption, Exile, AcceptRedemption, PostBounty, LeadFoundingExpedition, JoinExpedition, Betray, StealTreasury, FleeAfterBetrayal, FormGrudge, Marry, Divorce, HaveChild, AdoptChild, VoteForSuccessor, DeclareCandidacy, AcceptSuccessor, Abdicate, AdoptCharter, AmendCharter, RevokeCharter`
- World: `BuildTower, LightSignal, ScoutRegion, MigrateToSettlement, Retire, AssumeClass, EnterBuilding, HoldFestival`

Conservative deduplicated count: **~110 distinct verbs**, comfortably in the proposal's stated 50–150 range.

A flat `14-action pointer head` (the V3 self-play architecture memoized in MEMORY.md) cannot express this. Collapsing to 14 means fusing semantically distinct actions (e.g. `Betray` vs `Spy` vs `Sabotage`) into ambiguous umbrella buckets, which destroys credit assignment during learning and makes the utility backend's scoring table unreadable. The v3 pointer action space proved at best 30% HvH win rate on a tiny 14-action vocabulary — scaling it to 110 naively would push it well below random. The proposal's multi-head structure, where the categorical head picks an *action kind* and then the pointer/continuous heads fill in the rest, is how modern game-RL work (OpenAI Five, AlphaStar, DeepNash) shapes similar problems.

### Each head does work the others can't

- **Categorical action kind**: picks the verb — `Attack`, `Trade`, `Court`, `DeclareWar`. 50–150 logits. Masked per-tick.
- **Pointer target**: selects a concrete entity from the union of observation slot tables (`nearby_npcs ∪ nearby_resources ∪ nearby_buildings ∪ active_quests ∪ party_members`). This is essential because actions like `Attack(t)`, `Trade(t)`, `ProposeMarriage(t)` all share the exact same target-selection sub-problem. If you one-hot-encoded target identity you'd inflate the output to `50 × 2000-entity-vocab = 100K logits`, which is infeasible to train. The pointer head over observation slots is O(K) = 12–32 per slot table. This is the same reasoning that pushed combat self-play from the flat V2 action space to the V3 pointer action space (MEMORY.md notes V2 at 54% HvH; V3 at ~30% *only because V3 was retrained from scratch on 110K params with noisy REINFORCE*, not because pointers were wrong).
- **Continuous magnitude**: `PayTax(amount)`, `OfferGift(target, amount)`, `SwearOath(stakes)`, `RunCaravan(src, dst, c, amount)` all take a scalar. One-hot-discretizing "amount" into 10 bins is the worst of both worlds: coarse-grained where you need precision (small tribute payments) and overprecise where you don't. A 1-D continuous head with sigmoid-scaled output is the textbook fix.
- **Continuous pos_delta**: `MoveToward` / `Migrate` / `Flee` want a 2-D direction vector. Again, textbook.

### Rare actions

The objection: "DeclareWar fires ~once per faction per 1000 ticks; the categorical head has 50+ logits, 49 of which are masked most of the time; training class balance is terrible." Three responses:

1. **Masking + sampling handles balance correctly**. If the mask says only 3 actions are valid on a given tick, you only compute loss over those 3 logits' softmax (standard for masked multinomial policies). The 47 invalid logits don't contribute gradient noise or class-imbalance loss.

2. **Rare actions are the *right* scale of decision for LLM backend**. The proposal's backend polymorphism already says `for_role(Leader) backend Llm`. A faction leader considers `DeclareWar` rarely; when they do, the LLM's reasoning latency is amortizable. This is not a bug — this is the intended mapping: rare + high-stakes → expensive backend.

3. **Macro/micro split is compatible**. If rare-action training instability *does* bite in practice, you can always add `policy Macro { observation { ... } action { DeclareWar | FormAlliance | ... } }` fired every 500 ticks alongside `policy Micro { ... move/attack/work/talk ... }` fired every 5 ticks. This is a policy-*decomposition*, not a schema change. The DSL already supports multiple named policies (the proposal shows `policy NpcDecision`, implying others exist). See position on Q5 below.

## Why backend polymorphism is correct

### The infrastructure already exists

- **Burn neural runtime**: already in `Cargo.toml` dependencies (per the AI module `src/ai/core/ability_transformer/weights.rs`). The combat sim's `ability_transformer` already loads exported weights and runs forward passes at scale — 96K parameters, <0.002 Python↔Rust agreement per MEMORY.md. The infrastructure for "evaluate a neural network per NPC per tick" is proven production code.
- **LLM NDJSON bridge**: `src/bin/sim_bridge/` is a standing headless-sim binary that an external agent (e.g. `~/Projects/lfm-agent` with LFM2.5-1.2B via vLLM) connects to. The sim_bridge protocol is the same shape as what the proposal's `backend Llm { endpoint: "ndjson://..." }` specifies. No new protocol needed.
- **Utility scoring**: the existing `action_eval.rs` utility engine is exactly the proposal's `default_backend Utility { /* declarative scoring rules per ActionKind */ }`. It works today at 20K NPCs, ~7.9% of tick time per `systems_social.md:38`.

### One-size-fits-all is the wrong tradeoff

Cost per decision:
- **Utility**: ~1 μs (table lookup + weighted sum). 20K NPCs × 0.2 Hz avg = 4K decisions/sec × 1 μs = 4 ms/sec. Trivial.
- **Neural (Burn, 96K-param transformer)**: ~100 μs per NPC single-batched; ~1 μs amortized in a batch of 200. 1K named-hero NPCs × 0.2 Hz × 1 μs = 200 μs/sec. Trivial.
- **LLM (LFM2.5-1.2B via vLLM)**: ~50–200 ms per decision at batch=1; ~10 ms at batch=16. 10 leaders × 0.01 Hz (decision every 100 seconds of game time) × 10 ms = 1 ms/sec. Also trivial.

Total: ~5 ms/sec of compute across 20K agents, using three different decision engines matched to the stakes of the decision. Running LLM on all 20K would be ~200 s/sec — infeasible. Running utility on the 10 leaders would lose the reasoning/planning/dialog capability the LLM is there for. The polymorphism is not a convenience; it's the only way the math works.

### "Small batch sizes for the LLM"

Yes, batch of 10 leaders per tick is small. But:

- **Leaders decide at low cadence**. Per the v2 politics analysis, faction-leader decisions (`DeclareWar`, `SignPeace`, `FormAlliance`) happen on the order of every 1000 ticks. With 10 leaders × 1/1000-tick cadence × 100-ms tick = one decision every 10 seconds of wall-clock time, total. Batch size 1 is fine at this rate; vLLM's TTFT on 1.2B params is ~100 ms.
- **Commoner "medium-cost" decisions** can use the neural backend at hero-class batch sizes (100–200 per tick), amortizing well.
- **Utility backend batches trivially** across 19,000 commoners with rayon parallelism.

The proposal's phrase "three batches" (`proposal_policy_schema.md:198`) is correct and each batch is sized appropriately for its engine.

### The trait contract is minimal

The proposed `PolicyBackend::evaluate_batch(observations, masks) → ActionBatch` is small enough to be trivially mockable for tests, trivially swappable for A/B experiments (see `twin_run_harness.md` in MEMORY.md — the existing oracle vs non-oracle twin-run already works this way for combat), and trivially composable for learning (the BC→REINFORCE pipeline in the ability_operator crate shows the pattern).

## Why slot-based + padded is correct

### Top-K-by-distance is already the universal access pattern

Every hot path in the current code does top-K spatial query:

- `action_eval.rs` builds `TypedSnapGrids { resources, buildings, combatants }` and scores top-K candidates per NPC (per `systems_social.md:37-42`).
- `agent_inner.rs` `update_known_resources` caches the top-K nearest resources per NPC (`state_npc.md:277-280`).
- Combat engagement uses top-K enemies-by-distance-then-priority (`src/ai/squad/intents.rs`).
- The social systems in v2 almost universally gate on "within SOCIAL_RANGE" and then score top-K candidates.

Declaring `slots nearby_npcs[K=12] from query::nearby_entities(self, kind=NPC, radius=50) sort_by distance` isn't a new abstraction — it's formalizing the abstraction the code has been using unsystematically.

### Padding + `exists` is the ML-standard

Every transformer / attention / RNN architecture working on variable-length inputs solves this the same way: pad to max K, set an `exists` (attention mask) bit, zero out padded positions. The ability transformer in `src/ai/core/ability_transformer/` already does this with up to 9 abilities per entity (MEMORY.md: "Ability LUT: 1452 entries × 9 abilities × 32d pre-computed [CLS] embeddings"). Nothing new.

The alternative — graph attention with explicit adjacency — is more expressive but:

1. **Unnecessary at present**: most decisions are "who's close to me?", not "what's the three-hop influence chain through the social graph?". Graph attention would be overkill for `Attack(nearest_enemy)`.
2. **Existing tech we already spent budget on**: the cross-attention in `ability_transformer` and `ActorCriticWeightsV3` is slot-based with attention over padded tokens. It works. Pivoting to graph attention is a separate research project.
3. **Compatible with future extension**: the DSL's slot primitive can later be augmented with explicit adjacency tensors if a new policy needs them. The slot API is the shared substrate.

### Graph-structured reasoning at the macro layer

I concede one refinement: **for faction-level reasoning** (the "macro" policy at leader cadence), slot-based top-K-by-distance is the wrong shape. A leader evaluating `FormAlliance(f)` wants to see the full faction graph (all neighbors, their alliances, their wars). The right answer is **a separate macro policy with graph-structured observation** — not a reshape of the per-NPC observation the proposal specifies. The proposal already permits multiple policy declarations via backend assignment per role. I'd make this explicit in the schema (see extensions). The *micro* policy — which is 99% of NPC-ticks — is correctly slot-based.

## Positions on the 10 open questions

### Q1. K=12 nearby NPCs vs K=20

**Recommend K=16 by default, configurable per policy.** Reasoning: 12 loses ~30% of "nearby" NPCs in dense settlement interiors (settlement capacity is ~300 per `state_aggregate.md:181`; a town square scene easily has 20+ NPCs in sight). 20 is fine but adds 208 features (26 × 8), non-trivial for the commoner-lean policy. K=16 balances. **For Leader-role LLM policy, raise to K=32**; an LLM can summarize 32-entity context and the information loss from clipping is a real decision-quality hit.

### Q2. Sorted vs unsorted slots

**Sort by distance for Utility backend; unsorted (attention-sorted) for Neural backend.** This is not a contradiction — the proposal's DSL can emit both packings, and the backend trait chooses which view to read. Reasoning:

- **Utility backend** needs stable slot indices to write scoring rules ("if `nearby_npcs[0]` is hostile and within `AGGRO_RANGE`, prefer Attack"). Unsorted would mean every rule needs a `argmin(slot.distance)` preamble, which is fine but shifts work to authors.
- **Neural backend** doesn't care about input order in theory — self-attention is permutation-invariant by construction. But empirically (MEMORY.md notes on entity encoder v3: "Fixed data pipeline (abilities scanned from DSL effects, not zeroed legacy fields)"), consistent sort order reduces distribution shift between training and inference. Even for neural, a sorted-by-distance input is the safest default.

Emit both shapes in the DSL compiler; backend chooses.

### Q3. Eager vs lazy observation construction

**Lazy per-NPC with eager batching inside a role-backend group.** Most NPCs at Low fidelity (per `state_world.md:261-275` FidelityZone) don't need a 1800-float observation every tick — they need a truncated 30-float lean observation at low cadence. The correct pipeline:

1. Tick-level: iterate all NPCs by fidelity zone.
2. For each (fidelity × backend) bucket, compute observations *only for NPCs that will actually make a decision this tick* (check backoff timer + mask-is-non-empty).
3. Pack packed observations for that bucket's NPCs into a single contiguous tensor.
4. Dispatch the backend once on the batch.

This is lazy *at the NPC level*, eager *at the batch level*. Saves 60–80% of the tensor-building work on idle NPCs while preserving GPU-friendly batching. The proposal's DSL supports this because packing is a compiler-emitted kernel per-field — the compiler can emit both eager and lazy variants.

### Q4. Per-policy vs per-entity backend assignment

**Per-role default with per-entity override.** This matches the proposal's current shape (`for_role(Hero) backend Neural; for_role(Leader) backend Llm`) and is the right model:

- Role is a stable entity property (changes at most a handful of times per NPC life). Per-role default is free.
- Per-entity override handles the named-character case: the player character, legendary NPCs, prophets named by the chronicle. These are <50 entities; a HashMap<entity_id, backend_override> is negligible.
- Per-policy-default is a sub-case of per-role (since policies are keyed by role).

Do NOT allow mid-tick backend switching — it would shatter batching.

### Q5. Rare actions handling

**Use macro/micro decomposition. Do not bury rare actions in the main categorical head.** Concretely:

```
policy NpcMicro {
  observation { ... rich per-NPC ... }
  action { categorical kind: { Attack, Flee, Eat, Work, Talk, Trade, MoveToward, ... ~40 common } }
  cadence: every 5 ticks
}

policy LeaderMacro {
  observation { faction context + neighbor slots + chronicle summary }
  action { categorical kind: { DeclareWar, FormAlliance, AdoptCharter, ... ~20 rare-but-stakes } }
  cadence: every 500 ticks
  for_role(Leader) backend Llm
}
```

Mask is still important in each but the class imbalance problem is contained. The macro policy's 20 logits are rarer but comparable in frequency *within that policy's tick budget*. This is the standard hierarchical-RL approach (feudal/options framework) and it composes cleanly with the proposal's existing role-based backend dispatch.

### Q6. Conditional vs always-emitted action heads

**Always emit all heads; mask unused.** Reasoning:

- Conditional emission (e.g. "only run the pointer head if the categorical pick is `Attack`") requires dynamic control flow in the policy, which breaks GPU batching. Every NPC in a batch must run the same kernel.
- The cost of running the pointer and continuous heads when unused is tiny — a few hundred FLOPs per NPC. Negligible compared to the transformer encoder.
- Masking invalid heads is standard. For learning, set the loss weight for unused heads to zero per sample. Done.

This is also the architecture `AbilityActorCriticV3` uses today (MEMORY.md) — the base head always emits 6 logits even when only 2 are valid.

### Q7. Observation versioning

**Emit a schema hash + migration table at DSL compile time.** This is one of the proposal's weakest spots and needs explicit attention. Concretely:

- DSL compiler emits `observation_schema_v{N}.json` with field offsets, types, and a fnv-1a hash of the canonical serialization.
- Weights file includes the schema hash it was trained against.
- At load time, if schema hash mismatches, either (a) fail loudly, (b) load a declared migration rule (zero-pad missing fields, drop removed fields), or (c) fall back to the utility backend.
- Every field in the DSL has an `@since v{N}` tag so the compiler can compute forward-compatible migrations.

This is not exotic — `.ability` files in `src/ai/effects/dsl/` already have versioning via the parser. Extending the same mechanism to policy schemas is a ~1-week job.

### Q8. Reward signal placement

**Per-policy reward in the DSL; world-level aggregate reward derived.** Reasoning:

- NPC-level reward should be expressible declaratively: `reward = Δ(self.gold) + Δ(self.esteem) × 0.1 + Δ(self.hp) × 10 + Δ(self.reputation) × 5`. The DSL's view primitives (`view::mood`, `view::reputation`) already compose into this.
- Per-policy reward lets different roles optimize different things. A Hero maximizes quest completion; a Leader maximizes faction `military_strength` and stability; a Merchant maximizes cumulative gold.
- World-level aggregate (e.g. "total chronicle events generated per 10K ticks" as a playtest metric) can be derived from the DSL's reward block — but is not the training signal.

Place the `reward { ... }` block inside the policy declaration, next to `observation` and `action`. Compiler emits a reward-computation kernel parallel to the packing kernel.

### Q9. Multi-NPC collective policies

**Reduce to per-agent decisions with coordination via events. Do not introduce multi-agent policies natively.** Reasoning:

- Every v2 "multi-agent" decision (marriage, alliance, civil war, expedition formation) already decomposes into matched pairs/chains of single-agent actions (`ProposeMarriage + AcceptMarriage`, `FormAlliance + RatifyTreaty`, `LeadFoundingExpedition + JoinExpedition`, `DeclareCivilWar + JoinFactionSide`). This pattern is *pervasive* in the audits — it's the natural shape of the reframing.
- Multi-agent policies in the RL literature (CTDE, MAPPO, etc.) carry major engineering tax and training instability. Worth it for competitive games; not worth it for a world sim where the coordination is event-mediated anyway.
- The one exception where CTDE might be appropriate: party combat (4–8 heroes on a quest acting in tight coordination). This can be layered *later* as a party-level policy analogous to the existing combat self-play, without changing the per-NPC DSL.

### Q10. Curriculum / role progression

**One policy per role with a shared observation schema; per-role action subset.** Concretely:

- The observation schema is shared across roles (no feature is role-specific — all come from NpcData fields every NPC has).
- The action space is a *subset* per role (Commoner can't `DeclareWar`, Leader can; Commoner can `Eat`, Monster can `Attack`).
- When an NPC changes role (`commoner → adventurer → leader`), the policy backend swaps but the observation packing does not.
- For training curriculum: start with commoner-role utility backend (known-good baseline from current sim), progressively train hero-role neural backend on imitation of utility, progressively train leader-role LLM on chronicle-based dialogue data.

This avoids the "role-conditioned masking on one monolithic policy" approach, which would force a single model to learn wildly different behavior regimes jointly — historically hard.

## Extensions you'd add

The proposal is right but doesn't go far enough on these points. Each is a concrete extension, not a reshape.

### 1. Schema versioning (reiterating Q7)

Add `@version` attribute per field in the DSL; compiler emits a schema manifest + migration rules. Essential the moment we train our first neural backend and commit to a weights file.

### 2. Training data emission

Every policy tick should optionally log `(observation, action, mask, reward)` tuples to an append-only parquet/ndjson stream — gated by a `training_capture: true` flag on the policy. This is how you get a replay buffer for offline RL without instrumenting the sim post-hoc. The proposal barely mentions training; this is a ~200-line addition to the compiler and gives you the dataset pipeline for free.

### 3. Lazy slot evaluation

Extend the slot primitive so that *derived-view atoms* inside a slot (e.g. `view::relationship(self, other).trust` for slot `other`) are only computed when the *using* backend actually reads them. Utility backends that score only the top-3 candidates shouldn't force computation of 12 slot-levels of relationship data. Implementation: slot atoms compile to thunks; backend requests unpack them. 2–5× speedup in the utility-dominated mass-NPC case.

### 4. Batched dispatch with rayon for Utility

The proposal says `evaluate_batch([N, OBS_DIM])` but doesn't specify parallelism. For utility backend with `N = 20K`, rayon `.par_chunks()` over the obs tensor is a 4–8× wall-clock speedup on modern multi-core. Spec it.

### 5. Per-backend observation subsetting

Leader LLM doesn't need all 1800 floats — it needs the *summary*: self-role, faction strength, war status, neighbor stance. An 80-float summary serialized to 200 tokens of JSON is a more natural LLM input. Compiler should emit a "compact projection" per backend, declared alongside the full schema. The `block` primitive already supports this (`block self.psychological { ... }` is a composable sub-schema); formalize which blocks each backend reads.

### 6. Observation debug tooling

A `decode_observation_bytes(obs: &[f32]) → serde_json::Value` inverse function, auto-generated from the schema. Essential for debugging mis-trained policies and for the `sim_bridge` NDJSON protocol (LLM agent should see human-readable observation, not raw floats).

### 7. Coordination primitives

For the rare multi-agent cases (party quest, siege formation), extend the action space with a `broadcast: bool` attribute — an agent emitting a `PartyAction(plan)` creates a shared event readable by all party members on the same tick, without introducing a multi-agent policy. Cheap, preserves the single-agent-decision architecture, covers 90% of coordination use cases.

### 8. Fidelity-tiered observation depth

Observation depth should vary by FidelityZone (`state_world.md:261-275`). High-fidelity NPCs get the full 1800-float observation; Low-fidelity NPCs get a 50-float summary. This is a schema-level concern the proposal glosses over. The DSL should support `observation @fidelity(High) { ... full ... } observation @fidelity(Low) { ... summary ... }`.

### 9. Cross-policy observation sharing

The settlement-context block (`treasury_log, food_status, threat_level, ...`) is identical across all NPCs in a settlement. Pack it once per settlement, reference per-NPC. The compiler should de-duplicate these and emit a gather-scatter kernel. Straightforward optimization; the proposal leaves it implicit.

### 10. Reward shaping hooks

Per-policy reward should support per-tick shaping bonuses (e.g. `+0.1 × is_in_combat` to speed up combat-learning early in training). The DSL should make these explicit and logged — see MEMORY.md on `--reward-shaping SCALE` in the transformer-rl pipeline; same mechanism, but declarative.

## Risks I accept

The proposal lists six risks. Each is real; none is disqualifying.

### Risk 1: "Over-engineering — 144 MB/tick observation tensor"

Already addressed above. Packing cost is ~1 ms on CPU, ~100 μs on GPU, well below tick budget. The more subtle concern — that most NPCs don't read most features — is addressed by lazy slot evaluation (Extension 3) and fidelity-tiered depth (Extension 8).

### Risk 2: "Action vocabulary explosion"

50–150 verbs isn't an explosion, it's the natural count from the v2 audits. Class imbalance is addressed via mask-aware softmax and macro/micro policy decomposition (Q5). Utility backend is fine with a 150-verb categorical because it's just a scoring table — no learning required. Neural backend handles 150-way categorical with standard softmax + class-balanced loss; well-precedented in AlphaStar (100+ action types) and OpenAI Five (~170K action space via pointer-attention).

### Risk 3: "Slot-based doesn't handle graph-structured data"

Genuinely true for macro/faction-level reasoning. Solution: separate macro policy with graph observation (Q10 + extension). Micro policy (which covers 99% of NPC-ticks) is correctly slot-based.

### Risk 4: "DSL complexity"

The DSL described in `proposal_policy_schema.md:123-170` is ~40 lines of schema for a full policy declaration. The equivalent Rust packing code, conservatively estimated from the `build_observation_for_policy` pattern in the ability_eval module, is ~400 lines per policy × 3 policies = 1200 lines to hand-maintain. DSL is a 10× reduction and a single source of truth. The risk is real only if we end up with 1–2 policies total; at 4+ (Commoner, Hero, Leader, Monster, plus role-specific variants) the DSL pays for itself trivially.

### Risk 5: "Backend polymorphism — small batches"

Addressed above: leader batches are small *because there are only 10 leaders*, and leader decisions are rare, so LLM throughput is not a bottleneck. Commoner utility scales to 20K trivially. Hero neural batches comfortably at 200.

### Risk 6: "State materialization conflicts with event-sourcing purity"

This is the most philosophically uncomfortable of the six, and I concede it partially. The event-sourcing purity goal in the v2 docs is real — `systems_economic.md:313-358` shows 30 of 32 systems collapsing into "action vocabulary + event fold index." For observation purposes, we do need materialized views (mood, trust, price_beliefs) because walking the event log every tick to recompute a policy's input is prohibitive.

But this is fine: the event log is the source of truth, and the materialized caches are regenerable from it on cold load. The event log doesn't need to be the *only* representation — it needs to be the *canonical* one. The observation layer reads from caches that the event fold writes. This is standard CQRS (command-query responsibility segregation). The proposal is compatible with this; the v2 reframing's claim is about *systems that currently mutate state* collapsing into event-fold views, not about the observation reading from a live cache.

## Closing argument

The proposal is the right shape because every one of its four commitments maps 1:1 to something the world-sim *actually needs*: rich observation because NpcData is decision-rich; multi-head actions because the 110-verb vocabulary needs shape; polymorphic backends because utility/neural/LLM have 1000× cost differences that the agent population structure exploits; slot-based observation because every hot path in the code is already doing top-K-by-distance queries unsystematically.

The objections are all tractable. Observation size is a bandwidth non-issue. Rare actions resolve through masking and macro/micro decomposition. Batch fragmentation is a feature, not a bug — it matches decision cadence to decision cost. Graph-structured reasoning is a macro-layer addition, not a rewrite.

The extensions I propose — schema versioning, training data emission, lazy slot evaluation, fidelity-tiered depth, coordination primitives — are all strictly additive. None reshape the core primitives. The proposal as written is the right foundation; these are the right next steps to make it shippable.

Approve with extensions. Build.
