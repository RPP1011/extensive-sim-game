# Policy Schema Proposal — Attacker's Brief

## Executive position

The proposal commits the world-sim to a schema whose costs are load-bearing assumptions (1500–1800-float observations, ~50–150-verb multi-head action space, three swappable backends, GPU-targetable DSL compilation) while its central claim — that this is the right "AI plug" — rests on work we haven't done yet. The attacks that follow, in order of strength: (1) the observation budget bakes in every current state field as a ML-training contract before anyone has evidence of which features matter; (2) the ~50–150-verb action vocabulary is the output of a paper exercise (`systems_*.md`) that has never been empirically verified to reproduce gameplay outcomes; (3) slot-by-distance observation is the wrong abstraction for the political/economic decisions that make up most of the vocabulary; (4) "backend polymorphism" is three simulators in a trench coat — we pay the cost of a unified contract but can only practically ship one backend; (5) event-sourcing purity collides with rich observation and the proposal hand-waves the resolution; (6) the "GPU-targetable" claim is scoped too broadly for the primitives it proposes; (7) the DSL surface is premature at one or two policies. The proposal should be split: commit to a minimal Utility DSL now with a clean, versioned observation façade; defer the neural/LLM abstractions, the packed-tensor eager construction, and the 150-verb action head until the Utility version proves what the policy actually needs to see.

Line numbers below refer to `docs/ecs_dsl/proposal_policy_schema.md` unless noted.

---

## Attack 1: The observation budget is an unforced contract against ML weights that don't exist

> "Per-NPC packed feature tensor. Total budget: ~1500–1800 floats per NPC." (line 17)
>
> "Rich per-NPC state: NpcData carries ~50 fields' worth of decision-relevant signal … The policy must see all of it; combat-style ~30-feature observations throw away most of the sim." (line 10)
>
> "Over-engineering: 1500–1800 floats per NPC × 20K NPCs = 144 MB observation tensor per tick." (line 193)

### The claim the proposal makes

Premise 4 asserts that because `NpcData` *holds* ~50 fields, the policy *must see* all of them, and that combat-style ~30-feature observations (the current `src/ai/core/ability_eval`) "throw away most of the sim." The conclusion is a 1500–1800-float per-NPC tensor (Components 1 of the schema, lines 17–54) whose exact composition — slot widths, block sizes, one-hot widths — is fixed in the DSL declaration and, by the proposal's own admission at line 186 ("Once trained ML weights exist, the observation schema is a contract. Adding a new feature requires either retraining or padding with zeros"), becomes a binding contract the moment we train anything.

### Why the premise doesn't support the conclusion

The inference from "NpcData stores X" to "the policy must see X" is unjustified. The simulation state contains vast amounts of data that are operational caches, not decision signal:

- `cached_path: Vec<(u16, u16)>` and `path_index: u16` (`state_npc.md` lines 246–247): pathfinding breadcrumbs. No decision reads them; movement follows them.
- `grid_id: Option<u32>`, `entity_index`, `group_index`, `settlement_index` (`state_aggregate.md` lines 374–379, `state_world.md` lines 32–34): all tagged `#[serde(skip)]` and explicitly labelled "rebuilt on load" or "derived from primary state." These are O(1)-lookup accelerators. A policy that observes `group_index` has observed nothing.
- `worker_class_ticks`, `builder_modifiers`, `owner_modifiers` (`state_npc.md` lines 358, 372–373): book-keeping used to *compute* specialization bonuses. The decision signal is the specialization itself, not the tick counter it was folded from.
- `perceived_personality.observation_count`, per-trait `confidence` arrays (`state_npc.md` lines 168–170): meta-cognitive bookkeeping that controls how the NPC's model of *another* NPC updates. It is not a feature of *this* NPC's decision.

The proposal's Component 1 nevertheless pulls several of these in (e.g., `mentor_lineage_depth`, `faction_id_embed(8)`, `credit_history`, `n_children`, cross-faction `diplomatic_stance` top-K) without articulating a single decision that uses them and couldn't be answered better by a query on demand.

### The training-contract problem compounds this

At 1500–1800 floats, the observation schema is not just a Rust type; it's a versioned ML artifact. The proposal admits this in passing at line 186 ("Should the DSL emit a schema hash + migration support?") and then defers it to "open design questions." That's the wrong category — it's the most expensive engineering decision in the entire proposal. Every schema change is either:

1. A retraining run. If we ship the Neural backend per Component 4 (line 106: `backend Neural { weights: "npc_hero_v3.bin", h_dim: 128 }`), every observation edit invalidates weights we've paid training cost for. In the combat-sim analog (per `CLAUDE.md`, Entity Encoder v3 + Ability Transformer v2) each iteration took tens of thousands of training steps and hand-tuned hyperparameters (`training/pretrain_entity.py`, `scripts/curriculum_d128.sh`). Amortize that across ~50–150-verb action heads and a world-sim-scale per-NPC context and retraining becomes the dominant ongoing cost.
2. A pad-with-zeros compatibility layer. This is what happened in the combat transformer pipeline twice already: the v2 `ability_transformer_weights_v2.json` and the later embedded-encoder `ability_eval_weights_v3_embedded.json` — each introduced strict padding conventions we now maintain by hand. Doing this at 30 features for ~40 ability slots is tedious; doing it at ~1800 features for every NPC in a 20K-NPC world is a new kind of tech-debt.

### The sparsity argument the proposal dismisses

At line 193 the proposal acknowledges "144 MB observation tensor per tick … affordable but non-trivial; if much of it is wasted (most NPCs aren't using most actions), we're packing data nobody reads." It then files this under "Risks" and does not address it. It should address it — a hostile reader can observe that most of the action vocabulary in `systems_*.md` is rare:

- `DeclareWar`, `FoundSettlement`, `LaunchCoup`, `CommissionGreatWork`, `AdoptCharter`, `Abdicate`, `IssueProphecy` — all per-faction-leader, low-cadence actions (see `systems_politics.md` lines 418–435). Maybe 50 NPCs can take these at all, and most of them fire maybe once per 1000 ticks.
- `ProposeMarriage`, `AnnulMarriage`, `BetrayFriend`, `Forgive` — once-per-lifetime actions (`systems_social.md` lines 399–408). The `faction_id_embed(8)` and `diplomatic_stance per neighbor (top-K=4)` features (lines 51–52 of the proposal) are only relevant to actions on the order of "per decade."
- The 6 emotion floats plus 5 personality floats exist at all times but most actions — `Eat`, `Harvest`, `Deposit`, `MoveTo`, `Wait` — don't meaningfully depend on them.

The proposal packs ~80 "Context blocks" (lines 49–54) assuming they're uniformly relevant; in practice they're relevant to the ~5% of the action vocabulary that actually operates at that scope. We pay the packing cost for every NPC every tick regardless.

### The budget breakdown doesn't add up the way the proposal claims

Add up the proposal's own numbers (lines 17–54):

- Self atomic (~50) + Self contextual (~100) = 150
- Slot arrays: 312 + 104 + 90 + 96 + 252 + 66 + 66 + 40 + 70 + 80 = **1176 floats** in slot arrays
- Context blocks (~80)

Total: 150 + 1176 + 80 = **1406 floats**, not 1500–1800. The proposal gives itself a ~25% padding margin to grow the schema, which is realistic. But 80% of the total budget is in slot arrays, of which `nearby_npcs` (312), `recent_memory_events` (252), `nearby_buildings` (90), and `threats` (96) alone sum to **750 floats — 53% of the budget.** Those four slot tables are the decision-relevant subset for combat-and-immediate-environment decisions. Every other slot table is tuned for rare actions and contributes dead weight to the common case.

Put another way: the proposal's own numbers admit that a tight schema focused on the common decision surface is ~750 floats plus the ~150 self features — about 900 total, not 1500–1800. The extra 600 floats are speculative capacity for rare political/economic actions that may never train. That's not a schema; that's a blast radius.

### Concrete alternative

Define **decision-relevant subsets per action class**, with the observation constructed per decision rather than per-entity-per-tick:

```
observation_class combat_decision {
  from self: hp_pct, shield_pct, attack_damage, attack_range, status_effects
  from slots[K=8]: nearby_npcs sorted by distance (vitals + threat only)
  // ~40 floats
}
observation_class economic_decision {
  from self: gold_log, inventory, income_rate_log
  from query: nearest_settlement.prices, stockpile summary
  from slots[K=4]: active_contracts
  // ~40 floats
}
observation_class political_decision {
  from self: faction_loyalty, role_one_hot, reputation_at_settlement
  from query: faction graph view (not slot-distance), diplomatic_stance
  // ~60 floats
}
```

Policies declare which classes they consume. The packing kernel only produces what's needed, gated by the action mask. A commoner in `Idle` state packs ~20 floats; a faction leader deliberating `DeclareWar` pulls the political decision set on demand. The expensive observation is rare; the cheap observation is common. The current proposal has it exactly backwards.

---

## Attack 2: The action vocabulary is a paper exercise, not a verified contract

> "Agentic-action emergence: ~70 of the 158 systems can be replaced by NPCs choosing actions. The DSL provides ~50–150 verb action vocabulary; system-level state changes happen as a consequence of NPC choices being applied as physics events." (line 8)
>
> "The agent vocabulary is large (~50–150 verbs) and reflects the agentic-emergence reframing of systems." (line 206)
>
> "Action vocabulary explosion: 50–150 verbs in the categorical head means the categorical distribution has many low-probability outputs. ML training may struggle with class imbalance; utility backend can ignore but mask must be correct or sampling breaks." (line 194)

The proposal's action space (Component 2, lines 57–78) and the "agentic-action emergence" premise depend entirely on the v2 system analyses (`systems_*.md`). Those analyses should not be taken at face value.

### What the v2 docs actually did

Read `systems_economic.md` carefully. Its "Methodology" (lines 5–14) sets up a strict rubric: "the default answer is 'this could be an NPC action, so it's emergent.'" The rubric's output is predetermined by its input — a system is EMERGENT if there is *any way* to reframe it as an NPC choice. Applied consistently, it produces the conclusions we see:

- `trade_goods.rs` (price formula + caravan) → "DERIVABLE-VIEW + EMERGENT" with actions `RunCaravan`, `BuyFromSettlement`, `Travel`, `SellToSettlement` (`systems_economic.md` lines 77–83).
- `bankruptcy_cascade` → "pure function of current treasuries" (`systems_economic.md` lines 192–199).
- `price_controls` ceiling behavior → agent `HoardCommodity` and `BuyUpShortage` (`systems_economic.md` lines 156–163).
- `mercenaries` → `ContractForHire`, `Desert`, `Betray` (`systems_economic.md` lines 246–253).

Each reframing is individually plausible. The problem is that "can be reframed as an NPC choice" is not evidence that "will produce the same gameplay outcomes as the current system." A central planner calling `apply_flat_tax(rate=0.1)` once per 500 ticks reliably collects taxes; an NPC with `LevyTax(rate)` and `PayTax(amount)` as two actions in a 50+-verb categorical head, under a trained policy, may or may not collect taxes depending on its utility weights, its masking, and its training loss. The v2 doc asserts equivalence; it does not verify it.

### Evidence from the combat analog

Our own combat system is the only place we've pushed an action vocabulary through an ML pipeline, and the experience does not support the "just reframe it as emergent" confidence of the v2 docs. From `CLAUDE.md`'s memory index:

- V2 flat action space achieved 54.4% HvH win rate; V3 pointer action space peaked at 30.4% despite having stronger priors, better architecture, and a larger parameter count.
- V3's BC baseline "collapses to hold without engagement heuristic" (MEMORY.md) — even with 97.5% behavior-cloning accuracy on oracle data, the deployed policy picks `hold` so often it needs a hand-coded engagement heuristic (`transformer_rl.rs`) to rescue it.
- "REINFORCE too noisy for pointer target selection (O(N) search space)" — this is precisely the regime the proposal wants to jump into for the political/economic heads where N is all nearby NPCs, all quests, all party members unioned (line 73).

Combat has 14 actions. The proposal wants 50–150 verbs, wider pointer targets, continuous heads, and per-role backend switching — on a sim we haven't built yet. The v2 reframing assumed this would work; the combat evidence says scaling the action space is where the hard problems live.

### The "rare actions" issue is a footnote where it should be a central constraint

Line 184: "Rare actions. DeclareWar happens once per faction per ~1000 ticks. Including it in the categorical head means the model has 50+ logits where 49 are usually masked. Should very-rare actions live in a separate 'macro decision' policy at lower cadence?"

This is framed as an "open question" at the bottom of the proposal. It is actually the question. The action vocabulary isn't a uniform distribution — it's a long-tail mixture where:

- Core loop (≈90% of emitted actions): `Eat`, `Harvest`, `MoveTo`, `Attack`, `Flee`, `Work`, `Deposit`, `Idle`, `Travel`, `ConsumeTravelRations`.
- Medium-frequency (≈9%): `Trade`, `PostContract`, `BidOnContract`, `ClaimItem`, `Gossip`, `Train`, `Teach`, `PracticeHobby`.
- Rare (≈1%): `DeclareWar`, `LaunchCoup`, `FoundSettlement`, `CommissionGreatWork`, `IssueProphecy`, `AdoptCharter`, `SignDemonicPact`.

A single categorical softmax over all of these is statistically poor. The ≈1% tail cannot be learned by REINFORCE-scale self-play; it needs either hand-authored utility rules, oracle supervision, or separate macro-policy scheduling.

### Specific v2 reframings that need empirical validation before being ratified

A partial list, picked to make the point that the v2 docs' confidence is not load-bearing in isolation:

- `bankruptcy_cascade` → "pure function of current treasuries." But `state_aggregate.md` lines 562–565 show a careful numerical model of contagion (`healthy.treasury -= total_loss / n_healthy`) that was presumably tuned to produce specific gameplay dynamics. Replacing a mutation with an on-demand query is mathematically equivalent but *scheduling different*; NPCs observing "my neighbor settlement is insolvent" at different ticks than the hit arrives changes behavior in ways we haven't measured.
- `price_controls` → "Read-time clamp + actions `SetPriceCeiling`, `PaySubsidy`, `HoardCommodity`" (`systems_economic.md` line 162). The original shortage behavior relies on simultaneous stockpile drain — announcing a ceiling and letting residents race to buy before stock runs out. An NPC `HoardCommodity` action is a per-agent decision whose aggregate might or might not produce the same shortage curve. This is a claim about emergent behavior; it requires a simulation to check.
- `mercenaries` → `ContractForHire`, `Desert`, `Betray` (`systems_economic.md` line 249). Whether mercenaries actually desert in the new regime depends on how `Desert` scores against the utility function of an NPC with that action mask. If they never desert (utility too low, risk weights wrong), the "mercenary economy" that was the v1 system's output doesn't emerge.
- `supply_lines` → `RaidCaravan`, `SeizeCargo`, `Patrol` (`systems_economic.md` line 268). The v1 system guaranteed raids happened at some rate via a centrally-applied probability. The new model requires raider NPCs to pick the action freely; if nothing else, the raider NPCs need to *exist* in the right places at the right times, which depends on their movement utility and their faction affiliation. The gameplay output "caravans get raided" could vanish.

The concern isn't that any of these reframings is individually wrong. It's that the policy schema proposal treats the 50–150-verb vocabulary as settled and ships infrastructure for it, before any of these predictions has been verified by running a sim.

### Concrete alternative

Two moves, both cheaper than the proposed schema:

1. **Ship the v1 action set first and measure.** Take the current combat-style ~10–15 action vocabulary plus the "core loop" listed above (~30 actions). Run the sim for long enough to see whether the political/social/economic outcomes the v2 docs *claim* will emerge actually emerge when central-planner systems are removed. Most of the v2 reclassifications predict emergent behavior; test a handful before building for all 150.
2. **Hierarchical action head, not flat.** Use `(macro_class, micro_verb, target, magnitude)` with macro ∈ {combat, economic, social, political}, conditioning the micro distribution on macro. This matches the rare-action structure and is standard in RL literature for exactly this reason. The proposal's Component 2 (line 57) is a single flat categorical; change it.
3. **Commit to the v2 vocabulary incrementally with a "graveyard" registry.** When a verb is added to the vocabulary, require a scenario in `scenarios/` or `dataset/scenarios/` that exhibits the expected emergent behavior with that verb present. If the scenario doesn't produce the expected outcome, the verb goes in the graveyard and doesn't ship. This turns the vocabulary from a paper list into an empirically validated one.

---

## Attack 3: Slot-by-distance drops the relationships that matter for most of the proposed action vocabulary

> "`nearby_npcs[K=12]` sorted by distance, padded: 26 features × 12 = 312 floats" (line 38)
>
> "Slot arrays — variable-cardinality nearby/known data (~1100 floats)" (line 36)
>
> "The pointer head selects an index into the observation slot tables — no entity-id translation needed, the observation already exposes the relevant entities. Categorical action determines which slot table (or 'none') is the valid target source for that action." (line 80)
>
> "Slot-based observation may not generalize to graph-structured social dynamics (faction networks, kin chains). Top-K-by-distance loses information about distant-but-relevant entities (a faction leader the NPC hasn't seen but knows of)." (line 195)

### Where slot-by-distance works, and where it doesn't

For combat and immediate-environment decisions — `Attack(nearest_enemy)`, `Flee(away_from_threat)`, `Harvest(nearest_resource)`, `Eat(food_source)` — sorted-by-distance slots are fine. This is the regime the combat policy operates in and it's well-matched.

For the ~70% of the v2 action vocabulary that isn't combat, slot-by-distance is the wrong shape.

### Concrete examples the proposal's own schema cannot represent

1. **`SwearVassal(lord)`** from `systems_politics.md` line 422. The target is a faction leader the NPC may never have been within 50 world-units of. The pointer head (line 73) selects from `nearby_npcs ∪ nearby_resources ∪ nearby_buildings ∪ active_quests ∪ party_members` — none of these contain "the leader of a faction my faction could theoretically swear to." The proposal's response is `faction_id_embed(8)` and `diplomatic_stance per neighbor (top-K=4)` in the Context block (lines 51–52), but those are *faction-level* scalars; there is no entity handle for the individual leader the NPC is swearing to. You cannot write `SwearVassal(t)` if `t` is out of slot range.

2. **`ProposeMarriage(t)`** from `systems_social.md` line 399. Mask condition (proposal line 91): `t.team == self.team ∧ distance(t) < SOCIAL_RANGE`. The mask *forces* targets to be nearby, which kills the political-marriage pattern — a princess in one settlement marrying a prince in another — that the rest of the politics doc gestures at. The in-world behavior quietly reduces to "marry a villager who happens to walk by," which is *Dwarf Fortress* at best and less than that at worst.

3. **`Gossip(partner, topic)` with `topic = PriceReport(settlement_x, commodity_c)`** from `systems_economic.md` lines 150–154. Price belief updates are inherently graph-structured: agent A hears from B about settlement X's price; A's price model now includes X. Slot-based observation gives A a list of 12 nearby NPCs. It does not give A a data structure over "settlements A has ever heard a price for." The proposal punts this to `price_beliefs(8)` in the Self block (line 33), one value per commodity — which is an aggregation over settlements and throws away the source. If the policy wants to decide "should I trust this report more than the one I heard yesterday?" the slot structure provides no handle.

4. **`PostBounty(target, reward)`** on an outlaw who fled the settlement — `systems_politics.md` line 431. By definition the target is not nearby. The proposal's `threats[K=8] sorted by time_to_impact` (line 41) doesn't apply either — an outlaw far away isn't a time-to-impact threat.

5. **`ContributeToGreatWork(project, amount)`** — `systems_politics.md` line 413. A great work is a multi-year multi-actor project; the NPC contributes gold now with a payoff in emergent chronicle events at completion. There is no slot for "great works I could contribute to." `active_quests[K=4]` is a quest-specific structure.

6. **`RunCaravan(src, dst, commodity, amount)`** — `systems_economic.md` line 82. The interesting case is arbitrage: the NPC knows settlement A has high prices for commodity C and settlement B has low prices, so they haul. That decision depends on price knowledge for *both* settlements, which is stored in `price_knowledge: Vec<PriceReport>` (`state_npc.md` line 229). The proposal's observation exposes `price_beliefs(8)` — one value per commodity — which collapses all settlements into one belief. There is no observational pathway from "the trader knows prices at 12 different settlements" to the pointer head selecting the best (src, dst) pair. The proposal's `known_named_resources[K=6]` and `nearby_buildings[K=6]` don't include settlements-as-trade-targets either. So either (a) the trader picks (src, dst) by utility scoring over a structure that isn't in the observation, meaning the policy has to do it without the information, or (b) we add another slot table (`known_settlements[K=??]`) and the 1800-float budget grows.

In each of these, the decision is over an entity the NPC *knows about* but isn't spatially near. Slot-by-distance cannot represent this without either (a) adding more slot tables for each non-spatial entity class (faction leaders, known settlements, great works, bounties, policies) until the observation is 80% slot tables, or (b) silently suppressing the action by never populating the target slot.

### Why this matters more than it looks

The proposal's action head routes through the pointer (line 80): "the observation already exposes the relevant entities." If the relevant entity isn't in a slot, the action is unfireable. This inverts from "the policy learns what actions are viable" to "the schema authors decide what actions are viable by choosing which slots exist." That is a central-planner decision moved one layer up — not a win for the emergent-agency framing.

### Concrete alternative

Use graph-structured observations with attention for the non-spatial decisions, accepting the extra complexity:

- Keep slot-by-distance for combat/immediate actions (what works, works).
- Add a **known-entity graph** with typed edges: (self → known_npcs, self → known_settlements, self → known_factions, self → active_obligations). Entities in the graph are tokenized, not packed into fixed-width slots. The target head selects via attention over the graph's entity tokens, the same way `src/ai/core/ability_transformer/` already handles variable-cardinality ability lists via cross-attention.
- The graph is sparse; most NPCs' graphs are small (a spouse, a few friends, their faction, their home settlement). A dozen or two entity tokens on average, not K=12 padded.

This is more complex than the current proposal. It's also the only way to actually write `SwearVassal(distant_lord)` without hand-waving.

---

## Attack 4: Backend polymorphism is a unified contract we pay for but can only run one of

> "for_role(Hero) backend Neural { weights: 'npc_hero_v3.bin', h_dim: 128 }
>  for_role(Leader) backend Llm { endpoint: 'ndjson://localhost:5555' }
>  for_role(Outlaw) backend Goap { plan: 'outlaw_culture.goap' }" (lines 106–108)
>
> "Backends share a single trait — same observation/mask/action contract, different decision impl. Compiler emits batched dispatch: one call per backend per tick." (line 111)
>
> "Backend polymorphism cost: different NPCs running different backends per tick means batch sizes shrink (10 LLM-policied leaders + 200 neural-policied heroes + 19,790 utility commoners = three batches). LLM batch of 10 may not amortize the call overhead." (line 197)

### What the proposal is buying

A `PolicyBackend` trait (lines 113–121) with a unified `evaluate_batch` signature. Four quoted backends: Utility (declarative scoring rules), Neural (Burn), LLM (NDJSON), GOAP (plan files). The selling point is swappability: ship Utility today, research Neural later, LLM for narrative scenes.

### What the unified contract costs

1. **Every observation feature must be supported by every backend.** The Utility backend can ignore a feature; the Neural backend bakes it into weights and fails silently if absent; the LLM backend has to render it into a prompt. If a feature is cheap for one backend and expensive for another (e.g., encoding `recent_chronicle[NUM_CHRONICLE_CATEGORIES]` as a natural-language narrative for LLM vs a vector for Neural), the unified contract forces you to pay the highest cost. Alternatively, the schema grows a per-backend projection layer for every feature — which is what `ability_eval` and `ability_transformer` already do for the combat policy, and which is why those two systems have 5+ weight artifacts in `generated/` that we maintain in lockstep.

2. **Mixed-batch dispatch overhead is not hypothetical.** The proposal admits at line 197 that "LLM batch of 10 may not amortize the call overhead." For the combat sim, our best-tuned vLLM rollout from the `lfm-agent` project batches into hundreds to achieve reasonable throughput (see `~/Projects/lfm-agent/bench_throughput.py`). A batch of 10 leader LLM calls per tick, at a 1-tick-is-100ms cadence, is a straight miss on LLM cost economics. The proposal treats "LLM-policied leaders" as a natural thing to do; it's hard.

3. **Three impls, three tests, three bug surfaces.** Even if one backend is the "production" one and the others are research, each must preserve:
   - Determinism (`SimState.rng_state` contract from `CLAUDE.md` / `src/ai/core/tests/determinism.rs`).
   - Identical observation and mask validity (or the action space silently differs across backends, breaking save/load and replay).
   - Action legalization (the mask is declarative per `ActionKind` — lines 87–98 — but each backend's sampler has its own implementation of "respect the mask").

We maintain three bug classes, not one. In the combat sim, maintaining parity between Python training and Rust inference is already a chronic issue: MEMORY.md notes "Python↔Rust agreement: <0.002 on cross-attention predictions" as an achievement worth recording because it's hard to preserve.

4. **The "per-role backend" is not per-decision.** Line 183 quietly surfaces this as "Per-policy backend assignment vs per-entity assignment. Current proposal mixes — default policy + role-based override." A commoner who becomes an adventurer who becomes a leader (a trajectory the politics doc presumes is common — `systems_politics.md` line 182) needs to cross between backends mid-sim. The neural backend's weights file was trained on one role's distribution; swap to "leader" and you're running out-of-distribution inference on warmed-up weights. Utility → Neural → LLM transitions across a lifetime of one NPC are the interesting case; the proposal doesn't handle them.

5. **The trait's batch signature is wrong for the stated backends.** `fn evaluate_batch(&self, observations: &PackedObservationBatch, masks: &PackedMaskBatch) -> ActionBatch;` (line 114) assumes every backend can consume a batch of packed observations uniformly. For Utility, that's a straightforward parallel loop. For Neural (Burn), it's a batched forward pass — but Burn requires the batch dimension and the observation dim to be known at compile time for the inference graph to be efficient; changing backends per role per tick thrashes the graph cache. For LLM, batching means concatenating NDJSON payloads and parsing a combined response — a format the NDJSON protocol at `src/bin/sim_bridge/` was not designed for (it's a per-request line protocol per the `CLAUDE.md` description). Making LLM calls "batched" at the trait level lies about the cost: the request is still serialized under the hood; you've just wrapped ten one-shot calls in a function signature that says "batch." The trait unification claim is load-bearing and also slightly false.

6. **Combat pipeline evidence.** The project already has working proof that mixing backends is painful. Per `CLAUDE.md` section "AI Decision Pipeline": Squad AI → Ability Evaluator (urgency interrupt) → Transformer → GOAP → Control AI. Each layer optionally overrides the next. This is the *only* multi-backend pipeline we've shipped and its entire architecture is "utility default, neural interrupt." It is not a batched trait dispatch; it is a stacked chain of override-or-fall-through. The proposed `PolicyBackend` trait erases this structure into a single `evaluate_batch` call, throwing away the one working pattern we have.

### What we actually need

One backend, shipped. Choose:

- **Utility** (declarative scoring DSL) as the production backend for world-sim. This matches what the current AI stack does well (`src/ai/squad`, `src/ai/utility.rs`, `src/ai/goap`) and doesn't require a training pipeline. The world-sim needs breadth of behavior (50+ verbs), not depth of optimization on any one; utility is the correct primitive.
- **Neural as a research artifact** that can slot in for one or two verbs if and when we have evidence they benefit from it. This is how the combat pipeline actually works today: `ability_eval` is a neural interrupt layer over a utility-scored default, not a full-stack neural policy (`src/ai/squad/intents.rs` line where ability-eval fires at urgency > 0.4, per MEMORY.md).
- **LLM as a tool for narrative content generation**, not per-tick decisions. The current `src/bin/sim_bridge/` runs LLM agents via NDJSON at per-tick cadence, and the `lfm-agent` experiments show this is expensive per call and better suited to story-beat generation than combat micro. Don't design the core DSL around a use case we don't have budget for.

Cut the `PolicyBackend` trait abstraction from the proposal. Write the Utility backend directly against the observation struct, as Rust code that the DSL compiler emits. If in six months we have evidence the Neural backend is a win, add the trait then.

---

## Attack 5: Event-sourcing purity vs materialized observations — the proposal hand-waves the conflict

> "Event-sourced runtime (per the v2 reframing): state mutations are replaced by typed events; current state is a fold over events + entity baseline. Truly essential systems shrink from ~71 to ~19 across the 158-system audit." (line 7)
>
> "State materialization conflicts with event-sourcing purity: many observation features require materialized 'current value' reads that the event-sourcing kernel was supposed to avoid. We end up materializing most caches anyway, which is fine for performance but means the event log is more 'audit log' than 'source of truth.'" (line 198)

### The tension the proposal admits but doesn't resolve

Premise 1 is "event-sourced runtime: current state is a fold over events." The v2 economic doc makes this concrete: `treasury(sid) = initial + Σ income − Σ spend` over the event log (`systems_economic.md` line 533). The v2 social doc generalizes: "every action resolves to one or more events; every event lives in the chronicle or `world_events` stream" (`systems_politics.md` line 441).

Now look at the observation (proposal lines 36–54). It contains:
- `settlement.treasury_log` (line 50)
- `settlement.food_status`, `population_log`, `infrastructure_level`, `is_at_war`, `n_active_wars` (line 50)
- `faction.strength_log`, `n_at_war_with`, `n_allies`, `military_strength`, `war_exhaustion`, `diplomatic_stance per neighbor` (line 51)
- `recent_chronicle_event_counts_by_category` (line 50)
- `action_outcome_EMA per ActionKind`, `price_beliefs(8)`, `cultural_bias per major action_tag` (line 33)

Every one of these is a fold over the event log. If the event log is the source of truth and the observation is a fold, then **every NPC every tick re-folds the entire relevant event history for its observation**. That's not feasible. The observation construction MUST read from materialized caches.

### What the proposal says vs what it would have to do

Line 198 admits the conflict: "the event log is more 'audit log' than 'source of truth.'" But the entire reduction argument in the v2 docs — the "71 → 19 essential systems" framing at line 7 — depends on the event log being the source of truth. If you reverse that and say "the primary representation is the materialized cache, events are the audit log," you have just undone the v2 reduction and re-introduced every system that materializes state.

Look at the "Required derived views" section of `systems_economic.md` (lines 533–575). It lists 30+ derived views. Each view is either:

- **Computed on demand** from the event log at observation time — rejected above as infeasible for per-NPC per-tick observations.
- **Materialized and updated incrementally** as events arrive — which means we have an update system for every view, which is *exactly* the system count we were trying to reduce.
- **Cached lazily**, which means invalidation logic per event, which is the same cost distributed differently.

The proposal ships the observation without resolving which of these it chose.

### Concrete failure mode

Take `route_strength(a, b) = Σ CaravanArrived(a→b).profit × exp(-(now−tick)/τ)` (`systems_economic.md` line 555). This is an exponentially-decaying weighted sum over a potentially long event history. Per-NPC observation at the merchant-decision layer needs this value for every (a, b) the NPC might care about. Options:

1. Recompute per observation: O(|events|) per read, per NPC, per tick. At 20K NPCs this is untenable even with event-log windowing.
2. Materialize `route_strength[a][b]` as a field. But then it's a `TradeRoute.strength` field, which is exactly what `state_aggregate.md` lines 204–211 already has, which is the system the v2 doc reclassified as "DERIVABLE-VIEW (mostly)." We'd have re-introduced `trade_routes` as mutable state.
3. Lazy-materialize at observation time: "compute when asked, memoize with invalidation on new `CaravanArrived` event for (a, b)." This is a cache with an invalidation protocol; it's a system.

The proposal picks none of these in its text. The DSL (lines 128–170) has `summary recent_chronicle[...] { from world.chronicle filter |e| now - e.tick < 200, group_by e.category, output count_log }` (line 162), which is option (1) — fold at observation time. Computing a 200-tick sliding fold per NPC per tick over a chronicle that contains every significant world event is expensive at the scales described.

### A second failure mode: partial materialization is worst of both worlds

Suppose we compromise: some views are materialized, others are fold-on-read. Now the observation tensor is half-materialized and half-computed — which means its construction has two code paths, and the "event log is source of truth" claim survives only for the unmaterialized half. Every time we promote a view from fold-on-read to materialized (for performance), we're adding a system — exactly the reduction the v2 docs promised to avoid. The `systems_economic.md` reduction count ("30 of 32 systems collapse into an action vocabulary + an event-fold index", line 590) is accurate only under the infeasible full-fold-on-read model. Under a mixed model the real system count is somewhere between 19 (the proposal's claim) and 71 (the v1 baseline), and we won't know which until we profile — by which time the DSL schema is shipped.

### Determinism under the event-sourced model

The existing sim's determinism comes from a single `rng_state` threaded through all systems (`CLAUDE.md`: "All simulation randomness flows through `SimState.rng_state` via `next_rand_u32()`. Never use `thread_rng()` or any external RNG in simulation code. Unit processing order is shuffled per tick to prevent first-mover bias."). In an event-sourced world where actions are emitted by NPCs and then applied by a physics kernel, the event *order* is decisive — and the order depends on NPC processing order. The proposal does not specify whether events emitted in tick T by NPCs processed in a shuffle order are applied in emission order or in some canonical order before the next tick's observation read. This is the kind of detail that's easy in the current mutation-based sim and hard in the event-sourced one, and it interacts with how observations read "current state": if observation is a fold over event log, then the order events were appended to the log determines the observation, which determines the next tick's actions, and the whole loop can become non-deterministic across implementations even if `rng_state` is threaded correctly.

### Concrete alternative

Commit explicitly to a **two-tier model** and document it:

- **Primary state** = materialized current-value fields (treasury, stockpile, needs, relationships). These are mutated by event application, the same way the current sim works.
- **Event log** = immutable audit trail, used for chronicle output, replay, debugging, and *training data generation* (where the event sequence IS the dataset). Not read by per-tick observations.
- **Observation** reads from primary state only. Any "fold over events" that the v2 docs propose must correspond to a materialized field maintained by an incremental-update rule. If that field doesn't exist, the observation cannot include the fold.

This is a less elegant story than the v2 docs tell. It is also the only story that compiles.

---

## Attack 6: "GPU-targetable" is a claim the proposal cannot cash

> "GPU-targetable: the DSL must compile to either native Rust (with rayon/SIMD) or CUDA/wgpu compute. State lives on GPU when targeting compute; observation construction and policy evaluation are kernels." (line 11)
>
> "Compiler emits: Packed struct with named field offsets (debug tools decode bytes back to fields), Packing kernel — per-NPC, gather all referenced fields/views, write to packed buffer, JSON schema for ML pipeline tooling, Validation: every observation field must reference declared `entity` field, `view`, or `event` source" (lines 172–176)

Premise 5 commits the schema to GPU compilation. This is load-bearing in the sense that if the observation packing runs on CPU only, the "144 MB/tick" number from line 193 implies PCIe transfer of 144 MB per tick (or higher if you bounce back for action results). At 10 ticks/sec that's 1.4 GB/sec sustained PCIe traffic just for observations, which is half of PCIe 3.0 x16 bandwidth for a benefit we haven't measured. GPU observation packing only helps if the downstream policy is also on GPU — which is true for Neural, false for Utility, and absurd for LLM.

Pick apart the "compiles to CUDA" story for each DSL primitive in lines 128–169:

1. **Atoms** (single scalar: `self.hp_pct = self.hp / self.max_hp` at line 131) — trivially GPU-kernelizable. One thread per NPC, read two floats, divide, write one. This works.

2. **Blocks** (reusable groups: `block self.psychological { from self.needs as vec(6), ... }` line 134) — also trivial, just scheduled reads.

3. **Slot arrays** (line 143: `slots nearby_npcs[K=12] from query::nearby_entities(self, kind=NPC, radius=50) sort_by distance(self, _)`) — this is a **kNN query per NPC per tick**. For N=20K NPCs with radius=50 on a world of who-knows-what spatial extent, this is the expensive thing in the schema. Common implementations: spatial hash (works on GPU but needs careful collision handling), BVH (builds on CPU, traversal on GPU is awkward), brute force (O(N²) — untenable at 20K NPCs: 400M distance comparisons per tick). Then sorting the top-K per NPC on GPU is a per-thread partial sort; possible but not trivially parallelizable. The proposal gestures at "kernels" without describing which kNN algorithm wins.

4. **Slot arrays from event rings** (line 151: `slots recent_events[K=12] from latest(self.memory_events)`) — the event ring is per-NPC mutable state. Writing to it requires either coherent GPU state (complex) or a CPU→GPU sync before each tick (fast path defeat). The proposal doesn't address how events written by NPC A's action in tick T are visible to NPC B's observation in tick T+1 if state lives on GPU.

5. **Cross-entity references in mask evaluation** (line 88: `Attack(t) when t.alive ∧ t.team ≠ self.team ∧ distance(t) < AGGRO_RANGE`) — for each (self, t) pair you're reading two entities' state. Per-NPC this is one gather against entity table on GPU. Fine for dense gathers; harder when `t` is a pointer into a sparse entity list with deletions. The proposal's entity spawn/despawn is per-tick (per `state_npc.md` line 14 and the `ResourceSpawned` / `NodeDepleted` events in `systems_economic.md` line 301), so the entity table has holes and renumbering, which doesn't pack cleanly onto GPU.

6. **Summary folds** (line 162: `summary recent_chronicle[NUM_CHRONICLE_CATEGORIES] { from world.chronicle filter |e| now - e.tick < 200, group_by e.category, output count_log }`) — this is a per-NPC scan over a global chronicle buffer with a filter predicate and a group-by. On GPU: reduce-by-key with predicate. Works, but requires a segmented scan per NPC, and the chronicle is a shared resource so contention is a concern.

7. **LLM backend on GPU** — the LLM runs on GPU, sure, but the NDJSON endpoint (line 107: `endpoint: "ndjson://localhost:5555"`) is a per-call network round-trip to a separate process. The observation has to be CPU-side before it's serialized. "State lives on GPU" collides directly with "LLM backend is NDJSON."

None of this is unsolvable. It is also not done. The proposal gestures at GPU as if it's a free drop-down, when in reality it's a major engineering undertaking that forecloses certain primitives (event rings) and imposes specific implementation choices (spatial hashing, entity compaction).

### The honest version

Scope the GPU story narrowly:

- Atom/block observation packing: GPU-friendly. Write that kernel.
- kNN slot arrays: committed to a spatial index data structure, with the knowledge that the index is rebuilt per tick and has a fixed cost curve with N.
- Event summaries / mask cross-entity reads / LLM dispatch: CPU-side.

Don't claim "the whole policy compiles to CUDA." Claim "observation packing for the common feature subset has a GPU kernel; the policy dispatch is CPU-side for masking and backend polymorphism; mixed-CPU-GPU pipeline with known sync points." That's a scope we can actually ship.

---

## Attack 7: DSL complexity is premature when you have one policy

> "The DSL exposes primitives so authors don't manually write packing code: … Atom, Block, Slot array with explicit padding, Slot array from event ring, Bitset → fixed one-hot, Summary fold over recent events (compiler emits a fold kernel)." (lines 125–169)
>
> "DSL complexity: the schema language above is non-trivial. May exceed what's worth building if we end up with one or two policy declarations. Compare to writing the equivalent Rust by hand." (line 196)

The DSL surface in the proposal has six distinct primitives (atom, block, slot array, bitset, summary, mask language) plus four action head types (categorical, pointer, continuous scalar, continuous vec2) plus a backend dispatch mini-language. That's more surface area than `src/ai/effects/dsl/` (the ability DSL, which has winnow-based parser → AST → lower.rs → AbilityDef for five effect dimensions).

How many policy declarations will we actually have? The proposal gestures at "default + role override" (line 183), suggesting maybe 3–6 policies total (commoner, hero, leader, outlaw, monster, maybe merchant). Building a DSL compiler to factor 3–6 declarations is build-first-use-later.

The ability DSL was worth building because there are ~856 abilities across ~200 heroes (`CLAUDE.md`, `assets/lol_heroes/` + `assets/hero_templates/`). The ratio of declarations to primitives is ~100:1. For policies, the ratio is going to be closer to 6:10. Writing them in Rust with a clean helper module (similar to `src/ai/core/ability_eval/game_state.rs`) is faster, more debuggable, and doesn't invent a third DSL we maintain parsers for (ability DSL + GOAP DSL + now policy DSL).

The proposal itself acknowledges this at line 196 as a risk and then builds the DSL anyway. The right sequencing: write two policies in Rust first. If the second policy shares ≥60% of its packing code with the first, extract a builder library. If five policies share enough, write a macro. If ten policies share enough, write a DSL. We are at zero.

### Maintenance tax of a third DSL

The project already has two in-tree DSLs and both take meaningful effort to evolve:

- `.ability` DSL — `src/ai/effects/dsl/` with a winnow parser, AST lowering, and five effect/area/delivery/trigger/tags dimensions. The DSL emitter (`src/ai/effects/dsl/emit.rs`, per MEMORY.md) is itself a non-trivial artifact.
- `.goap` DSL — `src/ai/goap/` with its own parser, planner, and "party culture" modifiers.

A third DSL means: a parser, a formatter (for round-trip editing), an AST type, a lowering pass, an error-reporting layer, tests, and debug-dump tooling. Call it four developer-weeks of setup plus ongoing maintenance as the schema evolves. Hand-written Rust with good builder helpers costs zero of that. The proposal pays the maintenance tax across the project's multi-year horizon to save a few hundred lines of repetitive packing code in one or two policies.

---

## Concessions

The proposal is right about several important things and should be acknowledged:

1. **Determinism as a non-negotiable.** The proposal doesn't explicitly call it out, but the `SimState.rng_state` contract (per `CLAUDE.md`) flows naturally through the Utility backend design, and the multi-head action space with explicit masks is determinism-preserving in a way loose "AI decides freely" architectures aren't. Keep this.

2. **Masks as first-class.** Component 3 (lines 84–100) puts hard constraints in a declarative mask language rather than burying them in scoring code. This is correct and matches what the combat engagement heuristic had to be bolted onto V3 after the fact (`transformer_rl.rs`). Having validity conditions live with the ActionKind declaration is good.

3. **The pointer head, for combat and immediate actions.** The pointer-into-slot design (line 80) avoids entity-ID translation layers and is simpler than what the combat V3 does. For the regime where slot-by-distance works, this is a clean design.

4. **Action-outcome EMAs as observation features.** `action_outcome_EMA per ActionKind` (line 33) is a lightweight way to give the policy self-knowledge of what's worked before. This is a genuinely good idea that combat ability-eval already exploits.

5. **Acknowledging the schema-as-contract problem.** Line 186 does name the issue, even if it punts the solution. Naming it is a prerequisite to solving it. Keep the framing.

6. **Separating physics from policy.** The philosophical move of "physics systems apply events, policy systems choose actions" is correct at the top level. The quarrel is with the scope of what ends up on each side of the line, not with drawing a line.

---

## Counter-proposal sketch

A smaller policy interface that commits less and preserves more optionality. Three changes to the proposed schema:

### 1. Observation classes, not a single packed tensor

Replace the 1500–1800-float monolithic observation with named observation classes. Each class is ~30–100 floats:

```
observation_class combat_state {
  self { hp_pct, shield_pct, attack_damage, attack_range, num_status_effects }
  slots nearby_threats[K=8] sort_by time_to_impact { ... 8 feats ... }
}

observation_class economic_state {
  self { gold_log, debt_log, num_contracts, income_rate_log }
  context { settlement.treasury_log, settlement.stockpile_pct[8] }
  slots active_contracts[K=4] { ... }
}

observation_class social_state {
  self { social, compassion, has_spouse, n_close_friends }
  graph known_npcs sparse { trust, familiarity, role_one_hot(8), exists }
}

observation_class political_state {
  self { role_one_hot, faction_loyalty, reputation_at_settlement }
  graph known_factions sparse { diplomatic_stance, strength_log }
  graph known_leaders sparse { faction_id, distance_log, trust }
}
```

Actions declare which classes they consume:

```
action Attack { consumes combat_state }
action RunCaravan { consumes economic_state, combat_state }
action SwearVassal { consumes political_state, social_state }
```

The packing kernel produces only the classes the fired actions need, and only when they fire. Idle NPCs pack nothing. A merchant running a caravan packs `economic_state + combat_state` (~70 floats). A leader considering vassalage packs `political_state + social_state` (~100 floats) at decision time, not every tick.

### 2. Hierarchical action head with per-level training signal

```
action_head {
  level macro_class: { Combat, Economic, Social, Political, Movement }
  level micro_verb: conditional on macro_class
    Combat -> { Attack, Flee, Focus, CastAbility, PickUp }
    Economic -> { Eat, Harvest, Work, Deposit, RunCaravan, Trade, PayContract, Bid }
    Social -> { Gossip, Train, Teach, ProposeMarriage, Forgive, BetrayFriend }
    Political -> { SwearVassal, DeclareWar, PostBounty, AdoptCharter, IssueProphecy }
    Movement -> { MoveTo, Travel, Idle }
  level target: pointer into graph+slot union specific to micro_verb
  level magnitude: optional, only for verbs that take it
}
```

Rare actions are hidden behind their macro class; the training signal at macro level is balanced (each class fires often enough that the macro softmax learns well); the rare micro inside a rare macro can be learned with less data than a flat 150-way softmax.

### 3. One backend (Utility) as the DSL target, with escape hatch

Cut the `PolicyBackend` trait. Have the DSL compile directly to a Utility scoring function + mask evaluator. Provide an escape hatch: any action can have a `override_eval("fn_name")` annotation that dispatches to hand-written Rust (which can call a neural model, a planner, whatever). This preserves the future-proofing argument without paying the three-backend tax.

If in a year we have a neural policy that outperforms utility on some verb subset, we add a new DSL directive (`neural_eval("weights.bin")` on those verbs) rather than restructuring the world around a polymorphic trait.

### 4. Action masking as a first-class fallback chain, not a sampler constraint

Rather than "mask + sampler" (proposal lines 84–100, which assumes the backend respects the mask during sampling), use the combat-pipeline pattern: a declarative chain of overrides where each layer can either commit an action or pass to the next. Outer layer is squad/role defaults; next layer is the utility scorer; innermost layer is an optional neural interrupt. If any layer picks an invalid action (mask violation), the action is rejected and the next layer takes over. This is how `src/ai/squad/intents.rs` already works and it scales to new verbs without retraining the underlying mask-respecting sampler.

### 5. Event log as training data, not observation source

Explicitly document:
- Primary state is mutated by event application (current architecture preserved).
- Events are emitted for chronicle, replay, and dataset generation.
- Observations read primary state only. Any "summary over events" view in the DSL is backed by an explicit materialized field with an update rule.

This is the honest version of the event-sourced claim. It keeps the audit-log benefit without pretending the log is the live source.

---

### What the counter-proposal explicitly doesn't commit to

Worth being explicit about where the counter-proposal leaves decisions unmade, because committing to those prematurely would replicate the original proposal's error:

- **Observation versioning scheme.** Defer until we have a reason to evolve a schema. The per-action-class design means individual classes can be versioned independently, which is a lighter-weight migration story than a monolithic schema hash.
- **Training pipeline.** Defer until we have evidence a given verb benefits from learning over utility. The escape-hatch `override_eval` hook is the extension point if and when that happens.
- **Pointer semantics for the graph structure.** Scaled-dot-product attention works; so do learned embeddings; so does a simple "distance + role" utility score for utility backends. Pick one when the policy is concrete, not before.
- **Reward signal for any future RL.** The combat pipeline's dataset generation path (`scripts/train_student.py`, `training/train_rl.py`) already has a reward-shaping model; port it when/if needed. Don't design reward into the schema up-front.

Leaving these unmade is the feature, not a bug. The original proposal's 10 "Open design questions" are load-bearing for its architecture to work; this counter-proposal's deferred decisions are genuinely optional until use cases force them.

---

## Closing argument

The proposal is ambitious in the right direction — a data-driven observation/action/policy interface, event-driven agency, swappable decision-making — but it over-commits in four places simultaneously. It commits to a schema (the 1500–1800-float tensor) before knowing what features matter; to a vocabulary (50–150 verbs) before knowing which of the v2 reframings actually produce the claimed emergent behavior; to a backend abstraction (the trait) before we have evidence we need more than one; and to a DSL before we have enough policies to factor. Each commitment is fine in isolation; together they produce a schema-as-contract that locks in every guess.

The way forward is not to tweak the proposal's numbers — not K=12 vs K=20 slot sizes, not 1500 vs 1200 float budgets, not categorical vs pointer head — but to downsize its ambition. Ship a Utility policy against decision-class observations with a hierarchical action head. Run the sim. Look at where the policy fails: where does it pick the wrong target, where does the mask gate out actions that should be legal, where does an observation feature turn out to have been dead weight. Use that evidence to design version two. The combat pipeline took four or five iterations of `ability_eval_weights_v2.json` → `v3_embedded.json` → transformer → actor-critic → pointer-V3 to converge on what features actually predicted good decisions; the world-sim is strictly harder and deserves the same evidence-gathering cadence.

The cost of shipping the proposal as written is a year of retraining compatibility layers and a schema we can't change without breaking every weight artifact and LLM prompt we've produced against it. The cost of shipping the smaller version is a weekend of rewriting when we have one or two more policies. That tradeoff points the same direction every time: build the small one, measure, expand.
