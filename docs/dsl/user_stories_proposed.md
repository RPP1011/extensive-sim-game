# Proposed User Stories — DSL Validation

Draft list of user stories for validating the DSL design. Greenlight the ones worth investigating; subagents will write a per-story analysis showing either "supported by the current design" (with the specific declarations that handle it) or "gap — here's what the design is missing."

Format: `As a [role], I want [thing], so that [outcome]. _Validates: what aspect of the DSL._`

---

## A. DSL authoring (sim engineer writing the world)

1. **Declare a new entity type with baseline fields.** As a sim engineer, I want to declare a new entity type (e.g. `Agent`, `Item`, `Group`) with its baseline fields, so that the compiler emits SoA storage + indexes + serialization. _Validates: entity declaration grammar, layout hints._

> This is an inherently dangerous operation. I would argue we shouldn't support it except as abstractions over the existing entity types as it will require DSL modifications and could greatly complicate code gen.

2. **Declare typed events.** As a sim engineer, I want to declare event types (e.g. `Damage`, `Harvest`) with typed parameters, so that all state transitions go through one inspectable channel. _Validates: event declaration, append-only buffer codegen._

> Good

3. **Declare a derived view.** As a sim engineer, I want to declare a view (e.g. `is_hostile(a, b) = relationship.valence < THRESH ∨ groups_at_war(a, b)`), so that downstream code reads a function instead of a stored field. _Validates: view declaration, lazy vs materialized hint, GPU view-fold codegen._

> Good

4. **Declare a physics cascade rule.** As a sim engineer, I want to declare `Damage(d) → hp[d.target] -= d.amount → if hp ≤ 0: emit EntityDied`, so that I don't write the cascade by hand. _Validates: physics-rule grammar, ordered cascade execution, cycle detection._

> Good

5. **Declare a spatial / non-spatial query.** As a sim engineer, I want to declare `query::nearby_agents(self, radius=50)` and `query::known_actors(self)`, so that observation packing and mask predicates can use them. _Validates: query primitives, indexing hints, push-vs-pull observation discovery._

> Good, but code gen for this has to be very smart to do it well.

6. **Declare an invariant.** As a sim engineer, I want to express "no agent has multiple Marriage memberships at once," so that the DSL catches violations at compile time (where possible) or runtime. _Validates: constraint language, where invariants live._

> Great

---

## B. Action and mask authoring (gameplay designer)

7. **Add a new agentic action verb.** As a gameplay designer, I want to add a new verb (e.g. `Pray`), so that NPCs can choose it. _Validates: action vocabulary extension surface; what touches when adding a verb._

> Mixed, again this is an inherently dangerous operation. Maybe we should introduce action primitives that would compose these verbs?

8. **Write a mask predicate.** As a gameplay designer, I want to write `mask Pray(self) when self.memberships ∋ Group{kind=Religion}`, so that only members of a religion can pray. _Validates: mask predicate grammar, cross-entity reference (membership lookup), compile-to-bool-tensor codegen._

> Great

9. **Add a new QuestType.** As a gameplay designer, I want to add `QuestType::BuildMonument` and its cascade rule, so that PostQuest can express monument construction without engine changes. _Validates: QuestType extension, reward semantics, group-scoped party expansion._

> Great, but again quest types are a concern for the same reason verbs are a concren.

10. **Define a new GroupKind.** As a gameplay designer, I want to add `GroupKind::Coven`, so that NPCs can form covens with their own membership rules and standings. _Validates: GroupKind extension, eligibility predicate language, standings semantics._

> Good

11. **Add a new creature_type.** As a gameplay designer, I want to declare `CreatureType::Centaur` with capability flags + default personality + hunger drives, so that wild centaurs behave plausibly without per-creature code. _Validates: creature_type declaration, capability composition, parameter override._

> Good

12. **Tune a mask without code change.** As a gameplay designer, I want to change `Attack` aggro range from 50 to 60 in a config, so that I can balance gameplay without rebuilding. _Validates: parameterization, hot-reload story._

> Good
---

## C. Observation schema (ML engineer)

13. **Inspect schema as JSON.** As an ML engineer, I want to dump the observation schema as JSON with field names + offsets + normalization constants, so that I can build a model whose input layout matches. _Validates: schema introspection, JSON emission, naming stability._
> Essential
14. **Add a new observation feature.** As an ML engineer, I want to add `self.fame_log` to the observation, so that the model can use it. _Validates: append-only schema growth, schema hash bump, migration path._
> Essential
15. **Schema versioning.** As an ML engineer, I want loading an old model against a new schema to fail with a clear error or pad-zero migrate, so that I don't silently corrupt training. _Validates: schema hash semantics, `@since` annotations, migration tables._
> ESSENTIAL. WE DO NOT WANT TO HAVE V1 AND V2 AND V3 in the same codebase. Git can be used to store versions, it is nothing but waste to support backwards compatibility in a solo project like this.


16. **Per-tick training data emission.** As an ML engineer, I want to log `(observation, action, log_prob, reward, terminal_flag)` per agent per tick to a buffer file, so that offline training can consume it. _Validates: training-data emission hooks, replay buffer format._

> Great

17. **Inspect a single agent's observation.** As an ML engineer, I want a CLI that decodes any agent's packed observation tensor at any tick into named fields, so that I can debug what the model saw. _Validates: schema introspection + debug tooling._

> Great

18. **Compare two policy backends.** As an ML engineer, I want to run a fixed scenario with two backends (Utility vs Neural) and get a side-by-side decision diff per tick, so that I can quantify model behavior changes. _Validates: backend swap surface, deterministic seeded comparison, decision logging._

> Great.

> I would like a way of evaluating probes on a known dataset as well.

---

## D. Training pipeline (trainer)

19. **Bootstrap from utility trajectories.** As a trainer, I want to run the sim with the Utility backend, log all trajectories, then start neural training from them via behavior cloning, so that I don't start from random. _Validates: backend interchangeability, trajectory format, IL pipeline integration._

> Great

20. **Declare reward in DSL.** As a trainer, I want to specify `reward { delta(needs.satisfaction) × 0.1 + 1.0 on event(EntityDied{killer=self}) ... }` in the policy block, so that the training pipeline computes rewards consistently with the policy decisions. _Validates: reward grammar, event subscription, credit assignment hooks._

> Great, but we will also want to support actor critic

21. **Up-weight rare actions in training.** As a trainer, I want to flag `DeclareWar` as a rare-event action that gets oversampled in the replay buffer, so that the model doesn't ignore it. _Validates: action-frequency annotations, prioritized replay support._

> Essential

22. **Curriculum / staged training.** As a trainer, I want to start training with `Hunt` and `Eat` actions only, then expand to all verbs, so that the model converges in stages. _Validates: action-mask scoping in DSL, training-time mask overrides._

> Essential

23. **Deploy a model checkpoint.** As a trainer, I want to drop a `.bin` weight file into the runtime and have the next tick use it, so that production updates are simple. _Validates: model loading, schema hash check, hot-swap semantics._

> Essential

24. **Detect mode collapse.** As a trainer, I want training metrics to flag when action entropy drops below a threshold (model picks one verb only), so that I catch broken training early. _Validates: training metric emission, action-distribution telemetry._

> Essential

---

## E. Runtime / sim engineer

25. **One batched neural forward per tick.** As a sim engineer, I want all alive agents' observations packed into one batch, dispatched to one model forward per tick, so that GPU utilization is high. _Validates: batched observation packing, dispatch model._

> Good

26. **Materialize hot views eagerly.** As a sim engineer, I want frequently-read views (e.g. `hp`, `treasury`, `mood`) materialized as fields updated by event handlers, so that observation packing doesn't fold the event log every tick. _Validates: view materialization hints, fold codegen._

> Essential

27. **Add a physics rule without breaking others.** As a sim engineer, I want to add a new cascade rule for a new event type without modifying existing rules, so that the rule set composes. _Validates: rule registration model, ordering semantics, conflict detection._

> Essential, should have "compile" time validation

28. **Compile to either Rust or CUDA.** As a sim engineer, I want to compile the same DSL source to either native Rust (rayon-parallel CPU) or wgpu/CUDA (GPU compute), so that I can target different platforms. _Validates: backend-abstraction layer, kernel codegen, what's CPU-only._

> Essential, use voxel-engine though, that is what that project is for

29. **Deterministic sim given seed.** As a sim engineer, I want the same seed + same DSL + same input → same output every run, so that I can reproduce bugs and validate behavior. _Validates: RNG semantics, event ordering, NPC processing order._

> Essential

30. **Zero per-tick allocations in steady state.** As a sim engineer, I want pooled buffers + fixed-size structures so that no allocations happen on a typical tick, so that GC/allocator pressure doesn't tank tps. _Validates: pooled scratch, fixed-cap collections, append-buffer reuse._

> Essential, ideally we can avoid memory allocation past startup altogether.

31. **Scale from 2K to 200K agents.** As a sim engineer, I want the same DSL to run at 100× current scale without architectural rewrite, so that content growth doesn't force re-engineering. _Validates: complexity-per-agent claims, event volume scaling, observation tensor scaling._

> Essential, the purpose behind this entire thing

32. **Save and reload mid-run.** As an operator, I want to checkpoint state + event log + RNG to disk and resume exactly, so that long simulations survive restarts. _Validates: serialization completeness, replay-from-checkpoint semantics._

> Essential

33. **Replay any range of ticks.** As an operator, I want a recorded run to be replayable from tick A to tick B with full fidelity, so that bug reports are reproducible. _Validates: event log replay, deterministic re-execution._

> Essential

---

## F. Debugging and observability

34. **Trace why a mask evaluated false.** As a debugger, I want to query "why didn't agent X take action Y at tick T," getting the failing mask predicate + the values that made it false, so that I can diagnose missing decisions. _Validates: mask explainability, predicate decomposition._

> Great

35. **Inspect cascade fan-out.** As a debugger, I want to see "tick T: ChosenAction → Damage → EntityDied → SuccessionTriggered," so that I can debug emergent chains. _Validates: event causal graph, chronicle integration._
> Great
36. **A/B compare two policy backends.** As a debugger, I want to run two backends in parallel on the same world snapshot and diff their decisions, so that I quantify policy changes. _Validates: deterministic forking, decision-equivalence checks._
> Great, seems emergent from the other stories though
37. **Step through a tick.** As a debugger, I want to advance one tick at a time and see (observations packed → backends fired → actions chosen → cascades → applied), so that I understand tick semantics. _Validates: tick-stepping support, intermediate-state introspection._
> Essential
38. **Self-contained tick reproduction.** As a debugger, I want to capture "a single tick" as state-snapshot + events + decisions in one bundle, so that I can attach it to a bug report. _Validates: snapshot serialization, debug bundle format._
> Essential

39. **Per-agent decision history.** As a debugger, I want a queryable log of every action an agent chose, with the observation that produced it, so that I can spot weird patterns. _Validates: per-agent decision trace, retention policy._
> Essential

40. **Flamegraph attribution to DSL declarations.** As a sim engineer, I want a flamegraph that says "view::mood: 4% of tick" or "mask Attack: 1.2% of tick," so that I optimize the right declaration. _Validates: per-declaration timing instrumentation._
> Essential
---

## G. Modding / extensibility

41. **Add a new ItemType.** As a modder, I want to declare a new item kind with its events + its effects, so that I can add content without forking the engine. _Validates: Item extension surface, event-handler registration._
> Essential

42. **Add a new civilization.** As a modder, I want to define a new "civilization" — a set of group structures, default class biases, cultural action priors — so that I can ship content packs. _Validates: configuration-vs-code boundary, default policy overrides._
> This is data, don't know if it should be part of the DSL unless you mean kind of civilization with its own cultural rules? I assume civilizations would be instances of a culture though, and that may not be the correct framing.

43. **Override a physics rule.** As a modder, I want to override a specific cascade rule with my own (e.g. "in my mod, Damage halves on undead"), so that mods can change physics in well-defined ways. _Validates: rule layering / precedence, conflict resolution._
> I think this is implicitly very dangerous as it will be easy to brick the system. Why not just make the entire system modifiable so we don't need this? They can just add or delete rules themselves? I would be ok with this as syntax sugar

44. **Hot-reload mod changes.** As a modder, I want to edit a mod file and see changes in the running sim without restart, so that iteration is fast. _Validates: hot-reload story, what survives a reload._

> GOod

---

## H. Player-experienced behaviors (validates the DSL supports actual gameplay)

45. **Wars are motivated.** As a player, I want NPCs to declare war on groups whose recent grievances exceed a threshold (raids, broken treaties, contested territory), so that wars feel earned. _Validates: PostQuest{Conquest} cascade chain, observation features that drive the decision, mask gating._

> Who would start the first war? What about wars for expansion?

46. **Succession dynamics on leader death.** As a player, I want a faction leader's death to trigger a succession contest (vote / coup / heir-claim), so that politics has continuity. _Validates: cascade rules from EntityDied + role=Leader, group governance fields, AcceptQuest mechanics._

> I think this should be emergent.

47. **Marriages form by relationship trust + compatibility.** As a player, I want NPCs to propose / accept marriages based on personality compatibility + relationship history + cultural eligibility, so that families form organically. _Validates: PostQuest{Marriage} → AcceptQuest cascade, observation features for the proposing decision, group=Family creation event._

> Sure

48. **Monsters defend dens, hunt prey.** As a player, I want a wolf pack to defend its den when threatened and hunt deer when hungry, so that ecology feels alive. _Validates: creature_type behavior parameters, group=Pack dynamics, predator/prey relationship semantics._

> Sure

49. **Some NPCs betray for personal gain.** As a player, I want some NPCs to defect from their faction to a rival for high-ambition + low-loyalty profiles, so that loyalty feels earned. _Validates: LeaveGroup + JoinGroup as agentic actions, reputation cascade, multi-membership conflict semantics._

> ESSENTIAL

50. **Renowned characters emerge from chronicles.** As a player, I want legendary heroes / infamous outlaws to emerge from chronicle event density (mentions, deeds), so that story emerges from gameplay. _Validates: derived view `is_legendary(agent)`, narrative-from-events pattern._

> Legendariness is kind of fluffy. There should be a reputation system, and abilities should be capable of being impacted by reputation, so that way a legendary hero can emerge organically without making a new system for it.

51. **Long wars affect economy.** As a player, I want a 2000-tick war to leave settlements impoverished, trade routes broken, populations depleted, via cascade — not via a hand-coded "war affects economy" system. _Validates: emergent-from-cascade design, that no special "economic war effect" system is needed._

> ESSENTIAL

52. **Children inherit parental traits.** As a player, I want children born to inherit personality + class biases + behavior tag tendencies from parents, so that lineages have character. _Validates: Spawn cascade for ChildBorn, inheritance computation in event handler, observation reflects lineage._

> ESSENTIAL

53. **Emotional reactions to witnessed events.** As a player, I want NPCs to record memory events when they witness deaths / battles / gifts, with grief / anger / pride spikes that decay over time, so that emotions are reactive. _Validates: Memory event emission, emotion drift derivation, observation features._

> ESSENTIAL


54. **Faction alliance brokered through bidding.** As a player, I want two factions to form an alliance after one bids resources / vassal commitments / shared interests in a Diplomatic auction, so that diplomacy has structure. _Validates: Bid mechanism for non-economic outcomes, AuctionKind::Diplomatic, multi-party AuctionResolution._


> ESSENTIAL

---

## I. Auction / unified mechanics validation

55. **Settlement commodity auction.** As a sim engineer, I want a settlement to hold periodic commodity auctions where local NPCs bid based on need / supply / price beliefs, so that prices emerge from supply-demand. _Validates: AuctionKind::Commodity lifecycle, Bid mask predicate, emergent price view._

> I am not aligned with this so much as I think the entire economy should operate on a bidding system. I don't mind if some emergent behavior causes community auctions, but I don't want a physics rule saying every 100 ticks a settlement will put all its merchants in the town square and they will dump inventory


56. **Faction-charter auction.** As a sim engineer, I want a faction to put up a settlement charter for bid (residents bid for governance terms), so that political systems use the same mechanism as commodity trade. _Validates: AuctionKind::Charter, reputation-as-payment, MutualAgreement resolution._

> This is too abstract.

57. **Auction state on GPU.** As a sim engineer, I want auction state (current bids, deadline, winner-so-far) to live on the GPU as a buffer, so that bid evaluation is parallel. _Validates: which auction state is GPU-amenable._

> This is too much of an implementation detail, also it breaks the backend we were previously discussing.

58. **Mercenary hiring as auction.** As a sim engineer, I want a settlement to post a Service auction for "we need 5 fighters for a Conquest quest" and idle adventurer NPCs bid for the contract, so that hiring uses universal Bid. _Validates: Service auction lifecycle, Bid → AcceptQuest cascade, mask for "currently looking for work."_

> Great

---

## J. Adversarial / failure modes

59. **Malformed action drops safely.** As a sim engineer, I want an agent that emits an action with an invalid pointer target (e.g. dead entity slot) to drop the action with a logged warning, not crash, so that the runtime is robust to model glitches. _Validates: action validation pre-execution, error semantics._

> Essential, keeps the runtime validation of the existing system

60. **Determinism bound on text generation.** As a sim engineer, I want events that include text generation (chronicle entries, NPC names) marked as non-replayable so that replay determinism is correctly bounded — text gen can produce different strings on replay without breaking the sim. _Validates: replay-determinism scope, event taxonomy._

> Essential, text gen should not be load bearing. Use numeric IDs for everything important


61. **Group dissolution with active quests.** As a sim engineer, I want a Group dissolution to gracefully handle active quests scoped to that Group (cancel? transfer? complete?), so that no dangling quest state remains. _Validates: cascade rules for DissolveGroup, group-scoped party reference semantics._

> ESSENTIAL

62. **Agent death mid-quest.** As a sim engineer, I want an agent dying while in a quest's party_member_ids to be cleanly removed from the quest without breaking quest progress evaluation, so that mortality doesn't corrupt quest state. _Validates: Quest membership as derived view (already proposed), EntityDied cascade._

> ESSENTIAL, unless agent is part of quest criteria

63. **Two agents propose marriage to each other simultaneously.** As a sim engineer, I want simultaneous PostQuest{Marriage} from A→B and B→A within one tick to resolve to one MarriageFormed (not two, not zero), so that race conditions don't corrupt marriage state. _Validates: cascade ordering, idempotency, event resolution semantics._

> I think Marriage should be more like a social group invite. Also, the per frame timing means asymetric relationships would never result in marriages?

64. **Adding a new ActionKind breaks model.** As an ML engineer, I want adding a new categorical action variant to bump the schema hash and require an explicit migration, so that I can't accidentally drop a model into a sim it doesn't understand. _Validates: schema-hash scope, action-vocabulary versioning._

> ESSENTIAL

---

## How to greenlight

For each story:
- ✅ "Investigate" — agent will produce a per-story analysis: how the current DSL declarations + events + views support this, OR what's missing (gap)
- ❌ "Skip" — drop from this round
- 🔄 "Defer" — interesting but not now

You can also mark whole categories (`A: investigate all`, `H: investigate all except 51`, etc.) or rank by priority.

Output of the investigation pass: per story, a markdown file `story_<NN>_<short-name>.md` (or one consolidated doc per category) showing the support story or the gap. Gaps surface as DSL extension proposals.
