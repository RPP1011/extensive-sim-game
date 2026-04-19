# Politics / Narrative Systems — Strict Reframing

## Methodology

Under the **strict rubric**, the v1 classification was too generous — it labelled ~17 of 36 systems as ESSENTIAL because they *write* faction/settlement/entity state. That test is wrong: the question is whether the write is *irreducible physics/bookkeeping* or whether it models *a choice a simulated NPC could make*. Any state change that corresponds to a plausible in-world decision (declare war, form an alliance, abdicate, betray, marry, have a child, spy, found a town, swear a prophecy, vote in council) belongs to the **NPC action space**, not to a privileged top-down "politics system". The politics module then reduces to:

1. **ESSENTIAL** — irreducible bookkeeping: `entity.spawn` (ID alloc + `alive=true`), chronicle emission (event recording is the event log itself), and *possibly* the leader-class decision function (parallel to `action_eval` for combat NPCs). Team flip is defensible as essential *only* because the entity store currently represents allegiance as a direct field; the event-sourced alternative is `npc.faction_id = derive_from_latest(FactionMembershipChanged)`.
2. **EMERGENT** — any state change that an NPC could choose as an action. These systems are replaced by entries in the NPC action vocabulary plus reducer logic driven by the emitted event.
3. **DERIVABLE-VIEW** — pure queries over chronicle / world_events / primary state. Legendary status, era names, culture labels, reputation, faction tech tiers, war exhaustion, haunted sites, lineages — all computable on demand.
4. **DEAD/STUB** — empty bodies (`war_exhaustion`, `defection_cascade`, `faction_tech`) and noise emitters whose own source comments admit "without X state, we approximate" (`charter`, `choices`, `rival_guild`, `victory_conditions`).

Aggressively applying this rubric collapses the v1 count of 14-18 ESSENTIAL systems to a core of **4**: `chronicle` (event recording), `settlement_founding.spawn` (ID alloc only), `family.births` (ID alloc only), `faction_ai` (as the leader-level decision function). Everything else is either agentic, derivable, or dead.

---

## Per-system reanalysis

### faction_ai
- **Old:** ESSENTIAL (owns `SettlementConquered`, regenerates `military_strength`).
- **New:** ESSENTIAL *(as decision function)*, with most *effects* reframed as EMERGENT consequences of leader-NPC actions.
- **Reframe:** `src/world_sim/systems/faction_ai.rs:12-42, 219-251` runs a per-faction AI loop: regen strength, pick a stance reaction, occasionally attack the weakest rival. Every `match` arm models a choice a **faction leader NPC** would make — "declare war on the weakest neighbour", "reclaim lost territory", "raise taxes to recover from unrest". This is structurally the same as `action_eval.rs` for combat NPCs. Treat `faction_ai` as the *decision head* for leader-class NPCs (parallel to `action_eval`), and emit each branch as an NPC action that fans out to events (`DeclareWar`, `ReclaimSettlement`, `LevyTax`, …). Strength regen = book-keeping `Tick` applied to a physics state (`military_strength`) and may remain inside the system or be derived from `Σ recruitments − casualties` across the chronicle.
- **Required NPC actions (leader):** `DeclareWar(opponent)`, `ReclaimSettlement(target)`, `LaunchConquest(target_settlement)`, `RaiseTaxes(rate)`, `RecruitMilitia()`, `SignPeace(opponent)`.
- **Required derived views:** `military_strength(f) = base + Σ recruitments − Σ casualty_events` (or retained as a primary field for perf, updated only as `RecruitmentHappened` / `UnitDied` events apply).
- **Required event types:** `WarDeclared`, `SettlementConquered`, `ReclaimAttempted`, `MilitaryRecruited`.
- **One-line:** Leader decision head stays; all war/conquest branches become leader NPC actions.

### diplomacy
- **Old:** ESSENTIAL (modifies `relationship_to_guild`, owns trade income).
- **New:** EMERGENT.
- **Reframe:** `diplomacy.rs:21-70` iterates faction pairs, auto-adjusts `relationship_to_guild` for peaceful pairs and runs trade income. Each threshold crossing ("relation > 20 → trade income") is a *diplomatic outcome* — what actually happened in the world is that a **leader NPC of faction A proposed a trade accord** which B accepted. Relation drift under a shared threat is the result of either leader choosing `MutualDefenseTalks(third_party_threat)`. Trade income is a recurring settlement-level consequence of a standing `TradeAgreementSigned` event.
- **Required NPC actions (leader):** `ProposeTradeAccord(target)`, `OpenDiplomaticChannel(target)`, `OfferGift(target, amount)`, `BreakRelations(target)`.
- **Required derived views:** `relation(f_a, f_b, tick) = sum of TradeAccord, Alliance, WarDeclaration, GiftSent, Betrayal events with recency decay`.
- **Required event types:** `TradeAccordSigned {a, b, terms}`, `DiplomaticGiftSent {from, to, amount}`, `RelationsBroken {a, b, cause}`.
- **One-line:** Every diplomacy threshold is the consequence of a leader action, not a background drift.

### espionage
- **Old:** ESSENTIAL (drains enemy `military_strength`, applies spy damage).
- **New:** EMERGENT.
- **Reframe:** `espionage.rs:68-100` walks hostile-faction NPCs near enemy settlements and auto-fires a drain-and-get-caught roll. This is a **commoner NPC action** (`Spy(target_faction)`) — any NPC with `STEALTH` / `DECEPTION` tags can choose to spy when positioned near an enemy town. The drain-to-enemy-strength is the consequence of a `SpyMissionSucceeded` event. The detection/wound path is the enemy settlement's *counter* action or a derived "detection chance" view, not a separate system.
- **Required NPC actions (commoner):** `Spy(target_faction)`, `Sabotage(target_settlement)`, `InfiltrateCouncil(target_faction)`.
- **Required derived views:** `spies_active(f) = Σ SpyMissionStarted − SpyMissionEnded for faction f`.
- **Required event types:** `SpyMissionStarted {agent, target_faction}`, `SpyMissionSucceeded {agent, target, impact}`, `SpyCaught {agent, defender_settlement}`.
- **One-line:** Spying is an NPC action, not a top-down drain.

### counter_espionage
- **Old:** ESSENTIAL (kills/wounds spies).
- **New:** EMERGENT (settlement-leader or guard-NPC reaction) + DERIVABLE-VIEW (detection chance).
- **Reframe:** `counter_espionage.rs:51-185` lets each settlement passively roll detection against nearby hostiles and kills/wounds them. The action model is: settlement leader NPC chooses `StandingOrder.Counterspy` once; whenever an enemy spy event fires nearby, a guard NPC action `ArrestSpy(agent)` or `ExecuteSpy(agent)` resolves. Morale boosts for allies come from the `SpyExecuted` chronicle entry, not from a system-wide broadcast.
- **Required NPC actions (leader/guard):** `SetStandingOrder(Counterspy, level)`, `ArrestSpy(agent)`, `ExecuteSpy(agent)`, `ExileSpy(agent)`.
- **Required derived views:** `detection_strength(s) = 0.08 + sqrt(pop)*0.04 + outpost_bonus` (kept as a formula used by the arrest action's resolve check).
- **Required event types:** `SpyArrested {agent, settlement}`, `SpyKilled {agent, settlement}`.
- **One-line:** Counter-intel is a standing order + reactive guard action, not a global sweep.

### war_exhaustion
- **Old:** STUB/DEAD (body commented out).
- **New:** DEAD + DERIVABLE-VIEW (when revived).
- **Reframe:** `war_exhaustion.rs:45-236` is entirely commented out. Don't revive it as a system — it is definitionally a derived view: `exhaustion(f) = duration_at_war × casualty_rate × treasury_drain_rate`. A leader's `SignPeace` action can *consult* this view but no system needs to write it.
- **Required NPC actions:** none.
- **Required derived views:** `war_exhaustion(f) = Σ_over_current_war (casualty_events × w1 + treasury_deltas × w2 + duration × w3)`.
- **Required event types:** none new (reads existing casualty / treasury / war events).
- **One-line:** Delete the system; keep the formula as a view consulted by `SignPeace`.

### civil_war
- **Old:** ESSENTIAL (mutates `escalation_level`, spawns crisis).
- **New:** EMERGENT.
- **Reframe:** `civil_war.rs:55-236` auto-ignites civil war when strength < 30 & unrest > 0.70 and then drains state until collapse/loyalist-win. Civil war is *always* a chain of NPC choices: a dissident leader NPC chooses `DeclareCivilWar(loyalist_side)`, citizens choose `JoinFactionSide(rebel|loyalist)`, the outcome resolves when one side's adherents drop below a threshold (derived). `escalation_level` becomes `civil_war_phase(f) = derive_from(latest CivilWarDeclared − resolution_events)`.
- **Required NPC actions:** `DeclareCivilWar(target_leader)` (leader/council), `JoinFactionSide(side)` (any citizen), `FleeCivilWar(destination)` (citizen), `SurrenderCivilWar()` (side-leader).
- **Required derived views:** `civil_war_status(f) = { phase, rebel_support, loyalist_support, duration }` from the action events.
- **Required event types:** `CivilWarDeclared {faction, instigator, grievance}`, `CivilWarSideJoined {citizen, side}`, `CivilWarResolved {victor, mechanism}`.
- **One-line:** Civil war is a cascade of faction-join actions around an inciting declaration.

### council
- **Old:** EMERGENT-CANDIDATE.
- **New:** EMERGENT.
- **Reframe:** `council.rs:39-99` tallies NPC faction distribution per settlement and nudges relationships/morale. Under the strict rubric, the *vote* is the core action: council members of a settlement each emit `VoteOnIssue(issue, choice)`, and the outcome is the majority view derived from the event log. The relationship/morale nudges are consequences of the resolved-vote event, not a continuous nudge.
- **Required NPC actions (council-member):** `VoteOnIssue(issue_id, choice)`, `AbstainVote(issue_id)`, `TableMotion(text)`, `FilibusterMotion(issue_id)`.
- **Required derived views:** `council_outcome(s, issue_id) = argmax(votes_by_choice)`, `council_composition(s) = histogram(npc.faction_id for home_settlement_id==s)`.
- **Required event types:** `CouncilVoteCast {voter, issue, choice}`, `CouncilMotionResolved {issue, outcome}`.
- **One-line:** Council is pure NPC voting — the system is a derived vote-tally, not a continuous relationship drift.

### coup_engine
- **Old:** ESSENTIAL (owns `coup_risk`, forces regime change).
- **New:** EMERGENT.
- **Reframe:** `coup_engine.rs:33-127` accumulates a `coup_risk` scalar per faction, then rolls for a coup attempt. A coup is the archetypal NPC action: an ambitious member of the ruling class chooses `LaunchCoup(target_leader)`. The success/failure branch resolves based on immediate state (loyalty, garrison, outside support), not a pre-rolled `coup_risk` scalar. Risk factors (unrest, treasury, escalation) become the *utility inputs* the ambitious NPC uses when deciding whether to act, not a stored resource.
- **Required NPC actions (ambitious member):** `LaunchCoup(target_leader)`, `PledgeLoyalty(leader)`, `BribeGarrison(amount)`, `FleeAfterFailedCoup(destination)`.
- **Required derived views:** `coup_conditions(f) = {unrest_avg, treasury_ratio, escalation, leader_legitimacy}` — fed as inputs to the ambitious NPC's action score.
- **Required event types:** `CoupAttempted {instigator, target_leader, success}`, `CoupSuppressed {instigator, defender}`, `RegimeChanged {faction, new_leader, mechanism}`.
- **One-line:** Coup is an action; `coup_risk` becomes the utility function evaluating it.

### defection_cascade
- **Old:** STUB/DEAD (body commented out).
- **New:** DEAD + EMERGENT (when revived).
- **Reframe:** `defection_cascade.rs:35-163` is gone. Its planned behaviour — one NPC defects, bonded allies follow — is not a system but a **cascade of `Defect(new_faction)` actions** triggered by social-bond utility spikes when a friend defects. The depth cap of 3 becomes a rule inside each NPC's decision model.
- **Required NPC actions:** `Defect(new_faction)`, `OfferDefectionBribe(target, amount)`.
- **Required derived views:** `defection_chain(seed_npc) = BFS over bond graph starting from seed until utility threshold fails`.
- **Required event types:** `NPCDefected {npc, old_faction, new_faction, trigger}`.
- **One-line:** Cascade is `Defect` actions propagating along bonds — no system.

### alliance_blocs
- **Old:** ESSENTIAL (TransferGold, military buffs).
- **New:** EMERGENT.
- **Reframe:** `alliance_blocs.rs:31-120` auto-transfers gold between friendly factions and buffs shared military strength. Every transfer is the executed term of a standing `AllianceTreaty` — the leaders *chose* to form the alliance (`FormAlliance`) and *chose* its aid clauses. The periodic aid transfers are the scheduled execution of the treaty's terms, best modelled as an `AllianceAidTickDue` timer plus the accepting leader's `DisburseAllianceAid` action, or inlined into a deterministic consequence of the treaty event.
- **Required NPC actions (leader):** `FormAlliance(partners, terms)`, `BreakAlliance(target)`, `DisburseAllianceAid(target, amount)`, `CallAllianceToWar(enemy)`.
- **Required derived views:** `is_ally(a, b) = exists(AllianceFormed without later AllianceBroken)`, `bloc(f) = transitive closure of current alliances`.
- **Required event types:** `AllianceFormed {factions, terms}`, `AllianceBroken {former_partners, breaker}`, `AllianceAidSent {from, to, amount, reason}`.
- **One-line:** Alliances are signed treaties with scheduled aid events, not a passive transfer loop.

### vassalage
- **Old:** ESSENTIAL (tribute transfers, rebellion surges).
- **New:** EMERGENT.
- **Reframe:** `vassalage.rs:53-290` handles auto-vassalage (weak factions drift toward strong ones), tribute, and rebellion. Each of these is an action: `SwearVassal(lord)` on the petitioner's side, `AcceptVassal(petitioner)` on the suzerain's, `RemitTribute(amount)` by the vassal each cycle, `Rebel()` when resentment boils over. The "drift" in relationship during auto-vassalage is post-hoc narrative for what was really a sequence of diplomatic overtures and pledges.
- **Required NPC actions (leader):** `SwearVassal(lord)`, `AcceptVassal(petitioner)`, `DemandTribute(vassal, amount)`, `RemitTribute(lord, amount)`, `Rebel(lord)`, `ReleaseVassal(vassal)`.
- **Required derived views:** `vassal_of(f) = chain of latest VassalageOath − ReleaseVassal events`.
- **Required event types:** `VassalageOathSworn {vassal, lord, terms}`, `TributeRemitted {vassal, lord, amount}`, `VassalRebelled {rebel, former_lord}`.
- **One-line:** Vassalage is an oath + tribute action pair; rebellion is also an action.

### faction_tech
- **Old:** STUB/DEAD (body commented out).
- **New:** DERIVABLE-VIEW.
- **Reframe:** `faction_tech.rs:32-189` is all commented out. Rather than store tech levels, compute them: `faction_tech_level(f, axis) = sum of TechInvestment events for that axis by that faction`. The "investment" is the leader action `InvestInTech(axis, amount)`. Milestone bonuses become derived multipliers any consumer system references (e.g., combat ability modifiers).
- **Required NPC actions (leader):** `InvestInTech(axis, amount)`, `PoachScholar(target_faction)`, `FoundAcademy(settlement)`.
- **Required derived views:** `faction_tech(f, axis) = Σ InvestInTech deltas with diminishing returns` ; `has_tech_milestone(f, axis, tier) = faction_tech(f, axis) ≥ tier_threshold`.
- **Required event types:** `TechInvested {faction, axis, amount}`, `TechMilestoneCrossed {faction, axis, tier}` (optional — can be derived).
- **One-line:** Tech level = `Σ InvestInTech`, no mutable tech field needed.

### warfare
- **Old:** ESSENTIAL (SOT for inter-faction war state, declares/ends wars).
- **New:** EMERGENT.
- **Reframe:** `warfare.rs:25-164` auto-declares war when the sum of cross-faction NPC grudges exceeds 50. Grudges exist and can be tallied — but *the decision to declare war* is a leader NPC action. Leader's utility consults the grudge tally (derived view), the current treasury, strength differential, and chooses. Peace is analogously `SignPeace(enemy)` by a leader when exhaustion view fires.
- **Required NPC actions (leader):** `DeclareWar(opponent, casus_belli)`, `SignPeace(enemy, terms)`, `RatifyTreaty(draft)`, `DemandReparations(enemy, amount)`.
- **Required derived views:** `grievance_matrix(a, b) = Σ over npcs_of(a) Grudge beliefs toward npcs_of(b)`, `at_war(a, b) = exists(WarDeclared(a,b)) without later PeaceSigned(a,b)`.
- **Required event types:** `WarDeclared {aggressor, defender, casus_belli, tick}`, `PeaceSigned {former_combatants, terms, tick}`.
- **One-line:** Leader chooses; grievances are *inputs* to that choice, not the cause.

### succession
- **Old:** ESSENTIAL (transfers leadership, may flip runner-up to Hostile).
- **New:** EMERGENT (+ ESSENTIAL bookkeeping for team flip, until team is derived).
- **Reframe:** `succession.rs:16-152` hard-promotes the top-level-sum NPC whenever a leader dies and optionally flips the runner-up to `Hostile`. Under strict: the leader death emits `LeaderDied`; each senior council member then chooses `VoteForSuccessor(candidate)`, and the runner-up chooses `AcceptSuccessor()` or `Rebel()`. The "top candidate auto-wins" is replaced by the *council-vote reducer*, identical to `council` but scoped to a successor question.
- **Required NPC actions (council / candidate):** `VoteForSuccessor(candidate)`, `DeclareCandidacy()`, `AcceptSuccessor(winner)`, `Rebel()` (reused from civil_war), `Abdicate()` (incumbent choice before death).
- **Required derived views:** `current_leader(s) = most-recent Succeeded event for s`.
- **Required event types:** `LeaderDied {settlement, predecessor}`, `SuccessorVoteCast {voter, candidate}`, `LeaderSucceeded {settlement, predecessor, successor, mechanism}`.
- **One-line:** Council votes choose the successor; Rebel is a citizen action.

### legends
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `legends.rs:18-175` detects legendary NPCs from `(chronicle_mentions ≥ 5, classes ≥ 2, friend_deaths ≥ 3)`, renames them to `" the Legendary"`, boosts settlement morale, and triggers world-wide mourning on their death. Legend status is **100% derivable** — the name mutation is a cache. The settlement morale bonus is a view consulted by mood calculations (`settlement_legend_halo(s) = +0.06 if a legend lives there`). The world-mourning effect *is* a one-shot state change, but it is the consequence of the existing `EntityDied` event plus the derived `is_legendary` predicate — no system logic needed beyond "emit `LegendMourned` event" when an `EntityDied` lands on a legend.
- **Required NPC actions:** none.
- **Required derived views:** `is_legendary(npc) = mention_count(npc) ≥ 5 AND class_count(npc) ≥ 2 AND friend_deaths(npc) ≥ 3`; `legend_halo(s) = max_legend_presence_bonus for home_settlement==s`; `legendary_name(npc) = base_name + " the Legendary" if is_legendary else base_name`.
- **Required event types:** `LegendAscended {npc, criteria}` (optional, for narrative), `LegendMourned {npc, tick}` (derived from `EntityDied` + predicate).
- **One-line:** Legend is a predicate; the mass-grief is a consequence of `EntityDied` when the predicate holds.

### prophecy
- **Old:** ESSENTIAL.
- **New:** EMERGENT (issuance) + DERIVABLE-VIEW (fulfillment).
- **Reframe:** `prophecy.rs:19-245` stores `state.prophecies[]` with a `fulfilled` bool. Issuance is a prophet NPC action `IssueProphecy(condition, effect)`. Fulfillment is a pure query: `fulfilled(p) = exists(event in world_events post p.tick matching p.condition)`. The dramatic effects (monster surge, morale boost, stockpile gift) are one-shot state transitions — they are the *consequence* of the fulfillment event `ProphecyFulfilled`, which can be emitted by a thin reducer rather than a 245-line system.
- **Required NPC actions (prophet):** `IssueProphecy(condition, effect)`, `Recant(prophecy_id)`, `InterpretProphecy(target_prophecy, reading)`.
- **Required derived views:** `prophecy_fulfilled(p) = exists_event_matching(p.condition, tick > p.issued_tick)`, `active_prophecies = prophecies where !fulfilled`.
- **Required event types:** `ProphecyIssued {prophet, condition, effect, tick}`, `ProphecyFulfilled {prophecy_id, tick, trigger_event}`.
- **One-line:** Prophecy is issue-once + derivable-predicate; fulfillment consequences fire from a generic reducer.

### outlaws
- **Old:** ESSENTIAL (flips team to Friendly on redemption; raids transfer gold).
- **New:** EMERGENT.
- **Reframe:** `outlaws.rs:15-179` handles three phases: raid, camp detection, redemption. Every phase is an NPC action: `Raid(target_npc)` by the outlaw, `FormBandit Camp(cluster_leader)` by a self-appointed camp leader, `SeekRedemption(settlement, amount)` by the outlaw himself. The team flip to Friendly is the resolution of `RedemptionAccepted` by the target settlement's leader. "Becoming an outlaw" is a first-class action `BecomeOutlaw()` chosen by disaffected NPCs or forced by `Exile(target)` from a settlement leader.
- **Required NPC actions (commoner):** `BecomeOutlaw()`, `Raid(target_npc)`, `FormBanditCamp(members)`, `SeekRedemption(settlement, gold_offer)`, `JoinBanditCamp(camp_id)`.
- **Required NPC actions (leader):** `Exile(target_npc)`, `AcceptRedemption(outlaw, payment)`, `PostBounty(outlaw, reward)`.
- **Required derived views:** `is_outlaw(npc) = latest of {BecameOutlaw, RedemptionAccepted} is BecameOutlaw`, `bandit_camps = cluster(outlaw_positions, radius)`.
- **Required event types:** `BecameOutlaw {npc, trigger}`, `RaidSucceeded {outlaw, victim, gold}`, `RedemptionAccepted {outlaw, settlement, payment}`.
- **One-line:** Outlaw status is event-derived; every state change is an action.

### settlement_founding
- **Old:** ESSENTIAL (spawns new settlement entity + colonist reassignment).
- **New:** ESSENTIAL *(spawn bookkeeping only)* + EMERGENT *(the decision)*.
- **Reframe:** `settlement_founding.rs:20-178` checks overcrowding and auto-launches 8 colonists. The decision — *should we leave?* — is a **commoner + aspiring-leader action**: a charismatic NPC chooses `LeadFoundingExpedition(target_region)`, and fellow residents choose `JoinExpedition(leader)`. The actual `state.settlements.push(new_s)` + `entity_id = next_id()` allocation at the moment of arrival is the irreducible bookkeeping step. The colonist reassignment (`home_settlement_id = new_s.id`) is the consequence of `ExpeditionArrived`.
- **Required NPC actions (leader-candidate):** `LeadFoundingExpedition(target_region)`, `ReturnFromFailedExpedition()`.
- **Required NPC actions (commoner):** `JoinExpedition(leader)`, `RefuseExpedition(leader)`.
- **Required derived views:** `is_overcrowded(s) = pop(s) / housing(s) > 1.5`, `viable_target_regions = settleable regions > min_dist from existing settlements`.
- **Required event types:** `ExpeditionLaunched {leader, members, target}`, `ExpeditionArrived {members, new_settlement_id}` (spawn point), `ExpeditionFailed {leader, cause}`.
- **One-line:** The decision is NPC choice; only the entity-ID allocation on arrival is essential.

### betrayal
- **Old:** ESSENTIAL (flips team, steals treasury, seeds grudges).
- **New:** EMERGENT.
- **Reframe:** `betrayal.rs:26-137` auto-selects treacherous NPCs, steals treasury, flips team, seeds grudges. The entire thing is **one NPC action**: `Betray(faction)`. Its resolution is the consequence event `BetrayalCommitted`, which: (1) optionally transfers gold on the same tick if the betrayer also chose `StealTreasury(settlement)`, (2) moves the betrayer's `faction_id` (or emits `FactionMembershipChanged`), (3) provokes residents' `FormGrudge(betrayer)` reactions on subsequent ticks. The 50-grudge mass insert at end of `advance_betrayal` collapses into each resident NPC's own grudge-formation utility responding to the event.
- **Required NPC actions (commoner):** `Betray(faction)`, `StealTreasury(settlement, amount)`, `FleeAfterBetrayal(destination)`, `FormGrudge(target)` (reactive to BetrayalCommitted).
- **Required derived views:** `treachery_score(npc) = STEALTH_tag + DECEPTION_tag − compassion*penalty` used as utility input.
- **Required event types:** `BetrayalCommitted {traitor, victim_faction, theft_amount}`, `GrudgeFormed {holder, target, cause}`.
- **One-line:** Betrayal is one action; grudges are reactive actions.

### family
- **Old:** ESSENTIAL (marriage, birth entity creation).
- **New:** ESSENTIAL *(birth entity alloc)* + EMERGENT *(the decision to marry / have children)*.
- **Reframe:** `family.rs:26-252` auto-matches compatible NPCs at 15% and auto-creates children after 2000 ticks married at 5%. Under strict: two NPCs each choose `Marry(target)` (bi-directional consent), spouse_id update is the consequence. Child creation is choose `HaveChild(spouse)` — the entity `push` + new NPC ID allocation + blended profile is the irreducible bookkeeping at resolution. Marriage compatibility formula becomes the *utility input* both sides' decisions consult.
- **Required NPC actions (commoner):** `Marry(target)`, `Divorce(spouse)`, `HaveChild(spouse)`, `AdoptChild(candidate)`, `LeaveSpouse(destination)`.
- **Required derived views:** `marriage_compatibility(a, b) = 1 − |social_drive_a − social_drive_b| − 0.5*|compassion_a − compassion_b|`, `child_cap_reached(couple) = |children| ≥ 3`, `settlement_at_cap(s) = pop(s) ≥ 300`.
- **Required event types:** `MarriageFormed {spouses, settlement}`, `MarriageEnded {former_spouses, cause}`, `ChildBorn {parents, child_id, settlement}`.
- **One-line:** Decisions are actions; only `ChildBorn` entity-ID allocation + profile blend is essential bookkeeping.

### haunted
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `haunted.rs:22-138` clusters `EntityDied` positions, then pushes fear/anxiety to nearby NPCs and flips a `LocationDangerous` belief. The cluster is a derived query. The belief-push on exposed NPCs is better modelled as each NPC's perception loop: when a commoner NPC walks within range of a haunted site (derived from recent deaths), they choose (or auto-react) `FormBelief(LocationDangerous)`. No top-down system required; `is_haunted(pos)` is a pure view.
- **Required NPC actions:** `FormBelief(LocationDangerous(site_id))` (perception reaction).
- **Required derived views:** `is_haunted(pos, tick) = cluster(death_positions in world_events where tick − t < window).any(|c| |c − pos|² < 900 AND c.count ≥ 5)`, `haunted_sites(tick) = {centres of qualifying clusters}`.
- **Required event types:** none new — reads `EntityDied`. Optional `HauntedSiteRecognized {pos, first_tick}` for chronicle.
- **One-line:** `is_haunted(pos)` is a window query over `EntityDied`.

### world_ages
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `world_ages.rs:22-93` scans a 2400-tick chronicle window and names the era. Pure chronicle classifier with zero gameplay effect. Replace with an on-demand view `current_world_age(tick) = classify(chronicle_window_stats(tick − 2400, tick))` called by UI / save-game / biography exports.
- **Required NPC actions:** none.
- **Required derived views:** `current_world_age(tick)`, `age_history = running label over chronicle windows`.
- **Required event types:** none.
- **One-line:** Pure classifier view over chronicle windows.

### chronicle
- **Old:** EMERGENT-CANDIDATE (note: "this system IS the narrative log emitter, but only for treasury/pop milestones").
- **New:** ESSENTIAL *(chronicle.push is the event recording primitive)* + DERIVABLE-VIEW *(milestone-detect sub-function)*.
- **Reframe:** The **act of writing chronicle entries** is the only truly irreducible part of the entire batch — it is literally how events are recorded. The existing `chronicle.rs:16-96` does only milestone detection (treasury/pop thresholds) and is a mis-named derived-view system; rename that part to `chronicle_milestones` and treat it as a DERIVABLE-VIEW emitter over `state.settlements`. The `chronicle.push(...)` primitive — invoked by *every* action resolver to record what happened — is the essential kernel of the event-sourced architecture.
- **Required NPC actions:** none (chronicle is not an action — every action *emits* a chronicle entry via the resolver).
- **Required derived views:** `treasury_milestones(s) = crossings of TREASURY_THRESHOLDS by s.treasury history`, `population_milestones(s) = crossings of POPULATION_THRESHOLDS by s.population history`.
- **Required event types:** `ChronicleEntry` is the universal event record; specific categories remain (Economy, Narrative, Combat, etc.).
- **One-line:** `chronicle.push` is the kernel primitive; milestone detection is a derived view on top of settlement history.

### crisis
- **Old:** ESSENTIAL (owns settlement stockpile drain, status effects).
- **New:** DERIVABLE-VIEW + EMERGENT.
- **Reframe:** `crisis.rs:34-174` picks one of 4 crisis types per high-threat region and mutates state. The *pick* is hashed on region+tick — it is not modelling a decision, it is a scripted random effect. Reframe: a region "in crisis" is a **view** (`is_in_crisis(r) = r.threat_level > 70`); the *response* by in-region NPCs is a set of actions (`FleeCrisisRegion`, `ReinforceCrisisRegion`, `RationSupplies`, `RiseAsUnifier`). The damage and commodity drain are framed as deterministic environmental hazards derived from the region's threat level — not a separate system, but a per-tick hazard consultation by NPCs and buildings.
- **Required NPC actions:** `FleeCrisisRegion(destination)`, `ReinforceCrisisRegion(region)` (leader or hero), `RationSupplies(settlement)` (leader), `RiseAsUnifier()` (charismatic NPC).
- **Required derived views:** `is_in_crisis(r) = threat_level(r) > 70`, `crisis_hazard(r, npc_pos) = function of proximity, threat, region_type`.
- **Required event types:** `CrisisBegan {region, type, tick}` (when view tips), `UnifierRose {npc, region}`, `RefugeesFled {region, destination, count}`.
- **One-line:** Crisis is a view; the responses are NPC actions.

### difficulty_scaling
- **Old:** ESSENTIAL.
- **New:** DERIVABLE-VIEW (meta).
- **Reframe:** `difficulty_scaling.rs:27-165` computes a global power rating and rubber-bands with damage or heals. This is a meta-game controller, not diegetic. Either delete (we don't want rubber-banding in a zero-player world-sim), or keep as a pure view that UI/training loops consult: `player_power_rating = score(...)`. The damage/heal side-effects do not belong in the simulation.
- **Required NPC actions:** none.
- **Required derived views:** `power_rating(state) = f(friendly_count, avg_level, treasury, territory, pop, monster_density)`.
- **Required event types:** none.
- **One-line:** Delete the write-path; keep the scalar as a view for training/UI.

### charter
- **Old:** DUPLICATIVE.
- **New:** DEAD.
- **Reframe:** `charter.rs:14-46` is a treasury-feedback loop that v1 already flagged as a proxy (comments `:27-29`). The real-world thing it gestures at — a settlement *adopting a charter* — is an EMERGENT leader action `AdoptCharter(template_id)` that should exist in the leader action space. Delete the current system; add the action.
- **Required NPC actions (leader):** `AdoptCharter(template)`, `AmendCharter(clauses)`, `RevokeCharter()`.
- **Required derived views:** `charter(s) = latest AdoptCharter/RevokeCharter for s`, `charter_bonuses(s) = lookup of active charter`.
- **Required event types:** `CharterAdopted {settlement, template, tick}`.
- **One-line:** Delete the noise; the action-space covers it.

### choices
- **Old:** DUPLICATIVE.
- **New:** DEAD.
- **Reframe:** `choices.rs:15-39` emits random ±gold jitter per settlement. Source comments (`:22-24`) admit it is a placeholder. Under strict event-sourcing, every "choice outcome" is a resolved NPC/leader action — there is no place for system-level random jitter.
- **Required NPC actions:** none new (the whole concept dissolves into the general leader/commoner action space).
- **Required derived views:** none.
- **Required event types:** none.
- **One-line:** Delete.

### rival_guild
- **Old:** DUPLICATIVE.
- **New:** DEAD.
- **Reframe:** `rival_guild.rs:15-69` auto-fires random sabotage/raids. Source comments (`:35-36`) confirm it is a stub. The rival guild — if it exists — is another faction whose leader chose `Sabotage(target)` or `Raid(target)`. Delete and rely on the action space; if specific rival-guild flavour is wanted, seed a rival faction at worldgen.
- **Required NPC actions:** (covered by `Sabotage`, `Raid` in generic action space.)
- **Required derived views:** none.
- **Required event types:** none new.
- **One-line:** Delete; seed a rival faction at worldgen instead.

### victory_conditions
- **Old:** DUPLICATIVE.
- **New:** DEAD *(current impl)* + DERIVABLE-VIEW *(as designed)*.
- **Reframe:** `victory_conditions.rs:16-55` damages NPCs in high-threat regions after tick 15000 — source (`:26-28`) calls it an approximation. A true victory check is a pure view: `victory_state(state) = classify({conquest, economic, narrative, survival})`. No write path needed.
- **Required NPC actions:** none.
- **Required derived views:** `victory_state(state, faction_or_player)` covering conquest, economic, cultural, survival axes.
- **Required event types:** `VictoryAchieved {faction, type, tick}` emitted once by the view's tip-crossing.
- **One-line:** Delete write path; compute victory as a view.

### awakening
- **Old:** EMERGENT-CANDIDATE.
- **New:** EMERGENT (if kept) *or* DEAD (recommended, per v1 passive-buff-clone finding).
- **Reframe:** `awakening.rs:14-96` rolls 1% chance for level-8+ NPCs to become "awakened" with a permanent 1.25× buff. An NPC choosing to "awaken" is coherent — `SeekAwakening()` at a sacred site — but the current impl has no such narrative. If kept: make it an NPC action. If we are being strict: this is one of four near-identical passive-buff systems and should be deleted.
- **Required NPC actions (commoner):** `SeekAwakening(sacred_site)`, `UndergoTrial(type)`.
- **Required derived views:** `is_awakened(npc) = exists AwakeningGranted event for npc`.
- **Required event types:** `AwakeningGranted {npc, trigger}`.
- **One-line:** Make it an NPC action with a location/trial, or delete.

### visions
- **Old:** DUPLICATIVE.
- **New:** DEAD.
- **Reframe:** `visions.rs:15-78` applies a 1.05× "prophetic" buff; source comments (`:43-45`) admit it is a proxy. Identical shape to awakening/bloodlines/legacy_weapons. If the concept must survive, it is an NPC action `ReceiveVision()` emitted by a prophet — but redundant with `IssueProphecy`. Delete.
- **Required NPC actions:** none.
- **Required derived views:** none.
- **Required event types:** none.
- **One-line:** Delete.

### bloodlines
- **Old:** DUPLICATIVE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `bloodlines.rs:15-79` applies a 1.10× "bloodline" buff to level-8+ NPCs. Source comments (`:46-48`) admit no actual bloodline tracking. A real bloodline is the **HasChild graph** — purely derivable. The buff model should read from that graph: `bloodline_bonus(npc) = f(ancestral_legend_count_in_lineage(npc))`.
- **Required NPC actions:** none (parent/child are covered by `family`).
- **Required derived views:** `lineage(npc) = chase HasChild(parent, npc) upward via ChildBorn events`, `descendants(npc) = chase downward`, `lineage_prestige(npc) = Σ is_legendary(ancestor) for ancestor in lineage(npc)`.
- **Required event types:** none new — `ChildBorn` already recorded.
- **One-line:** Lineage is a chain query over `ChildBorn`; buff is a view on that chain.

### legacy_weapons
- **Old:** DUPLICATIVE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `legacy_weapons.rs:15-80` applies a 1.10× "legacy_weapon" buff. There is no weapon entity. Real legacy weapons = items with kill-count history = derivable from an item's `WeaponKill` event history. Bonus is a view on that.
- **Required NPC actions:** `WieldLegacyWeapon(item)`, `PassDownWeapon(heir, item)` (if item entities become part of the world).
- **Required derived views:** `legacy_weapons = items where kill_count(item) ≥ N`, `legacy_wielder_bonus(npc) = bonus if wielded item ∈ legacy_weapons`.
- **Required event types:** `WeaponKill {weapon_id, wielder, victim}`, `WeaponPassedDown {weapon_id, from, to}`.
- **One-line:** Legacy status = derived from kill-count events; no system write needed.

### artifacts
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `artifacts.rs:18-112` names an "artifact" in the chronicle for high-level multi-class NPCs — **no item entity is created** (comment notes this). Pure chronicle emission. Can be computed on demand: `artifact_of(npc) = derived name if level ≥ 35 AND classes ≥ 4 AND roll < 0.03`. If we want actual artifact items, introduce a `ForgeArtifact` action (leader or grandmaster NPC) and track the item — but the current v1 behaviour is just a view.
- **Required NPC actions (grandmaster):** `ForgeArtifact(material, dedication)` (optional future).
- **Required derived views:** `eligible_artifact_bearers = npcs where level ≥ 35 AND class_count ≥ 4`, `artifact_name(npc) = deterministic hash of (npc_id, tick_at_eligibility)`.
- **Required event types:** `ArtifactForged {smith, item, settlement}` (if item entities exist).
- **One-line:** Current system is a pure derivable view; real artifacts would need a `ForgeArtifact` action.

### great_works
- **Old:** ESSENTIAL (deducts treasury, commissions named monuments).
- **New:** EMERGENT.
- **Reframe:** `great_works.rs:12-89` auto-commissions monuments on wealthy, populous settlements, deducting treasury. Under strict: commissioning a great work is a **settlement-leader action** `CommissionGreatWork(type, cost)`, and `ContributeToGreatWork(amount)` is a commoner action for volunteer labour/donations. Completion is the reducer emitting `GreatWorkCompleted` when total contributions ≥ cost. Treasury deduct = consequence of the commission action.
- **Required NPC actions (leader):** `CommissionGreatWork(type, budget)`, `CancelGreatWork(project)`.
- **Required NPC actions (commoner):** `ContributeToGreatWork(project, amount)`, `VolunteerLabour(project, hours)`.
- **Required derived views:** `great_works(s) = ongoing + completed projects at s`, `total_works_completed(s) = count(GreatWorkCompleted where settlement==s)`.
- **Required event types:** `GreatWorkCommissioned {settlement, type, budget}`, `GreatWorkContribution {contributor, project, amount}`, `GreatWorkCompleted {settlement, project, type, tick}`.
- **One-line:** Leader commissions, commoners contribute, completion fires from the reducer.

### culture
- **Old:** EMERGENT-CANDIDATE.
- **New:** DERIVABLE-VIEW.
- **Reframe:** `culture.rs:18-142` reduces per-NPC behaviour tags into four culture axes (martial/mercantile/scholarly/spiritual) and applies settlement-level buffs and morale writes. This is a textbook reduction-over-residents view: `culture_mix(s) = normalized_sum(behaviour_value[axis_tags] over home_settlement_id==s)`. The "effects" (threat_level−, infra+, morale+) are views any consumer can apply — or, if we must cache, they are idempotent derived writes refreshed from the view each tick.
- **Required NPC actions:** none.
- **Required derived views:** `culture(s) = {martial_pct, mercantile_pct, scholarly_pct, spiritual_pct}`, `dominant_culture(s) = argmax of culture(s)`, `culture_modifier_threat(s) = −(martial_pct − 30)*0.01 if martial_pct > 30`, `culture_morale_bonus(s, npc) = +(spiritual_pct − 30)*0.01 if spiritual_pct > 30`.
- **Required event types:** `CultureShifted {settlement, new_dominant, tick}` (optional narrative emission when dominant crosses over).
- **One-line:** Culture is a reduction-over-residents view; every "effect" is that view consulted by another system.

---

## Reduction summary

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

### Counts

| Class | v1 | v2 |
|---|---:|---:|
| ESSENTIAL | ~14-18 | **4** (chronicle.push, settlement_founding.spawn, family.birth alloc, faction_ai *as decision head*) |
| EMERGENT | 0 (not a class) | **14** (diplomacy, espionage, counter_espionage, civil_war, council, coup_engine, alliance_blocs, vassalage, warfare, succession, prophecy issuance, outlaws, betrayal, great_works, + the action-branches of faction_ai / settlement_founding / family) |
| DERIVABLE-VIEW | ~7 | **11** (faction_tech, legends, haunted, world_ages, culture, crisis, difficulty_scaling, bloodlines, legacy_weapons, artifacts, war_exhaustion formula, victory_conditions, prophecy fulfillment) |
| DEAD | ~3 | **7** (war_exhaustion, defection_cascade, faction_tech *write*, charter, choices, rival_guild, victory_conditions *write*, visions, awakening) |

(Totals sum to >36 because several systems split across buckets: `chronicle` is both ESSENTIAL-kernel and DERIVABLE-VIEW-milestones; `family` is ESSENTIAL-birth + EMERGENT-decisions; `prophecy` is EMERGENT-issuance + DERIVABLE-VIEW-fulfillment; `settlement_founding` is EMERGENT-decision + ESSENTIAL-spawn.)

---

## Required action vocabulary (split by NPC role)

### Common NPC actions
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

### Leader NPC actions (faction / settlement level)
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

## Required event types

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

## Required derived views

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

## Truly essential (irreducible) set in this batch

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
