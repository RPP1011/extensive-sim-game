# Stories H — Player-experienced behaviors (acceptance tests)

This batch validates that the DSL supports **the gameplay the player will experience**. Each story is an end-to-end acceptance test: observation → policy decision → action → cascade → emergent outcome. Stories that cannot be traced end-to-end surface real gaps in the architecture.

Schemas referenced:
- `proposal_policy_schema.md` — observation, action space, mask, reward
- `proposal_universal_mechanics.md` — PostQuest / AcceptQuest / Bid vocabulary
- `state_npc.md`, `state_aggregate.md`, `state_world.md` — field-level state

---

### Story 45: Wars are motivated
**Verdict:** PARTIAL (grievance-driven path supported; cold-start + expansion paths need work)

**User's framing note:** "Who would start the first war? What about wars for expansion?" The grievance-threshold mask predicate doesn't bootstrap on an empty chronicle, and doesn't explain wars that aren't reactive at all.

**End-to-end trace (grievance case — the easy one):**

1. **Observation (leader of G_self):**
   - `self.creature_type`, `self.ambition` (personality, 5-d)
   - `memberships[K=8]` — includes G_self with `my_role_in_group = Leader`
   - `known_groups[K=6]` — G_rival with `standing_with_me = Tense`, `group_strength_log`
   - `nearby_actors` / `known_actors` — members of G_rival I recognise as sources of past grievances
   - `recent_memory_events[K=12]` — past `WasAttacked / FriendDied / BetrayalWitnessed` events whose `entity_ids` resolve back to G_rival members
   - **Derived view** `grievance_matrix(G_self, G_rival)` — sum of negative memory events of G_self members toward G_rival members with recency decay. Feeds observation as a scalar feature on the `known_groups[G_rival]` slot. The policy schema alludes to this at §2.1.5 (`n_opposed_groups`) and in `systems_politics.md:131` (`grievance_matrix(a, b) = Σ over npcs_of(a) Grudge beliefs toward npcs_of(b)`).
   - World context: `tick_normalized`, season, treasury, military_strength.

2. **Policy:** emits
   ```
   PostQuest{
     type=Conquest,
     party_scope=Group(G_self),
     target=Group(G_rival),
     reward=Spoils(territory),
     deadline=tick+5000
   }
   ```

3. **Mask (gate):**
   ```
   PostQuest{type=Conquest, party=Group(g), target=Group(t)}
     when g ∈ self.leader_groups               // I lead G_self
        ∧ g.kind ∈ {Faction, Pack}              // only factions/packs wage Conquest
        ∧ g.military_strength > 0               // can actually fight
        ∧ g.standings[t] ≠ AtWar                // not already at war
        ∧ eligible_conquest_target(self, t)     // OPEN — see below
   ```

4. **Cascade:**
   - `QuestPosted{quest_id, type=Conquest, party_scope=Group(G_self), target=Group(G_rival)}` fires.
   - **Cascade handler** updates `G_self.standings[G_rival] = AtWar` and reciprocally `G_rival.standings[G_self] = AtWar`. (This replaces the old `faction.at_war_with.push()` side effect — see `systems_politics.md:131`.)
   - `G_self.active_quests.push(quest_id)`.
   - Members of G_self see the new `active_quests[K=4]` slot on next tick. Their `at_war` view flips. Per-tick policy emits micro primitives (`MoveToward(rival_settlement)`, `Attack(...)`) gated by the war-context.
   - `known_groups[G_rival]` standing flips in the observation of every member.

5. **Emergent outcome (player-visible):**
   - Raids, battles, settlement conquest events land in the chronicle.
   - Treasury drains (units + upkeep). Casualty events accumulate.
   - Eventually either `QuestCompleted{type=Conquest}` → Spoils cascade, or `QuestExpired` → war-exhaustion view fires → leader emits `PostQuest{type=Peace}` / `Bid` in a diplomatic auction.

**Cold-start case (fresh world, no chronicle):**

The grievance path breaks because `grievance_matrix(a, b) ≈ 0` for every (a, b) pair. Something has to break the symmetry or the world is frozen in peace forever.

Three complementary mechanisms that *together* bootstrap first-ever wars from the observation/mask already in the schema:

- **(a) Personality-driven bellicosity.** `self.personality.ambition` is a primary input, drifted by events. Seeded at spawn. A high-ambition leader with high `known_groups[G_other].group_strength` contrast and adjacent territory chooses `PostQuest{Conquest, target=G_other}` with nonzero probability even at `grievance=0`. The reward function biases this: `+Δ(fame_log)`, `+Δ(self.esteem)`, and `+Spoils` on `QuestCompleted`. Without any grievance, ambition + a weaker rival is enough for a ratty leader to open hostilities.

- **(b) Territory contestability feature.** Add an observation feature surfacing contestable-territory signal so the policy can see "expansion is profitable" without depending on a chronicle. Proposed: on `known_groups[K=6]` slot, add
  - `controls_resource_i_lack` (boolean/soft): whether the group controls a biome/commodity our home_settlement is short on
  - `is_adjacent_territory` (boolean)
  - `military_strength_ratio` (signed log)
  - `standing_with_me_one_hot` already exists

  These let a leader observe "there's a weaker group sitting on food I need" as raw features, not as an accumulated grievance.

- **(c) Ambition as a reward-shaping signal.** The reward block already includes `+Δ(self.fame_log)` and `+2.0 on QuestCompleted`. That's enough to create a learned gradient toward "I gain by expanding my group." A learned ambition-biased leader converges on expansion wars during training even when grievances are nil.

**Expansion-for-expansion's-sake case:**

Same observation, different triggering features:
- `G_self.military_strength_ratio_vs[G_other]` high
- `territory_size(G_self)` small
- `G_other.controls_resource_i_lack` true
- `self.personality.ambition` high
- `self.needs.esteem` low (ambition unmet)

Policy emits `PostQuest{Conquest, target=G_other}` purely because the observation + reward gradient points to it. The mask passes because G_other isn't already at war with us. No grievance required.

**How the DSL supports this:**

- `PostQuest{type=Conquest, party_scope=Group(g)}` — universal action (proposal doc §1)
- Mask predicate for Conquest on leader groups (proposal doc §2.3)
- Rich observation includes `known_groups[K=6]` with standing/strength/size, agent personality, self emotions/needs (§2.1)
- Reward block sums `Δ(fame_log)`, `QuestCompleted+2`, need-recovery — sufficient to shape ambition-driven wars if data is balanced.

**Implementation walkthrough:**

1. Extend `known_groups[K=6]` slot features with `military_strength_ratio`, `controls_scarce_resource`, `is_adjacent`. These are cheap derived views over `Group.military_strength`, `Group.stockpile`, spatial query over territory cells.
2. Add the `eligible_conquest_target(self, other)` mask predicate. Minimum gate: `other.kind ∈ {Faction, Pack}` ∧ `other.alive` ∧ `standings[other] ≠ Vassal` ∧ (optional) `distance_to_territory(other) < CAMPAIGN_RANGE`.
3. Add cascade rule `QuestPosted{type=Conquest}` → set both groups' `standings = AtWar`.
4. Confirm reward includes `+Δ(self.personality.ambition × fame_gain)` or leave as pure fame — during training the ambition dimension becomes correlated with expansion behaviour.
5. During bootstrap (utility backend), hand-score `Conquest` with a small base `ambition × military_ratio × need_esteem_gap` term so the initial trajectories have some expansion wars, then distill into the neural backend.

**Gaps / open questions:**

- **No "war declaration event" distinct from `QuestPosted{Conquest}`.** Consumer systems (UI chronicle, diplomatic cascade) currently hunt for `WarDeclared`. Proposal: `QuestPosted{type=Conquest}` cascade rule emits a derived `WarDeclared{a, b, casus_belli}` chronicle entry. Casus belli is read from `QuestPosted.terms` (new field), defaulted to `Grievance(count)` when `grievance_matrix > 0` else `Expansion`.
- **No opposite-side auto-acceptance.** When G_self posts Conquest against G_rival, G_rival's leader doesn't need to `AcceptQuest` — they get drafted into a war as defender. Proposal: cascade fires a *separate* auto-generated `Defend` quest on G_rival's side (`PostQuest{type=Defend, party=Group(G_rival), target=Location(G_rival.home)}`), implicitly accepted by leader. Members see it in `active_quests[K=4]`.
- **`territory_size` + `controls_scarce_resource` aren't in the current observation spec.** Need to append to `known_groups` slot schema. Append-only per the versioning rule (§4 of schema).
- **Casus belli enum.** `Grievance`, `Expansion`, `Revenge`, `ReligiousDifference`, `Succession`, `Reclaim`. Each slot on `QuestPosted` becomes narrative fuel.
- **Credit assignment.** Conquest reward lands 2000+ ticks after the decision. Flagged as open in schema §6 (Q14). Not unique to this story but biting here.
- **Minimum-viable extension:** schema additions for `known_groups` (`military_strength_ratio`, `is_adjacent_territory`, `controls_scarce_resource`), mask predicate `eligible_conquest_target`, cascade rule `QuestPosted{Conquest} → WarDeclared + Defend counter-quest`, and a hand-scored `ambition_expansion_utility` in the utility backend for bootstrap. Total: ~1 week.

**Related stories:** 46 (succession — leader chosen for ambition reshuffles war chances), 49 (betrayal — war stress amplifies defections), 51 (long war → economy), 54 (alliance bidding — wars trigger mutual-defense auctions).

---

### Story 46: Succession dynamics on leader death
**Verdict:** PARTIAL → SUPPORTED once the observation exposes a "leader vacancy" feature. User is right that succession should be emergent; no special cascade rule is needed.

**User's framing note:** "Should be emergent." No hand-coded succession system. Agents must observe "leader vacancy" and compete via normal actions.

**End-to-end trace:**

1. **Trigger event.** `EntityDied{entity_id=leader, cause=...}` fires (combat, assassination, disease — all existing events). Cascade handler updates `G.leader_id = None` and emits `LeaderVacant{group_id, predecessor_id, predecessor_role_tenure}` chronicle entry.

2. **Observation update (all senior members of G):**
   - `memberships[k]` for the group `G` now has `group.leader_id_absent = true` (new slot feature, boolean).
   - `known_actors[k]` slots for potential rivals (fellow officers) update with `is_competing_for_leadership = true` if they've already acted.
   - `recent_memory_events[K=12]` includes the `EntityDied{leader}` event, so every member who witnessed the death has it in their ring.
   - `self.standing_in_group` (already present in `memberships` slot) surfaces my own claim strength.
   - `self.personality.ambition` + `self.esteem_need` drive willingness to compete.

3. **Policy (claimant path):** A senior member with `role ∈ {Officer, Founder}`, high `standing_in_group`, high `ambition` emits:
   ```
   PostQuest{
     type=Submit,                    // invert the relation: "all members submit to me"
     party_scope=Group(G),
     target=Agent(self),             // I am the successor candidate
     reward=Reciprocal,              // "I lead, you benefit"
     terms=SuccessionClaim{predecessor=X}
   }
   ```
   or equivalently a new `Claim` quest type (`type=Claim, party=Group(G), target=Role(Leader)`). Both work; `Claim` is cleaner because `Submit` currently implies vassalage.

4. **Policy (voter path):** Other members observe `active_quests[K=4]` filled with one or more Claim quests. They emit:
   ```
   AcceptQuest{quest_id=claim_of_rival_A}
   ```
   as a vote. Mask passes `AcceptQuest` only if `self ∈ quest.eligible_acceptors` — here the eligibility predicate is `self ∈ G.members ∧ self.role ∉ {Outcast}`.

5. **Resolution (cascade):** When a Claim quest collects acceptance majority (either by `MutualAgreement` auction resolution over current members, or by hitting `party_member_ids.len() / G.members.len() > 0.5`), cascade fires `LeaderSucceeded{group_id, new_leader, predecessor, mechanism=Election}`. Losing claimants see their own quest `QuestExpired`, which hits their emotion.grief + grief_penalty reward. High-ambition losers may then emit `LaunchCoup` (→ `PostQuest{type=Coup, target=new_leader}`) or `LeaveGroup(G)` + `JoinGroup(G_rival)`.

6. **Mask (gate):**
   ```
   PostQuest{type=Claim, party=Group(g), target=Role(Leader)}
     when g ∈ self.memberships
        ∧ g.leader_id = None            // vacancy exists
        ∧ self.role_in(g) ∈ {Officer, Founder}
        ∧ tick - self.data.joined(g) > MIN_TENURE
   AcceptQuest{q}
     when self ∈ q.eligible_acceptors
        ∧ q.type = Claim
        ∧ q.group = g ∈ self.memberships
        ∧ ¬conflicts_with_my_active_quests(q)    // can't back two claimants
   ```

7. **Emergent outcomes player observes:**
   - **Uncontested succession:** only one senior member claims, others auto-accept → smooth transition. Chronicle: "After Lord X died, his nephew Y was crowned."
   - **Contested succession:** two officers claim simultaneously → partial acceptance on each → neither reaches majority → `QuestExpired` on both → frustrated claimants try again with better diplomacy (bribes via `Bid` in a private Political auction) or `LaunchCoup`.
   - **Coup:** a claimant who lost the vote PostQuest's Conquest against their own group (`party=cabal of my_faction_loyalists, target=Group(G)`). Runs as any civil war.
   - **Collapse:** no one claims (tenure predicate fails for all; group decapitated). `G.dissolved_tick = now` cascade rule when `G.leader_id = None` for > DECAPITATION_GRACE ticks and no claimants remain.

**How the DSL supports this:**

- `Group.leader_id: Option<AgentId>` field already exists (`state_aggregate.md:588`). Vacancy = `None`.
- `memberships[K=8]` slot exposes `my_role_in_group`, `my_standing_in_group` — enough for self-claim scoring (§2.1.3).
- `PostQuest{Claim}` + `AcceptQuest` + `AuctionResolution::MutualAgreement` = the full contested-vote machinery (proposal_universal_mechanics.md §3).
- `Group.governance: GovernanceKind` already differentiates `Hereditary / Elective / Council / Theocratic` (`state_aggregate.md:590`). The eligibility predicate on Claim quest reads governance to decide who can claim: Hereditary groups permit only `self ∈ G.leadership_chain`; Elective permits all officers; etc. This is data-driven — NO "succession system" — just richer mask predicates.

**Implementation walkthrough:**

1. Extend `QuestType` with `Claim` variant (as already listed in OPEN). Target type `Role(RoleTag)` — add to `QuestTarget`.
2. Add cascade rule: when `EntityDied{id}` fires and `id` appears as `leader_id` on any Group, set that Group's `leader_id = None` and emit `LeaderVacant{group_id}` chronicle entry.
3. Add `auto-dissolve` cascade: `G.leader_id = None ∧ tick - G.leader_vacated_tick > DECAPITATION_GRACE ∧ no_pending_claim(G)` → `DissolveGroup{G}`.
4. Add mask predicates for `PostQuest{Claim}` / `AcceptQuest{Claim}` keyed on `governance`.
5. Extend `memberships[k]` with two floats: `group_leader_vacant` (0/1) and `my_tenure_relative_to_other_seniors`. These are the features that cause the policy to "notice" the vacancy.
6. Reward component for successful Claim: `+Δ(self.esteem)`, `+Δ(fame_log)`, `+Δ(my_standing_in_group)`. For failed Claim: `+grief emotion`, `-self.esteem`. Shapes ambition-contested succession.

**Gaps / open questions:**

- **Need a `Claim` quest type** — currently not in `QuestType` enum (`proposal_universal_mechanics.md:115`). Using `Submit` with inverted target is semantically wrong; add `Claim`.
- **`Role(RoleTag)` target** not yet in `QuestTarget` enum (§"QuestTarget" in mechanics doc). Small addition.
- **No "senior officer" feature** — `role_in_group_one_hot` has `{Member, Officer, Leader, Founder, Apprentice, Outcast}`. Officer and Founder are the senior claim roles — sufficient.
- **Auction resolution for election.** `MutualAgreement` resolves when all targeted acceptors say yes — wrong semantics for majority vote. Need a new `AuctionResolution::Majority` or let Claim quests resolve via `FirstAcceptable` after N ticks with `winner = quest with most acceptances`. Auction state machine is already OPEN (schema §5, mechanics §"Auction state machine").
- **Governance-sensitive masks.** Hereditary → only next in `leadership_chain` can claim; Elective → all officers; Theocratic → charter-designated pool. These are ~5 predicate rules each gated by `governance` enum.
- **Minimum-viable extension:** add `Claim` QuestType + `Role` target + vacancy cascade + `Majority` auction resolution + 2 memberships slot features. ~2 weeks including the auction state machine work.

**Related stories:** 45 (a war-weary leader dies mid-conquest → succession contest during war), 49 (disgruntled loser defects), 50 (reputation determines who accepts whose claim), 52 (inherited traits bias claimant ambition).

---

### Story 47: Marriages by trust + compatibility
**Verdict:** SUPPORTED

**User's framing note:** "Sure."

**End-to-end trace:**

1. **Observation (proposer, Agent A):**
   - `self.spouse_id = None` (implicit in `has_spouse = false` on §2.1.2 "Social ties")
   - `self.creature_type.can_marry` (derived from species)
   - `known_actors[K=10]` — up to 10 known agents including potential mates. Per-slot features:
     - `relationship.trust`, `relationship.familiarity`
     - `relationship_kind_one_hot(8)` (Friend, Rival, Spouse, Kin, Mentor, Apprentice, Stranger, Sworn Enemy)
     - `perceived_personality` (embedded via the cross-block psychological view — proposal mentions `perceived_personality` at `state_npc.md:175`). Compatibility is a dot-product between `self.personality` and `perceived_personality.traits`.
     - `n_shared_groups` — cultural eligibility (same faction? same religion?)
     - `has_spouse` of other (so A doesn't propose to already-married agent)
   - `self.personality.social_drive` + `self.personality.compassion` — marriage propensity
   - `self.needs.social` low, `self.needs.esteem` depending on cultural status — drive
   - `memberships[k].group.kind = Family` of self — am I already in a Family?

2. **Policy:** emits
   ```
   PostQuest{
     type=Marriage,
     party_scope=Individual(A),
     target=Agent(B),
     reward=Union,
     terms=ExclusiveAcceptance,
     deadline=tick+500
   }
   ```

3. **Mask:**
   ```
   PostQuest{type=Marriage, party=Individual(self), target=Agent(t)}
     when self.creature_type.can_marry
        ∧ ¬has_spouse(self)
        ∧ ¬has_spouse(t)
        ∧ t.creature_type.can_marry
        ∧ t.alive
        ∧ relationship(self, t).trust > MIN_TRUST    // familiarity must exist
        ∧ marriage_eligibility(self, t)               // cultural rule (view)
                                                     // distance NOT required
   ```
   `marriage_eligibility` is a derived view over `memberships` + group eligibility predicates. Example: a Stone Clan religion only permits in-clan marriage → predicate rejects cross-clan pairings.

4. **Cascade (post):** `QuestPosted{quest_id, type=Marriage, party_member_ids=[B]}` — B is the sole eligible acceptor. Quest lands in B's `active_quests[K=4]` on next tick with `type=Marriage` and `posted_by=A`.

5. **Observation (acceptor, Agent B):**
   - `active_quests[k]` now surfaces a Marriage quest targeting me from A.
   - Same compatibility features in `known_actors[B's view of A]`: `trust`, `familiarity`, `perceived_personality`, shared groups.
   - B's own `personality`, `needs`, `emotions`.

6. **Policy:** emits either
   ```
   AcceptQuest{quest_id}
   ```
   or doesn't emit (passive refusal → quest expires at deadline).

7. **Cascade (accept):**
   - `QuestCompleted{quest_id}` fires.
   - Cascade handler for `Marriage` emits:
     - `MarriageFormed{a=A, b=B, tick}` (`systems_politics.md:197`)
     - `FoundGroup{kind=Family, founder_ids=[A, B]}` (or append both to existing family — if one spouse has kin group).
     - A.spouse_id = B.id; B.spouse_id = A.id; both get `memberships.push(Membership{group_id=new_family, role=Founder})`.
     - A.personality.social_drive, B.personality.social_drive += 0.02 (drift).
     - A.emotions.joy += 0.3, B.emotions.joy += 0.3.

8. **Observation (everyone else):** `known_actors` slots for A/B update `is_spouse`; Family group becomes known to both spouses' `known_groups[k]`.

**Emergent outcomes player observes:**
- Proposals cluster around NPCs with high familiarity + shared groups (work coworkers, adventuring party members).
- High-compassion NPCs marry high-compassion NPCs (trait homophily emerges from perceived_personality-weighted choice).
- Cross-faction marriages form political alliances (see story 54).
- Failed proposals cause grief on A (+0.15 emotion.grief on QuestExpired). Multiple rejections shift A's personality (curiosity drift, social drive drift down).

**How the DSL supports this:**

- `PostQuest{Marriage}` → `AcceptQuest` cascade fully specified in `proposal_universal_mechanics.md` §"How 'marriage is a quest' plays out" (:282).
- Observation features in `known_actors[K=10]` — `relationship_valence`, `relationship_familiarity`, `relationship_kind_one_hot`, `n_shared_groups` (§2.1.5).
- `self.personality` (5d) + `known_actors[K].perceived_personality` → compatibility dot-product computable as a derived feature within the `known_actors` slot.
- Mask predicate in §2.3 is explicit about marriage.
- `Group{kind=Family}` as the post-marriage container (`state_aggregate.md:638`).

**Implementation walkthrough:**

1. Add the `FoundGroup{Family}` cascade rule on `QuestCompleted{type=Marriage}` — already specified in mechanics doc.
2. Add `marriage_eligibility(a, b)` derived view reading both agents' membership predicates (religious/faction exclusivity). Data-driven from Group.eligibility_predicate.
3. Extend `known_actors[k]` slot with `compatibility_score` = cosine(self.personality, other.perceived_personality.traits × confidence). Purely derived; cheap.
4. On `QuestExpired{type=Marriage}`: emit `MarriageRejected{proposer, refused_by}` → proposer.emotion.grief += 0.15, proposer reward -1.0.
5. Cap: one active Marriage PostQuest per NPC (avoid spamming proposals). Mask predicate `¬any_active_marriage_quest(self)`.

**Gaps / open questions:**

- **Who proposes?** Either side — mask is symmetric (`can_marry` ∧ `¬has_spouse` ∧ `relationship.trust > MIN`). Training will produce both. OK.
- **Multi-spouse cultures.** Current `spouse_id: Option<u32>` is single-slot. A polygamous culture needs `spouses: Vec<u32>`. Out of scope for v1; make single-spouse assumption.
- **Arranged marriage (by parent).** A's parent posts on behalf of A. That's `PostQuest{Marriage, party_scope=Individual(parent_of_A), target=Agent(B)}` with `terms=ArrangedFor(A)`. Cascade wires A into the resulting Family. Not hard; just another cascade rule on `terms`.
- **User's hint about story 63 (invite).** If "Invite" becomes a first-class quest type distinct from PostQuest (e.g. `Invite{target=B, kind=Marriage}`), the same machinery applies — mask + cascade. I treat this as a naming nit: Marriage-as-PostQuest already reads as an invite to B.
- **Minimum-viable extension:** `marriage_eligibility` view, `FoundGroup{Family}` cascade, `compatibility_score` feature on `known_actors` slot, `MarriageRejected` reward hook. Small; 2–3 days.

**Related stories:** 52 (children inherit parental traits, requires marriage to pair parents), 49 (marriage is NOT a betrayal-immune bond; high-ambition low-loyalty profile can LeaveGroup(Family) via `AnnulMarriage`).

---

### Story 48: Monsters defend dens, hunt prey
**Verdict:** SUPPORTED (with small observation-feature additions)

**User's framing note:** "Sure."

**End-to-end trace (hunting case):**

1. **Observation (wolf alpha, Agent W):**
   - `self.creature_type = Wolf` → one_hot(~8) in self.atomic (§2.1.1, `creature_type_one_hot`). Derivation: wolf has `can_speak=false`, `has_inventory=false`, `can_build=false`, `can_trade=false`; `is_predator=true`.
   - `self.needs.hunger` high
   - `self.personality.risk` (reused for prey-boldness)
   - `memberships[k]` includes `Group{kind=Pack, leader_id=self or pack_leader_id, eligibility_predicate=creature_type==Wolf}`
   - `nearby_actors[K=12]` sorted by distance — each slot has `creature_type_one_hot(~8)`, `hp_pct`. `is_hostile(self, other)` is the derivation:
     ```
     is_hostile(self, other) = relationship_valence < HOSTILE
                             ∨ groups_at_war(self, other)
                             ∨ predator_prey(self.creature_type, other.creature_type)
     ```
     (Proposal §2.1.4 footnote.) A deer in `nearby_actors` is hostile-to-hunt to a wolf via the `predator_prey(Wolf, Deer)` branch.
   - `home_den: Option<(f32, f32)>` (`state_npc.md:255`) — surfaces as `distance_to_den` feature in self contextual.

2. **Policy:** emits
   ```
   micro_kind=Attack
   target=nearby_actors[deer_slot_idx]
   ```
   Micro primitive, not a quest. `Attack(t)` passes mask because `is_hostile(W, t)` via predator_prey.

3. **Cascade:** standard combat. `Kill` event → wolf's inventory not used (no `can_trade`), but `Eat` action consumes the corpse next tick (`Eat` mask: `corpse_nearby ∧ self.needs.hunger > EAT_THRESH`).

4. **Pack coordination:** alpha wolf's kill event fires `BehaviorObserved{witness=pack_member, subject=alpha, kind=HuntSuccess}` for every pack member in `nearby_actors`. This populates each member's memory ring, subtly increasing their hunting action-outcomes EMA. Over time the pack learns to hunt together. No special pack system.

**Den defense case:**

1. **Trigger:** a Human enters `nearby_actors[k]` of wolf W, with `is_hostile(W, Human) = predator_prey(Wolf, Human) = false` (wolves aren't prey-predators on humans unless hungry — predicate returns false by default), but the intruder's `distance_to_den < DEN_RADIUS`.
2. **Observation:** new slot feature `near_my_den[K=12]` (boolean on each nearby_actors slot). Alternatively, a derived `is_trespasser(self, other)` view:
   ```
   is_trespasser(self, other) = self.home_den.is_some()
                              ∧ distance(other.pos, self.home_den) < DEN_RADIUS
                              ∧ other ∉ self.memberships[Pack].members
   ```
   Surfaces as a feature on `nearby_actors[k]`.
3. **Mask:** `Attack(t)` is already gated by hostility. Extend `is_hostile` to include `is_trespasser(self, t) ∧ t.creature_type ∈ threatening_types`. Or add a new mask path: `DefendDen(t) when is_trespasser(self, t) ∧ self.needs.safety < SAFE_THRESH` that compiles to the same Attack action.
4. **Policy:** with `need.safety` dipping (aggressor near den → fear increase), action_eval picks Attack. Pack members nearby also see the intruder; `memberships[Pack]` + proximity triggers them to join via normal Attack actions (no special defense cascade).

**How the DSL supports this:**

- `creature_type` tag differentiates the wolf from a human using the same struct (`state_npc.md:11`).
- `home_den: Option<(f32, f32)>` already in per-agent data (`state_npc.md:255`), listed for `Territorial/PackPredator`.
- `predator_prey(ct1, ct2)` — derived view keyed on the two types. Data-driven table in `Registry`.
- `Group{kind=Pack, eligibility_predicate=creature_type==Wolf}` — same group primitive (`state_aggregate.md:639`).
- Pack members see each other's kills via `recent_memory_events[K=12]` — `BehaviorObserved` or `HuntSuccess` event.

**Implementation walkthrough:**

1. Populate the `predator_prey` table in Registry: `{(Wolf, Deer), (Wolf, Rabbit), (Dragon, Cow), ...}`. Symmetric-false; explicit pairs.
2. Spawn wolves with `home_den = Some(pos)`, `memberships=[Pack{leader=alpha}]`.
3. Add `is_trespasser(self, other)` as a derived view; surface on `nearby_actors[k]` as a boolean feature.
4. Verify mask for `Attack` uses `is_hostile(self, t)` that includes both predator_prey and trespasser branches.
5. Confirm the `Eat` action has a `Corpse` target (an Item with `durability=low` and `food_value`) accessible via `nearby_resources` — derived view over dropped-item events.

**Gaps / open questions:**

- **`is_trespasser` feature** — needs adding to the `nearby_actors[k]` slot spec. Small: +1 boolean per slot. Append-only.
- **Pack morale / loss of alpha cascade.** When alpha dies, pack succession runs via story 46's Claim machinery — need to confirm Claim quest mask permits Pack. Today's mask gates Claim on `governance ∈ {Hereditary, Elective, ...}`. Packs should use `governance = AlphaContest` (fight-winner succeeds) — another governance enum variant.
- **Monsters can't speak or use quests.** `creature_type.can_speak = false` → mask rejects `PostQuest` / `AcceptQuest` / `Bid` / `Converse` / `ShareStory`. Monsters are constrained to micro primitives. Good; prevents nonsensical "wolf posts a trade auction."
- **Prey awareness.** Deer sees wolf in `nearby_actors`; `is_hostile(Deer, Wolf) = predator_prey(Deer, Wolf) = false` but `predator_prey(Wolf, Deer) = true` creates asymmetry. Deer's mask for Flee: `threats[k]` slot sorted by `time_to_impact`. Add predator_prey-reversed threats slot population: any nearby actor for which `predator_prey(other, self) = true` appears in threats. This is straightforward.
- **Minimum-viable extension:** Registry `predator_prey` table, `is_trespasser` slot feature, `AlphaContest` governance variant, reversed-predator_prey threats inclusion. ~1 week.

**Related stories:** 46 (alpha succession by fight — cleaner Claim variant), 49 (defection — a lone wolf LeaveGroup(Pack) joins another pack; multi-membership not relevant for packs but action verbs are).

---

### Story 49: Some NPCs betray for personal gain
**Verdict:** SUPPORTED

**User's framing note:** "ESSENTIAL."

**End-to-end trace:**

1. **Observation (agent N, member of factions G_home and considering G_other):**
   - `self.personality.ambition` high
   - `self.loyalty` (legacy float, `state_npc.md:315`; maps into observation via `self contextual` block)
   - `memberships[K=8]`: slot for G_home with `my_standing_in_group`, `my_tenure_log`, `group_activity_log`. Low standing + long tenure + low activity = "I'm undervalued here."
   - `known_groups[K=6]`: slot for G_other with `standing_with_me_one_hot`, `group_strength_log`, `my_membership ∈ {none, exiled, member, leader}`.
   - `known_actors[K=10]`: G_other's leader or officer — if my `relationship(me, them).trust > 0`, there's a welcome channel.
   - `recent_memory_events`: grievances against G_home leadership (unpaid wages, missed promotion, betrayal by ally).

2. **Policy path (step 1 — Leave):** emits
   ```
   LeaveGroup{group_id=G_home}
   ```

3. **Mask for LeaveGroup:**
   ```
   LeaveGroup(g) when g ∈ self.memberships
                    ∧ ¬self.binding_oaths_with(g)
                    ∧ ¬self.role_in(g) ∈ {Founder ∧ only_founder_alive(g)}
   ```
   The `binding_oaths_with` predicate reads `active_oaths(npc)` view from the memory (`systems_social.md:254`). A sworn oath blocks defection unless `BreakOath` is emitted first (explicit betrayal).

4. **Cascade (LeaveGroup):**
   - `NPCLeftGroup{npc=N, group=G_home, reason=...}` event.
   - `G_home.members` set recomputed (or if cached, N removed).
   - N's memberships mutated to drop G_home.
   - **Reputation cascade:** every existing member of G_home with `relationship(member, N).familiarity > 0` gets a memory event `WasBetrayedBy{betrayer=N}` (if they had active loyalty stake) or `AllyLeft{ex_ally=N}` (neutral). Their relationship.trust toward N drops by 0.3 (WasBetrayedBy) or 0.05 (AllyLeft). These are event-cascade reward hooks, not background scans.
   - Chronicle entry `NPCDefected{npc, old_group, new_group=None_yet}` if leaving with intent to join.

5. **Policy path (step 2 — Join):** same tick or later tick, N emits
   ```
   JoinGroup{group_id=G_other, role=Member}
   ```
   or equivalently `AcceptQuest{type=Service, poster=G_other_leader}` if G_other posted a recruitment bounty.

6. **Mask for JoinGroup:**
   ```
   JoinGroup(g) when g.recruitment_open
                   ∧ self.eligible_for(g)      // eligibility_predicate
                   ∧ g.kind ≠ Family           // you can't just "join" a family
                   ∧ ¬g ∈ self.memberships     // not already in
   ```

7. **Cascade (JoinGroup):** `G_other.members` grows, N's memberships gain G_other. Relationship trust with G_other's existing members starts at baseline + small welcome bump.

8. **Multi-membership conflict (the load-bearing case):**
   If `self.binding_oaths_with(G_home)` rejects LeaveGroup, agent can still emit `JoinGroup(G_other)` WITHOUT leaving G_home. Then self belongs to both. The observation slots now carry G_home and G_other in `memberships[K=8]` with possibly-conflicting standings (`G_home.standings[G_other] = AtWar`). Derived feature `n_conflicting_group_pairs(self) = count of (g1, g2) ∈ memberships² where g1.standings[g2] = AtWar`.
   - When G_home ∧ G_other are AtWar, mask predicates for `Attack(t)` read group-war standings and fire ambiguously: "attack member-of-G_other is `is_hostile(G_home view)` but `¬is_hostile(G_other view)`." The reward hook resolves it: attacking a friend is `-1.0` (via `target ∈ self.close_friends`) and attacking an enemy is `+1.0` — agent learns to not attack own-side members. A sufficiently dramatic conflict triggers a forced choice via cascade: after N ticks of being in warring groups, either group's standings-cascade auto-emits `NPCLeftGroup` for one side based on... (OPEN — see gaps).

**Emergent outcomes player observes:**
- A high-ambition merchant in the merchant guild watches gold flow away because guild leader is corrupt. Eventually LeaveGroup, takes trade secrets, JoinGroup(rival guild). Chronicle narrates. Former guildmates' relationship trust drops — they harbor grudges → revenge memory events stored → later betrayers face assassination bounties.
- A soldier on the losing side of a war defects to the winning faction. `JoinGroup(winning)` because their memberships slot shows high-war-risk + low-standing.
- A monk who swore oaths cannot defect cleanly — must `BreakOath` first (public betrayal) with a much harsher reputation cascade.

**How the DSL supports this:**

- `LeaveGroup` + `JoinGroup` universal actions (`proposal_policy_schema.md` §2.3).
- `Group.standings[other]` replaces `faction.at_war_with` — multi-group standings queryable (`state_aggregate.md:604`).
- Per-pair `relationship` records carry the reputation cascade (trust drop for ex-allies) (`state_npc.md:172`).
- `memberships[K=8]` observation slot supports multi-membership with conflict-observable features (`n_shared_groups`, `n_opposed_groups`).
- `active_oaths(npc)` view gates betrayals (`systems_social.md:255`).

**Implementation walkthrough:**

1. Cascade rules on `LeaveGroup`:
   - Remove from Group.members.
   - For each pre-existing member with familiarity > 0, emit `WasBetrayedBy{betrayer=self}` memory event weighted by prior alliance.
   - Chronicle entry.
2. Cascade rule on `JoinGroup`:
   - Add to Group.members.
   - For each existing member, emit `AllyJoined` memory event (mild positive valence).
3. `binding_oaths_with(g)` view — reads active_oaths from memory.beliefs where `oath.kind ∈ {Loyalty, Service}` and `oath.target = g`.
4. `n_conflicting_group_pairs(self)` feature on self contextual block.
5. Reward: `-0.5 on LeaveGroup(g) where self.role_in(g) = Leader` (leaders carry a higher cost for leaving) — incentivises Founder-attrition story arcs.
6. `BreakOath(oath_idx)` as a distinct action (`systems_social.md:254`). Mask permits only when the oath exists. Triggers `OathBroken` chronicle + reputation drop across the target Group.

**Gaps / open questions:**

- **Multi-membership conflict resolution.** If I'm in G_home and G_other and they go to war, what forces me to pick? Options:
  1. **Passive — agent's own reward gradient.** Attacking own-group-members is -1.0; avoiding creates tension; eventually agent LeaveGroup one side. Emergent but slow.
  2. **Cascade — automatic expulsion.** When `G_home.standings[G_other] = AtWar` is declared and `self ∈ G_home.members ∩ G_other.members`, emit `GroupExpelled{group, agent, reason=CompetingLoyalty}` on one side (which? lowest standing? lowest tenure?).
  3. **Mask pressure.** Add a mask predicate that forces a "must emit LeaveGroup(one_of_them)" action within N ticks when conflict detected. Violates single-tick-action semantics.
  Cleanest: (1) with the optional cascade for extreme cases — `tick - conflict_detected > 500 ∧ agent_took_no_side` → cascade picks lowest `standing × tenure` group and emits `GroupExpelled`.
- **Atomicity of "defect."** LeaveGroup + JoinGroup are two actions. Between them the agent is factionless. Is that OK gameplay? Probably yes — feels realistic. A failed defection (got kicked out of home, never accepted by rival) produces the interesting "wandering exile" state.
- **`JoinParty` variant.** Current doc mentions `JoinParty` for party/quest contexts (`proposal_universal_mechanics.md:64`). Unify under `JoinGroup` — a party is just `Group{kind=Party}`.
- **Minimum-viable extension:** `LeaveGroup`/`JoinGroup` cascade handlers, `WasBetrayedBy` memory-event emission on leave, `binding_oaths_with` view, `n_conflicting_group_pairs` feature. ~1 week.

**Related stories:** 45 (war pressure motivates defection), 46 (losing claimant defects), 50 (low reputation increases defection attractiveness), 54 (diplomatic auction may include "vassal commitments" that function as enforced membership).

---

### Story 50: Renowned characters emerge from reputation
**Verdict:** SUPPORTED (after the user's reframe — drop `is_legendary` boolean; reputation is a continuous signal that abilities read)

**User's framing note:** "Legendariness is fluffy. There should be a reputation system, and abilities should be capable of being impacted by reputation, so that way a legendary hero can emerge organically without making a new system for it."

**End-to-end trace:**

1. **Reputation is a continuous derived view, not a stored field.**
   ```
   reputation(npc) = Σ_witnessed_events_in_settlement weighted_by (impact × recency_decay)
   ```
   (Per `systems_social.md:532` — already in the derivable-view list.) Features the view pulls from:
   - `chronicle_mentions(npc)` — count of chronicle entries mentioning npc
   - `witnessed_positive_events(npc)` — sum over all `memory.events` of all other npcs where npc is a subject with positive `emotional_impact`
   - `witnessed_negative_events(npc)`
   - Both decayed by `recency_decay(tick - event.tick)`.

   Output: a scalar per (npc, observer) pair, OR a global scalar per npc. Both are useful; the global view is cheap (one float) and drives world-level phenomena; the per-pair view is richer (the same deed is famous in hometown, unknown abroad).

2. **Reputation in observation:**
   - `self.fame_log` already in self contextual (§2.1.2) — this becomes the primary reputation atom. Rename to `self.reputation_log` to clarify.
   - `self.reputation_summary` already listed as a feature.
   - `known_actors[K=10]` slots: add `other_reputation_known` float per slot. Derived.
   - `nearby_actors[K=12]` slots: add `other_reputation_visible` if nearby actor is famous enough. Derived.

3. **Reputation drives action emission (policy):**
   - High self-reputation + high ambition → higher probability of emitting ambitious PostQuests (Conquest, Charter, FulfillProphecy). Policy learns this from reward correlation (`+Δ(fame_log)` on important deeds).
   - Observing a high-reputation rival raises anxiety / fear (observation features: `known_actors[k].other_reputation_visible` → self.anxiety via emotion kernel).

4. **Reputation drives mask (eligibility):**
   - Charter petitions: `PostQuest{type=Charter, target=Settlement(s)}` mask gates on `self.reputation_log > CHARTER_FAME_THRESH ∨ self ∈ settlement.charter_eligible_by_peerage`. High-reputation commoners can petition; low-reputation nobles can by membership.
   - `Claim` quests in high-reputation groups: `self.reputation_log > MIN_CLAIM_REP` to prevent commoners from claiming a kingdom.
   - Auction visibility: some Diplomatic auctions only open to `visibility ⊇ reputation > THRESH`.

5. **Reputation drives ability effects (user's explicit reframe):**
   Abilities in `.ability` DSL files already carry effect + tag + scaling definitions. Extend them to read reputation as a scaling input:
   ```
   ability Intimidate {
     effect Fear {
       target: Agent
       magnitude: base + 0.1 × self.reputation_log
       duration: 3 ticks
     }
     conditions: relationship(self, t).trust < 0  ∨  t.creature_type.can_speak
   }
   ```
   Combat damage scaling:
   ```
   ability Attack {
     effect Damage {
       target: Agent
       magnitude: self.attack_damage × (1 + 0.05 × self.reputation_log)
     }
   }
   ```
   Persuasion checks: `Bid` payment in a Diplomatic auction can include `reputation_bid` as a Payment variant. Reputation spent decreases; reputation earned from successful Bid increases.

6. **Cascade emits reputation-affecting events:**
   - Every meaningful event already has `chronicle_category` + `entity_ids`. The reputation view sums over chronicle mentions by default.
   - `BehaviorThresholdCrossed{npc, tag, tier}` event (`systems_social.md:460`) fires when an NPC crosses cumulative behavior thresholds (e.g. "killed 10 monsters" or "saved 3 settlements"). This events reads like a reputation spike emission; cheap to derive from behavior_profile.
   - `LegendAscended` is NO LONGER EMITTED AS A SPECIAL EVENT. Instead, "legendary" is a predicate: `reputation_log(npc) > LEGENDARY_THRESH`. At UI render time we can label npcs that cross the threshold as "the Legendary," but that's cosmetic — the underlying data is the continuous reputation.

**Emergent outcomes player observes:**
- An unknown commoner grinding small deeds slowly raises reputation. Eventually their combat damage is measurably higher, they can petition charters, and nearby NPCs react with respect/fear.
- A "legendary hero" is just the top-K reputation NPC(s) at any tick; no special flag. Death of a legend is `EntityDied` of a high-reputation npc → cascade increases grief in everyone who witnessed their deeds, via memory.events lookup.
- Infamous outlaws have negative reputation (NO separate infamy dimension — just signed). Their Intimidate scales from negative-rep = fear; their charter petitions are mask-rejected.

**How the DSL supports this:**

- Abilities are data-driven in `.ability` files — effects already have scaling expressions. Adding `self.reputation_log` to the scaling expression grammar is purely additive (proposal_policy_schema.md §3 "DSL surface").
- `reputation(npc) = Σ chronicle_mentions × weighted` is a derived view, no new state.
- `fame_log` already in observation (§2.1.2).
- Continuous scalar means the decision policy can learn graduated eligibility, not cliff-edge behavior.

**Implementation walkthrough:**

1. Implement `reputation(npc) -> f32` as a derivable-view over chronicle + memory.events. Memoize per-tick.
2. Rename `fame_log` to `reputation_log` in observation (or keep the alias; both point to the same view).
3. Add `self.reputation_log` and `other.reputation_visible` as accessible scaling inputs in ability DSL grammar.
4. Port mask predicates that previously read `is_legendary(npc)` (per `systems_politics.md:149`) to continuous thresholds: `reputation_log(npc) > LEGENDARY_THRESH`.
5. Chronicle a "Legend ascended" entry automatically when `reputation_log(npc)` crosses the threshold — cosmetic, for UI. Do not gate gameplay on this entry.
6. Extend ability DSL examples: Attack scaled by reputation; Persuasion check scaled; intimidation fear duration scaled. Hero templates opt-in.
7. Reward hook: `+0.02 × Δ(reputation_log)` per tick. Already present via `Δ(self.fame_log)` — just rename.

**Gaps / open questions:**

- **Per-pair vs global reputation.** Global scalar is cheap and good enough for most. Per-pair is more realistic (only locals know the hero's name) but costs more — O(npcs × known_actors). Compromise: global scalar plus a `reputation_known_in_settlement[home]` bonus, so locals over-represent the hero. Leave per-pair for future iteration.
- **Reputation decay.** Old deeds fade. `recency_decay(Δt) = exp(-Δt / HALF_LIFE_TICKS)`. Parameter to tune.
- **Negative vs positive reputation.** Single signed scalar? Or two dimensions (fame + infamy)? User's reframe says single continuous. I lean single signed; an outlaw has reputation=-5, a hero has +5. Intimidate ability uses |reputation|.
- **Fairness across settlement sizes.** Small-settlement hero can hit high reputation cheaply (few witnesses, dense chronicle). Normalize by `chronicle_density(settlement)`? Only if it produces visibly broken gameplay.
- **Minimum-viable extension:** `reputation(npc)` view, observation feature rename, ability DSL scaling grammar extension, chronicle cross-linking for automatic legend-threshold narration. ~1 week.

**Related stories:** 45 (high-rep leader's Conquest more likely to succeed through morale), 46 (reputation gates Claim eligibility), 53 (witnessing high-rep heroes' deaths triggers heavier grief), 54 (reputation as bid currency in diplomatic auctions).

---

### Story 51: Long wars affect economy
**Verdict:** SUPPORTED (and this story is the strongest argument for the emergent-cascade design)

**User's framing note:** "ESSENTIAL."

**End-to-end trace — the cascade chain:**

T=0: Leader posts `PostQuest{type=Conquest, party=Group(G_self), target=Group(G_rival)}`. Cascade sets `G_self.standings[G_rival] = AtWar`, vice versa.

T=1 … T=2000: the following cascades fire as emergent consequences. Name each event and the derived view it feeds.

**(A) Settlement raid cascade:**
1. G_self member emits `micro_kind=Attack` against G_rival member near G_rival's settlement S_rival. Standard combat events: `EntityAttacked`, `EntityDied`.
2. `G_self` leader periodically emits `PostQuest{type=Raid, target=Settlement(S_rival)}`. Party members who accept move to S_rival and attack defenders. `BattleStarted{grid_id=S_rival.grid_id}` / `BattleEnded{grid_id, victor}`.
3. On raid success: cascade rule fires `TreasuryLooted{settlement=S_rival, amount=T_loot, looter_group=G_self}` — moves gold from `S_rival.treasury` to `G_self.treasury`. A dedicated event, not a rule hidden in a random system.
4. Cascade also emits `BuildingDamaged{building_id, damage_amount}` for affected structures.

**(B) Settlement treasury depletion (view):**
```
S_rival.treasury_trend = delta_per_tick(S_rival.treasury, window=200)
```
Not a stored field. A derived view that `S_rival.leader_id` observes as a context feature. When `treasury_trend < -CRISIS_THRESH`, leader policy sees the crisis. No special "depletion event" fires; it's just the observed state.

**(C) Trade route disruption:**
1. Caravan agents (merchants) have `economic_intent = Trade` and their normal policy is to `MoveToward(trade_destination)` + `AcceptQuest{type=Trade}`.
2. Their mask for `Travel(through_rival_territory)` is:
   ```
   Travel(path) when safe(path)  // derived view
   safe(path) = ∀ cell ∈ path: cell.owning_group.standings[self.memberships.max_standing_group] ≠ AtWar
   ```
   Once `G_self.standings[G_rival] = AtWar`, routes through G_rival territory are no longer safe. Caravans' policy reroutes (longer paths → less profit → fewer trips) or refuses.
3. No caravan → `TradeRoute.strength` decays (`state_aggregate.md:213`, "decays without activity, abandoned < 0.1"). Purely time-based decay of the emergent route quality.
4. Chronicle: "The silk road between X and Y has fallen silent."

**(D) Commodity price shift (view over `state_aggregate.md:22`):**
```
S_rival.prices[c] = base_price[c] / (1 + S_rival.stockpile[c] / (population × halflife))
```
With fewer caravans delivering to S_rival and consumption ongoing, `stockpile[c]` drops. Prices rise. Local NPCs' `price_beliefs` update via trade observation → they emit more `Bid` actions at high prices, migrate to cheaper settlements, or starve.

**(E) Population loss:**
1. Every `EntityDied` event with `entity.home_settlement_id = S_rival` decrements the population view. `population(S) = count(alive entities where home_settlement_id == S)` — derived.
2. Conscription: `G_rival` leader emits `PostQuest{type=Defend, party=Group(G_rival)}` pulling more members into the war → more casualties → more deaths → compounding.
3. Starvation from (D): `NeedStarvation{npc}` events fire when `needs.hunger < STARVE_THRESH` for N ticks → `EntityDied{cause=Starvation}`.
4. Net: population(S_rival) drops over war duration.

**(F) Work output decline (view):**
- `production_rate(S) = Σ agents in S with work_state=Work, weighted by skill`. As population drops, production drops. No special production system needed.
- Buildings with `worker_ids.len() < work_capacity × 0.3` stop producing at full rate (already modeled via work_capacity).

**(G) Morale collapse:**
- Every `EntityDied{entity.settlement = S_rival}` triggers `BondGrief{witnessing_npcs}` via memory ring. Each witness's emotion.grief spikes by 0.6, decays slowly (-0.01/tick = >60 ticks to decay to 0).
- Aggregate effect on settlement: `morale_summary(S) = avg(morale over agents with home_settlement_id = S)` drops. Observable by S_rival leader.

**(H) Rebellion / defection cascade (story 49 referential):**
- Low-morale, low-treasury, war-weary members with high ambition emit `LeaveGroup(G_rival)` or `JoinGroup(G_self)`. Defection rate scales with `war_exhaustion(G_rival)` derived view — see below.

**(I) war_exhaustion view:**
```
war_exhaustion(G) = Σ_over_current_war (
    casualty_events × w1 +
    treasury_deltas × w2 +
    duration × w3
)
```
(Per `systems_politics.md:59`.) Surfaces on leader's self.contextual block. When high, leader policy emits `PostQuest{type=Peace, target=G_self, reward=Reparations}` or `PostQuest{type=Submit}` (vassalization). Cascade resolves the war as a Marriage-like MutualAgreement auction.

**Total player-visible chain:**

Leader posts war → raids loot treasury → treasury drops (view) → trade routes fail safety-mask → caravans stop → commodities scarcen → prices spike → starvation + low morale → population drops → production drops → defection spikes → new war weariness → leader sues for peace or loses succession.

**What is NOT in this design:** a system called "war_effects_on_economy" that scans factions and applies drops. Every link in the chain is either (a) an existing action (Raid, LeaveGroup, Travel) with a pre-existing mask predicate, (b) a cascade rule on a first-class event (TreasuryLooted, EntityDied), or (c) a derived view (treasury_trend, safe_path, population).

**How the DSL supports this:**

- `standings[G]: Map<GroupId, Standing>` drives the `safe(path)` mask predicate that disrupts trade (`state_aggregate.md:604`).
- `TradeRoute.strength` is already a time-decayed derived view (`state_aggregate.md:213`).
- `population` is a derived view: `count(alive entities where home_settlement_id == S)` (`state_aggregate.md:558`).
- Price formula is already derived from stockpile and population (`state_aggregate.md:446`).
- `war_exhaustion(G)` view is already spec'd (`systems_politics.md:59`).
- Cascade rules on `EntityDied`, `TreasuryLooted`, `BuildingDamaged` compose.

**Implementation walkthrough:**

1. Confirm cascade rules emit: `QuestPosted{Conquest}` → update standings; `BattleEnded{victor}` → emit `TreasuryLooted` when near settlement, `BuildingDamaged` when applicable.
2. Add `safe(path)` derived view; `Travel` mask gates on it. Caravan agent policy reroutes.
3. Confirm `TradeRoute.strength` decays per tick without activity (already spec'd).
4. Confirm population view recomputes O(n_entities) and is cached.
5. Add `war_exhaustion(G)` view accumulating casualty + treasury + duration contributions. Surface as a context feature on leader observation.
6. Add `PostQuest{Peace}` mask: `any(standings[g] = AtWar for g) ∧ war_exhaustion(g) > EXHAUSTION_THRESH`.
7. Verify `price_beliefs` update from trade observation → drives Bid policy without special code.

**Gaps / open questions:**

- **Granularity of raid damage.** `TreasuryLooted.amount` is what? `min(S.treasury × loot_fraction, carry_capacity_of_raiders)`. Parameter tuning; not a design gap.
- **Who raids?** A `party_scope=Group(G_self)` Conquest quest implies all G_self members can raid, but who decides when? Either leader emits sub-PostQuests `Raid(S_rival)` during the Conquest quest window, or members autonomously `MoveToward(S_rival) + Attack`. Both plausible. Cleanest: sub-quests for structured raids; free action for opportunistic raiders.
- **Neutral settlements affected by neighboring war.** A settlement unaligned with either group may still see caravans re-route around contested roads. This falls out automatically — the `safe(path)` predicate applies wherever the path crosses contested territory, regardless of the caravan's origin.
- **War ending during peace quest.** Peace quest's MutualAgreement auction needs both leaders' acceptance. What if one leader dies mid-war? Succession runs (story 46), new leader inherits ongoing wars via group standings. They inherit — they don't auto-inherit the peace offer. Good behavior.
- **Minimum-viable extension:** `safe(path)` view + Travel mask, `TreasuryLooted` event + cascade, `war_exhaustion` view + Peace mask. The cascade chain is largely already spec'd across system docs; wire up the missing views. ~2 weeks.

**Related stories:** 45 (triggering war), 46 (leader death mid-war + succession), 49 (defection from losing side), 54 (treaties through diplomatic auctions).

---

### Story 52: Children inherit parental traits
**Verdict:** SUPPORTED

**User's framing note:** "ESSENTIAL."

**End-to-end trace:**

1. **Trigger.** Two married agents A and B in a Family group co-located and personality.social_drive high → A emits:
   ```
   PostQuest{
     type=HaveChild,                 // new QuestType variant
     party_scope=Group(Family_id),
     target=Agent(self),             // symbolic — "our partnership is the target"
     reward=Birth,
     terms=RequireAcceptance(B)
   }
   ```
   (Or simpler: a non-quest cascade — when both spouses emit `Cohabit` and `needs.social_drive` met for both, spontaneous conception event fires probabilistically. The quest form gives more agency; the spontaneous form is simpler. Either works. I'll use spontaneous for MVP.)

2. **Conception event.** `ChildConceived{parents=[A, B], tick}` fires. After gestation delay (say 500 ticks), `ChildBorn{parents=[A, B], child_id=new_id, settlement=Family.home}` fires.

3. **Spawn handler (ChildBorn cascade — the essential bookkeeping):**
   - New AgentId allocated via `next_id()`. (`systems_politics.md:197` identifies this as essential bookkeeping.)
   - New agent spawned with:
     - `creature_type = A.creature_type` (inherit species)
     - `alive = true, level = 0, hp = initial_hp, pos = Family.home_pos`
     - `personality[i] = blend(A.personality[i], B.personality[i], with_variance)` — proposal:
       ```
       child.personality[i] = lerp(A.personality[i], B.personality[i], 0.5)
                            + gaussian_noise(mean=0, std=0.1)
                            clamped to [0, 1]
       ```
     - `needs[i] = default_newborn_needs[i]`
     - `memberships = [Membership{group=Family, role=Member}]`  (inherits family group)
     - `home_settlement_id = A.home_settlement_id`
     - `parents = [A.id, B.id]`, `children = []`
     - `class_tags = []`, but with a `class_bias_weights` derived from parents' behavior_profile tops — so the child starts with "kitchen smell" of mother's cooking bias, "hammer smell" of father's smithing bias. Not class levels, just biases that speed up learning.
     - `cultural_bias = avg(A.cultural_bias, B.cultural_bias) + noise`
   - A.children.push(child); B.children.push(child). A.emotions.joy += 0.4, B.emotions.joy += 0.4.
   - Settlement.population grows by 1 (derived view, recomputed next tick).

4. **Observation of newborn child:**
   - Standard observation — same schema as adults (they use the same observation packer).
   - Newborn features:
     - `self.level = 0, self.hp_pct = 1.0, self.max_hp_log = small`
     - Personality tensor has parents' blend — already in spawn.
     - `memberships[k]` includes Family.
     - `known_actors[k]` empty initially — no established relationships. A and B appear in `nearby_actors` once proximity established (same home), and relationship_kind_one_hot = Kin via the Family membership overlap.
   - A minimum-age mask ensures the child doesn't emit adult actions until sufficient level. Actions like `PostQuest{Marriage}` mask on `self.level > ADULT_LEVEL`; `Attack` mask on `self.level > COMBAT_LEVEL`.

5. **Emergent outcomes player observes:**
   - Child of a compassionate healer + compassionate priest → inherits high compassion → natural healer.
   - Child of high-ambition leaders → inherits ambition → grows into Claim-quest candidate.
   - Child's initial trust toward parents = HIGH (via memory events of being raised, OR just seeded high: `relationship(child, parent).trust = 0.7`).
   - Family lineages develop characteristic personality clusters over generations.

**How the DSL supports this:**

- Agent fields `parents: Vec<u32>`, `children: Vec<u32>` already exist (`state_npc.md:283–284`).
- `Group{kind=Family}` is the lineage container (`state_aggregate.md:638`).
- Personality is a 5-dim float — trivial to blend (`state_npc.md:102`).
- Cascade rule on `ChildBorn` emits spawn with inherited fields — event-sourced, declarative.
- Observation is uniform — children use the same packer as adults.

**Implementation walkthrough:**

1. Define `ChildConceived`, `ChildBorn` events.
2. Spawn handler emits new agent with:
   - `personality = 0.5 × (A.personality + B.personality) + gaussian(0, 0.1), clamped`
   - `class_bias_weights = merge_top_k(A.behavior_profile, B.behavior_profile, k=3)` — seeds learning biases
   - `cultural_bias = avg(A, B) + noise`
3. Add `Family.members.push(child); A.children.push(child); B.children.push(child)`.
4. Add mask predicates for adult actions gated on `self.level > N`: `PostQuest{Marriage}`, `PostQuest{Conquest}`, `Bid`, `SwearOath`. Children are observers + learners only for early ticks.
5. Add `child_compatibility(child, parent) = 0.7 + 0.1 × personality_similarity` as relationship trust bootstrap. Seeded at spawn.

**Gaps / open questions:**

- **Gestation duration vs tick scale.** 500 ticks = 500 × 100ms = 50s of real-time. Reasonable for a long game. Parameter.
- **Child aging / leveling.** Children need to "grow up" — levels increase via some XP gain. Either automatic (`level = min(age_in_ticks / ADOLESCENCE_TICKS, MAX_CHILD_LEVEL)` — derived view, no XP) or active (child works simple tasks, earns XP). Either works; automatic aging is simpler and doesn't require special teaching code. Adult status is a predicate: `level > ADULT_LEVEL`.
- **Orphaned children.** If A and B die before child.level = ADULT, does the child survive? Cascade on `EntityDied{entity=parent}` should check for dependent children; if parent dies with children, either the Family group continues under `surviving_parent` or the child is informally adopted by another Family member via `JoinGroup` (cascade-triggered, not agentic — the child isn't agentic yet).
- **More than two parents / adoption / same-sex unions.** Design should accept any (parents: Vec<AgentId>) tuple from Family.members. The blending formula handles k>=1 by averaging. Culture determines marriage eligibility; the physics is uniform.
- **Creature_type inheritance for cross-species.** `predator_prey(A, B)` probably blocks most cross-species marriages via mask (marriage_eligibility). Stick to same-species for v1.
- **Teaching / apprenticeship bias.** Extra bias if child `JoinGroup(Apprentice)` under a parent — mentor_lineage already exists (`state_npc.md:285`). The child joins as `role=Apprentice`; teacher's behavior_profile accelerates child's class xp gain. Works.
- **Minimum-viable extension:** `ChildBorn` spawn handler with personality blending + cultural_bias averaging + class_bias seeding, age-gated masks, `child_compatibility` trust bootstrap. ~3 days.

**Related stories:** 46 (grown child becomes Claim candidate using inherited ambition), 47 (marriage precedes this), 49 (children inherit allegiance to parent's groups as starting memberships).

---

### Story 53: Emotional reactions to witnessed events
**Verdict:** SUPPORTED

**User's framing note:** "ESSENTIAL."

**End-to-end trace:**

1. **Event emission (ground truth).** A significant physical event fires — `EntityDied{id=victim, killer=attacker, tick}`. This is always first-class.

2. **Witness detection.** For the event's settlement/location, compute the set of nearby agents:
   ```
   witnesses = { a : distance(a.pos, event.location) < WITNESS_RADIUS ∧ a.alive }
   ```
   Plus long-distance witnesses via affinity — agents with strong relationships to victim/attacker learn of the event via gossip after a delay (`heard_about_event` cascade). For the per-tick case, spatial witnesses are enough.

3. **Per-witness `Witnessed` event emission (cascade):**
   For each witness W, emit:
   ```
   Witnessed{
     observer=W,
     subject=victim,          // or attacker, or both
     event_type=FriendDied | Murder | Battle | ...
     location=event.location,
     tick,
     entity_ids=[victim, attacker],
     emotional_impact=f(relationship(W, victim), relationship(W, attacker), event_severity)
   }
   ```
   (Per `systems_social.md:450`. This is already the central emission channel.)

4. **Memory update (cascade side-effect):** The `Witnessed` event lands in `W.memory.events` (VecDeque, capped 20). Oldest discarded.

5. **Emotion derivation (view over memory):**
   ```
   emotions(W) = fold_with_decay(
     W.memory.events,
     kernel: emotion_response_kernel,
     decay: per_tick
   )
   ```
   Kernel table (from `state_npc.md:114`):
   - `FriendDied` (relationship.trust > 0.5) → grief += 0.6
   - `WasAttacked` → fear += 0.5, anger += 0.3
   - `WonFight` → joy += 0.3, pride += 0.3
   - `TradedWith (success)` → joy += 0.1
   - `MadeNewFriend` → joy += 0.2, social += 5

   Per-tick decay happens naturally: `agent_inner` applies `emotions.grief *= 0.99` (= -0.01/tick as spec'd). No separate emotion drift system needed; it's integrated into the primary agent tick.

6. **Observation (self contextual block, §2.1.2):**
   - `emotions(6)`: joy, anger, fear, grief, pride, anxiety — all derived floats from memory fold.
   - `recent_memory_events[K=12]`: raw slot array — each slot has event_type one-hot, age, target_in_nearby index, valence.
   - `memory_beliefs_summary`: counts of LocationDangerous / EntityTrustworthy / etc.
   - `recent_witness_count_5tick`, `recent_witness_count_50tick`: how many witnessed events in recent windows (modulates anxiety).

7. **Behavioral consequences (policy-driven, not hard-coded):**
   - High grief → lowered productivity (action_outcomes EMA on `Work` dips when grief > threshold → policy picks Rest/Socialize).
   - High anger → aggressive action preference (`Attack` utility up on enemies in nearby_actors).
   - High fear → Flee micro primitive preferred.
   - Emotions feed personality drift slowly: repeated grief from friend deaths lowers compassion (witness to cruelty hardens the heart).

**Emergent outcomes player observes:**
- After a battle in the town square, nearby NPCs visibly mourn — their actions shift to Rest/Socialize, productive work halts.
- A shopkeeper witnessing a theft becomes anxious, installs tiles to block storefront (Build action with defensive tiles).
- Children witnessing their parents' deaths have grief persisting longer (shallower emotion decay via personality.compassion bias) — shapes orphan personality over many ticks.

**How the DSL supports this:**

- `Witnessed { observer, subject, kind, impact, tick }` is the central memory-event emission (`systems_social.md:450`).
- `emotions(npc) = fold(recent memory.events × emotion_kernels, time_decay)` is a derived view (`systems_social.md:518`).
- Observation surfaces both the derived emotions AND the raw `recent_memory_events[K=12]` slot array (policy can learn patterns beyond the hand-designed kernel).
- Decay built into `agent_inner` per-tick.

**Implementation walkthrough:**

1. Ensure every first-class event (EntityDied, EntityAttacked, TradedWith, BuildingCompleted, QuestCompleted, ChildBorn, MarriageFormed) has an associated `Witnessed` emission cascade.
2. Witness radius derivation: default spatial (50 units). Social ties expand radius: `W.relationships[subject].familiarity > 0.5` means W hears about the event within N ticks via gossip (`heard_about_event` cascade, delayed emission).
3. Define emotion kernels per `MemEventType`. Weighted table in Registry.
4. `emotions(npc)` view: lazy fold over memory.events with per-event kernel × exp(-age × decay_rate). Memoized per tick at first observation-pack.
5. Confirm observation §2.1.2 exposes both raw and summarized features.
6. Reward hook on witnessed events: `-0.5 on Witnessed{FriendDied}`, `+0.2 on Witnessed{QuestCompleted{ally}}` — encourages learning about social reward.

**Gaps / open questions:**

- **Memory ring size (20).** May be too small for long emotional arcs. If a soldier sees 50 deaths in a battle, only 20 land in memory. The emotion kernel summary partially addresses this (grief accumulates per event then decays slowly). But a legendary-tier trauma warrants longer memory. Proposal: tier memory — last 20 in ring + last 5 "indelible" events that never evict (weighted on `emotional_impact > 0.8`).
- **Belief formation from events.** `beliefs` = `fold(events → belief deltas)`. A Witness{FriendDied at location L by EntityX} should create/strengthen a `LocationDangerous(L)` and `EntityDangerous(X)` belief. Existing semantic-layer work (`state_npc.md:157`); confirm the fold is automatic.
- **Emotion → action reward coupling.** The RL reward should include `-0.01 × self.emotions.anxiety` etc. so policy learns to mitigate negative emotions (socialize after grief, rest after fatigue). Already implicit via `need-recovery × 0.1` in reward block (§2.5), but emotions aren't in the need vector. Add `emotions.avg_valence` as a reward term.
- **Long-distance witness (gossip).** A distant ally should hear about a friend's death eventually. Cascade rule: `FriendDied{victim, tick}` + `relationship(W, victim).trust > 0.5 ∧ W ∉ witnesses` → schedule delayed `Witnessed` emission for W at `tick + gossip_latency(distance)`.
- **Minimum-viable extension:** Witness emission cascade for all major events, emotion kernel table, `emotions(npc)` view with decay, gossip-delay cascade for long-distance witnesses. ~4 days.

**Related stories:** 46 (leader death triggers massive witness cascade, reshapes successor selection), 49 (betrayal witnessed degrades trust broadly), 50 (high-rep NPC death = legendary mourning cascade).

---

### Story 54: Faction alliance via bidding
**Verdict:** PARTIAL (auction machinery needs extension for multi-party diplomatic auctions; single-seller, multiple-bidders model is close enough for v1)

**User's framing note:** "ESSENTIAL."

**End-to-end trace:**

1. **Initiation.** Leader of G_self observes threat from rival G_rival: `known_groups[G_rival].standing = AtWar` and `G_self.military_strength < G_rival.military_strength`. Leader emits:
   ```
   PostAuction{
     kind=Diplomatic,
     item=AuctionItem::AllianceTerms{
       obligations: MutualDefense(vs=G_rival),
       duration: 5000 ticks
     },
     seller=AuctionParty::Group(G_self),
     visibility=Visibility::Groups(top_K_neighboring_factions),  // broadcast
     deadline_tick=now+100,
     reserve_price=Payment::None,
     resolution=AuctionResolution::MutualAgreement   // both sides must accept
   }
   ```
   (This is `PostQuest{type=Diplomacy}` in current doc but better expressed as `PostAuction{kind=Diplomatic}` — the mechanism is an auction, not a quest. Unless we overload `PostQuest` to fire an auction internally. Both work; the important thing is the observation + policy.)

2. **Observation (bidder, leader of G_candidate):**
   - `known_groups[K=6]` slot for G_self: `standing_with_me`, `group_strength_log`, `is_at_war_with_my_enemies` (new feature).
   - `active_auctions[K=4]` (new slot, analogous to active_quests) surfaces the new Diplomatic auction. Features: kind, seller, obligations_summary, deadline, visibility.
   - `self.G_candidate.standings` map — do I have interests aligned with G_self (common enemies) or with G_rival (common markets)?
   - Military strength ratios.

3. **Policy (bidder):** emits
   ```
   Bid{
     auction_id,
     payment=Payment::Combination([
       Payment::Promise(MutualDefense(vs=G_rival), duration=5000),
       Payment::Gold(1000),                      // treasury contribution
       Payment::Service(RaiseArmy(troops=50))    // concrete troop pledge
     ]),
     conditions=BidConditions::MustInclude(MutualDefense)
   }
   ```
   Multiple G_candidates may emit bids; G_self's alliance may only accept one (single-winner MutualAgreement) OR accept multiple (multi-winner coalition).

4. **Mask:**
   ```
   Bid(auction_id, ...) when auction.visibility ⊇ self
                           ∧ self.leader_groups[auction.seller.kind]     // I lead a group of same kind
                           ∧ self.memberships[0].kind ∈ AllianceEligible
                           ∧ self.G_lead.standings[auction.seller] ≠ AtWar  // can't ally with someone I'm fighting
   ```

5. **Resolution cascade (MutualAgreement):**
   At auction deadline (or when all parties have bid and seller's "accept_bid(id)" action fires):
   - If seller emits `AcceptBid(auction_id, winning_bid_id)` → `AuctionResolved{auction_id, winner, payment}`.
   - Cascade rule: `AuctionResolved{kind=Diplomatic, winner=G_winner}` →
     - `AllianceFormed{between=[G_self, G_winner], terms=auction.item.obligations, duration=auction.item.duration}` chronicle entry.
     - `G_self.standings[G_winner] = Allied`; `G_winner.standings[G_self] = Allied`.
     - If terms include MutualDefense: whenever `G_self.standings[X] = AtWar`, G_winner gets auto-generated `PostQuest{type=Defend, party=Group(G_winner), target=G_self}` (obligation kick-in).
     - Payment transfers: Gold and Service promises encoded as scheduled future events.

6. **Emergent outcomes player observes:**
   - A weaker faction (G_self) broadcasts for help. Multiple stronger neighbors bid. Seller picks the one offering most concrete military support (policy learns to evaluate Service promises > Gold > Reputation).
   - Alliance chronicled. Flavour text narrates the terms.
   - When G_rival invades G_self, cascade auto-drags G_winner into the defense via the scheduled Defend quest. G_winner's leader may then emit AcceptQuest or try to wriggle out (BreakOath → reputation cascade).
   - Broken alliances are `BreakAlliance(target)` leader actions → `AllianceBroken` event → reputation cascade harms the breaker's Bid chances in future auctions.

**How the DSL supports this:**

- Universal `Bid` verb (`proposal_policy_schema.md` §2.2).
- `AuctionKind::Diplomatic` spec'd (`proposal_universal_mechanics.md:231`).
- `AuctionResolution::MutualAgreement` for two-sided consent (`proposal_universal_mechanics.md:240`).
- `Group.standings[G]` drives ally/war switches via cascade (`state_aggregate.md:604`).
- `Payment::Promise(Reward, u64)` for deferred obligations (`proposal_universal_mechanics.md:193`).

**Implementation walkthrough:**

1. Extend `AuctionItem` enum with `AllianceTerms{obligations, duration}`. `obligations` = enum of `MutualDefense(vs=GroupId)`, `TradeExclusive`, `VassalCommitment`, etc.
2. Extend `Payment` with `Service(ServicePromise)` for troop pledges. Schedule future `RaiseArmy` event.
3. Add mask predicates for PostAuction{Diplomatic} and Bid{Diplomatic}.
4. Add resolution cascade: `AuctionResolved{kind=Diplomatic}` → update standings + scheduled obligation triggers.
5. Add `active_auctions[K=4]` observation slot with Auction summary features (kind, seller, best_current_bid, deadline_tick).
6. Cascade rule for obligation enforcement: on `WarDeclared{G_aggressor, G_self}`, if `G_self.allies (via standings = Allied ∧ obligations include MutualDefense vs G_aggressor)`, emit `PostQuest{Defend}` on ally's behalf automatically.

**Gaps / open questions:**

- **Multi-party auction semantics.** The current `MutualAgreement` resolution is binary (seller + one bidder). Coalitions of 3+ factions require either:
  - Sequential pairwise auctions (A posts, B accepts; B then posts with A already allied, C accepts; ... → coalition), or
  - True multi-party auction with resolution when > K parties accept. Mechanics doc §"Auction state machine" doesn't cover this. Add `AuctionResolution::Coalition{min_parties=K}` variant.
- **Bid currency parity.** Can you bid Reputation in a Diplomatic auction? Services? Gold? The `Payment::Combination` allows heterogeneous bids, but seller's scoring function needs to handle apples-to-oranges. Simplification: `total_bid_value(payment, auction.kind) = Σ component_value_for_kind(c)` — a per-kind valuation function. Ad-hoc but workable.
- **Obligation enforcement.** When ally fails to honor MutualDefense (doesn't accept the auto-generated Defend quest), how does the alliance break? Needs a cascade: `AcceptQuest failed × N_retries` → `AllianceBroken{breaker=ally, obligation_violated}` chronicle entry, standings flip to Neutral (or Tense depending on severity), reputation penalty.
- **Auction fatigue.** If every alliance petition spams a Diplomatic auction each tick, the observation's `active_auctions` slot overflows. Cadence per AuctionKind (schema §6 Q12 already flagged) — Diplomatic auctions maybe 1-per-leader-per-1000-ticks max.
- **`PostAuction` vs `PostQuest{type=Diplomacy}`.** Are they the same or different? I lean: `PostAuction` as a distinct macro verb for auction-initiation (existing `Bid` is the corresponding participate verb). Cleaner than overloading PostQuest.
- **Minimum-viable extension:** `AuctionItem::AllianceTerms`, `Payment::Service`, `AuctionResolution::Coalition`, `active_auctions` observation slot, alliance-obligation cascade, `PostAuction` macro verb. ~3 weeks including the base auction state machine work (which is OPEN in schema §6 Q1).

**Related stories:** 45 (alliances form to counter rising hegemon), 46 (leader change mid-alliance may break terms), 49 (defection from faction also terminates their alliance obligations), 51 (wars trigger cascading alliance activations).

---

## Cross-story summary

Player-facing stories ideally all end-to-end-trace cleanly through:

```
policy observation → masked action selection → event emission → cascade rule → derived view update → next-tick observation
```

### What holds together

- **Single action vocabulary.** All ten stories use PostQuest / AcceptQuest / Bid / LeaveGroup / JoinGroup / micro primitives. No story needed a new top-level verb — richness lives in `QuestType`, `AuctionKind`, mask predicates, and cascade rules. This is the strongest validation of the unification in `proposal_universal_mechanics.md`.
- **Reputation as continuous.** Story 50's reframe (drop `is_legendary` boolean, use `reputation_log` scalar throughout) integrates cleanly with the existing schema and with ability DSL scaling. Extending abilities to read reputation is a pure additive grammar change.
- **Emergent cascades via views, not systems.** Story 51's war→economy chain is the canonical proof: every link is an event-cascade or a derived view on existing state. No "economic war effect" system appears.
- **Memory ring → emotion view.** Story 53 falls out of existing spec almost verbatim.

### What's still open (cross-story)

1. **Missing QuestType variants.** `Claim` (succession), `Peace`, `Raid`, `Defend` (as auto-cascade counter-quest). All small.
2. **Missing AuctionKind extensions.** `Diplomatic` with `AllianceTerms` + `Coalition` resolution. Alliance obligations as scheduled deferred events.
3. **Missing observation features.** Each story identifies 1–3 small feature additions:
   - `known_groups[k]`: `military_strength_ratio`, `is_adjacent_territory`, `controls_scarce_resource`, `is_at_war_with_my_enemies`
   - `memberships[k]`: `group_leader_vacant`, `my_tenure_relative_to_other_seniors`, `standings_conflict_count`
   - `nearby_actors[k]`: `is_trespasser`, `other_reputation_visible`
   - Self contextual: `reputation_log` (rename of `fame_log`), `war_exhaustion_of_my_lead_group`, `n_conflicting_group_pairs`
   - New slot array: `active_auctions[K=4]`
4. **Cascade-rule inventory (missing).** Story 45 needs `QuestPosted{Conquest}→standings=AtWar`. Story 46 needs `EntityDied→LeaderVacant` (if leader). Story 47 needs `QuestCompleted{Marriage}→FoundGroup{Family}`. Story 49 needs `LeaveGroup→WasBetrayedBy` emission. Story 51 needs `BattleEnded→TreasuryLooted`. Story 52 needs `ChildBorn→spawn blended child`. Story 53 needs `FirstClassEvent→Witnessed emission`. Story 54 needs `AuctionResolved{Diplomatic}→standings=Allied+Defend cascade`. These are the cascade rules the DSL needs as first-class declarative blocks.
5. **Reward credit assignment.** Conquest (story 45), Alliance (story 54), and to some extent Marriage (story 47) all have delayed rewards. Schema §6 Q14 already flags this. Not story-specific; a foundational RL problem.
6. **Auction state machine (OPEN in mechanics doc).** Stories 46 (election), 49 (defection bidding), 54 (alliance) all depend on real auction implementation. Stub (92-line) is not enough.

### What the batch proves

- The PostQuest/AcceptQuest/Bid architecture + derived views + cascade rules genuinely collapses ~30 systems into ~5 mechanisms AND supports the entire H-category acceptance test set, with the qualifications above.
- The reframe of story 50 (reputation > `is_legendary`) works better than the original design and suggests other derived views should likewise be continuous rather than categorical (e.g. `fame`, `morale`, `loyalty`).
- Story 45's cold-start problem is real and is solved by treating ambition + contestable-territory features as observation-level signals. Bootstrapping via personality-driven exploration in the utility backend + reward gradient during training covers it.
- Stories 46, 49 validate that succession/defection need NO special system — they emerge from existing actions + the right observation slot features.
- The MISSING pieces are consistent: observation feature appends (small), cascade rule declarations (medium), auction state machine + diplomatic extension (larger). Total implementation effort across H: ~8–10 weeks of focused work, with the auction state machine being the largest single piece.
