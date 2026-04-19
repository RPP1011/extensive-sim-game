# Batch H — Player-Experienced Behaviors, Implemented in the DSL

This doc dogfoods `spec.md` by writing the actual DSL declarations that realise each Batch H story. The premise: pretend the compiler exists, target the settled grammar from `spec.md` §§2–3, and flag every place where the grammar forces a workaround or a spec gap.

Per-story structure:

- **Story reference** — behaviour statement from `stories.md` §H.N.
- **Narrative outcome** — what the player sees.
- **DSL implementation** — the actual declarations a developer writes (entities / events / views / masks / cascades / reward hooks / verbs / probes).
- **Grammar fit** — what worked, what forced a workaround.

A consolidated grammar-gap report closes the doc.

Where the spec has changed since `stories.md` (renames, new primitives, settled enums), the implementation uses the post-settlement grammar. Story text is treated as the target behavior, not the target syntax. Specifically:

- `SetStanding(target, kind)` replaces hand-coded `DeclareWar` / `MarkHostile`.
- `WithdrawQuest(quest_id)` is taker-only (§9 D9).
- `Resolution::{Coalition{min_parties}, Majority}` exist (§9 D1).
- `Payment::Combination{material, immaterial}` with per-agent `intrinsic_value` view (§9 D10).
- `movement_mode: enum {Walk, Climb, Fly, Swim, Fall}` is primary on Agent.
- `Agent.spouse_ids: SortedVec<AgentId, 4>` (§9 D17).
- `Relationship.believed_knowledge: Bitset<32>` + `believed_knowledge_refreshed: [u32;32]` (§9 D28).
- `Document { author_id, tick_written, seal, facts: [FactPayload; 16] }` (no stored `trust_score`, §9 D27).
- `Quest.child_quest_ids: SortedVec<QuestId, 8>` (§9 D15).
- `QuestType` includes `Claim`, `Peace`, `Raid`, `HaveChild`, `Defend` (§3.3).
- `AuctionKind::Diplomatic` with `AllianceTerms` + `AuctionResolution::Coalition` / `Majority` exist.

---

## Story H.45 — Wars are motivated

**Narrative outcome.** A Faction leader with a long ledger of grievances against a rival — or with high ambition, adjacent territory, and a resource the rival controls — posts a Conquest quest. The rival's standing flips to `Hostile` as a cascade consequence. Members of both groups begin raiding, defending, and dying. Caravans re-route; markets shift (story H.51 chains in). Eventually either side wins territory (QuestCompleted, Spoils cascade) or exhausts and sues for peace (`Peace` quest).

### DSL implementation

```dsl
// 1. Entity extension — Faction/Pack leaders need richer state for conquest decisions.
entity Faction : Group {
  kind: Faction,
  eligibility_predicate: creature_type.can_speak && !hostile_to_faction_kind(creature_type),
  recruitment_open: true,
  governance: Elective,
  default_standings: [(Faction, Neutral), (Pack, Neutral), (Family, Neutral)],
  required_fields: [
    territory_cells,         // Set<CellId>
    military_strength,       // f32
    treasury,                // f32
    stockpile,               // Map<Commodity, f32, 32>
    active_quests,           // SortedVec<QuestId, 16>
    standings,               // Map<GroupId, StandingValue, 32>
  ],
}

// 2. Events — war-startup consequences are all cascade-driven.
@replayable event QuestPosted { quest_id: QuestId, type: QuestType, poster: AgentId, party: PartyScope, target: QuestTarget, reward: Reward, deadline: Tick }
@replayable event WarDeclared { aggressor: GroupId, defender: GroupId, casus_belli: CasusBelli, tick: Tick }
@replayable event DefendQuestSpawned { parent_quest_id: QuestId, defender: GroupId, quest_id: QuestId }

enum CasusBelli { Grievance(u32), Expansion, Revenge, ReligiousDifference, Reclaim, Succession }

// 3. Views — derived, never stored.
@materialized(on_event = [MemoryEvent, EntityDied, EntityAttacked])
view grievance_matrix(aggrieved: GroupId, target: GroupId) -> f32 {
  initial: 0.0,
  on MemoryEvent{observer, subject, kind} where
       observer in members_of(aggrieved) &&
       subject  in members_of(target) &&
       kind in {WasAttacked, FriendDied, BetrayalWitnessed} {
    self += e.emotional_impact * recency_decay(now - e.tick)
  }
  clamp: [0.0, +inf],
}

@lazy
view controls_scarce_resource(observer: AgentId, other: GroupId) -> bool {
  exists c in Commodity:
    observer.home_settlement.stockpile[c] < 0.2 * pop(observer.home_settlement)
    && other.stockpile[c] > 0.8 * pop(home_settlement_of(other))
}

@lazy
view eligible_conquest_target(self: Agent, t: GroupId) -> bool {
  t.kind in {Faction, Pack}
  && count(a in members_of(t) where a.alive) > 0
  && self.leader_groups.any(|g| g.standings[t] != Vassal)
  && distance_to_territory(self, t) < CAMPAIGN_RANGE
}

// 4. Masks — gate Conquest PostQuest on leadership, capability, non-war, territorial reach.
mask PostQuest{type: Conquest, party: Group(g), target: Group(t)}
  when g in self.leader_groups
     && g.kind in {Faction, Pack}
     && g.military_strength > 0
     && g.standings[t] != Hostile
     && eligible_conquest_target(self, t)

// 5. Cascade — posting Conquest flips standings both ways and spawns a counter-Defend quest.
physics conquest_start_cascade @phase(event) {
  on QuestPosted{quest_id: qid, type: Conquest, party: Group(g_agg), target: Group(g_def), poster: leader} {
    // Determine casus belli from grievance_matrix at cascade time.
    let cb = if grievance_matrix(g_agg, g_def) > GRIEVANCE_HIGH { CasusBelli::Grievance(grievance_matrix(g_agg, g_def) as u32) }
             else if controls_scarce_resource(leader, g_def) { CasusBelli::Expansion }
             else { CasusBelli::Expansion };

    emit SetStandingEvent { setter: g_agg, target: g_def, kind: Hostile }
    emit SetStandingEvent { setter: g_def, target: g_agg, kind: Hostile }
    emit WarDeclared { aggressor: g_agg, defender: g_def, casus_belli: cb, tick: now }

    // Auto-spawn Defend quest on defender's leader — no explicit acceptance needed.
    let defender_leader = g_def.leader_id;
    let defend_qid = next_id();
    emit QuestPosted {
      quest_id: defend_qid, type: Defend,
      poster: defender_leader, party: Group(g_def),
      target: Location(g_def.home_pos),
      reward: Spoils(preserve_territory),
      deadline: now + 5000,
    }
    emit DefendQuestSpawned { parent_quest_id: qid, defender: g_def, quest_id: defend_qid }
  }
} @terminating_in(3)

// 6. Reward hook — fame/esteem on successful conquest + ambition-driven shaping via reward decomposition.
reward {
  delta(self.emotions.joy)      * 0.05
  +Δ(self.reputation_log)       * 0.02
  +2.0 on event(QuestCompleted{type: Conquest, party.includes(self)})
  -1.0 on event(QuestExpired{type: Conquest, party.includes(self)})
  // Leader-specific: esteem need recovery scales with ambition — no special reward, comes
  // from need-recovery term × self.personality.ambition bias.
  +delta(self.needs.esteem) * (0.1 + 0.2 * self.personality.ambition)
}

// 7. Observation additions — features the policy needs for both grievance-driven and expansion wars.
observation {
  // Append-only additions to known_groups slot (§9 D5 — slots stay at K=12).
  slots known_groups[K=6] from query::known_groups(self) sort_by relevance(self, _) {
    atom military_strength_ratio = log1p(other.military_strength) - log1p(self.lead_group.military_strength)
    atom controls_scarce_resource = view::controls_scarce_resource(self, other)
    atom is_adjacent_territory    = view::is_adjacent(self.lead_group, other)
    atom is_at_war_with_my_enemies = any(g in self.lead_group.enemies, other.standings[g] == Hostile)
    atom grievance_felt            = view::grievance_matrix(self.lead_group.id, other.id) / GRIEVANCE_MAX
  }
}
```

**Grammar fit.** Clean. `SetStandingEvent` + cascade rule replaces hand-coded "declare war". The cold-start problem (empty chronicle) is handled by `controls_scarce_resource` + ambition bias in the reward, not by a special mechanism. `@terminating_in(3)` bounds the cascade (start-war posts auto-Defend, which doesn't itself post further). No grammar gaps.

---

## Story H.46 — Succession dynamics on leader death

**Narrative outcome.** When Lord X dies, Group G's `leader_id` becomes `None`. Senior members with long tenure and high ambition post `Claim` quests. Other members accept their preferred claimant. Majority carries; coronation chronicle entry fires. Losing claimants grieve; high-ambition losers may launch coups (Conquest against their own group) or defect (LeaveGroup → JoinGroup). If no claimants act before the grace period, the group dissolves.

### DSL implementation

```dsl
// 1. Events — vacancy and resolution.
@replayable event LeaderVacant { group_id: GroupId, predecessor_id: AgentId, predecessor_tenure_ticks: u32 }
@replayable event LeaderSucceeded { group_id: GroupId, new_leader: AgentId, predecessor: AgentId, mechanism: SuccessionMechanism }
@replayable event GroupDissolved { group_id: GroupId, reason: DissolutionReason }

enum SuccessionMechanism { Election, UncontestedAscension, AlphaContest, Coup, HereditaryLine }
enum DissolutionReason { Decapitation, MemberExodus, LeaderOrder }

// 2. Views.
@lazy
view has_leader(g: GroupId) -> bool { g.leader_id.is_some() }

@materialized(on_event = [LeaderVacant, LeaderSucceeded])
view leader_vacated_tick(g: GroupId) -> Option<Tick> {
  initial: None,
  on LeaderVacant{group_id: gid} where gid == g { self = Some(e.tick) }
  on LeaderSucceeded{group_id: gid} where gid == g { self = None }
}

@lazy
view active_claim_quests(g: GroupId) -> [QuestId] {
  query::active_quests_where(|q| q.type == Claim && q.party == Group(g))
}

@lazy
view senior_member(self: Agent, g: GroupId) -> bool {
  let m = self.membership_in(g);
  m.is_some() && m.role in {Officer, Founder, Alpha}
    && now - m.joined_tick > MIN_SUCCESSION_TENURE
}

// 3. Masks — governance-sensitive Claim eligibility.
mask PostQuest{type: Claim, party: Group(g), target: Role(Leader)}
  when g in self.memberships
     && !has_leader(g)
     && view::senior_member(self, g)
     && match g.governance {
         Hereditary    => self in g.hereditary_chain,
         Elective      => true,                                 // any senior officer
         Council       => count(g.members.officers) >= 3,        // council quorum
         Theocratic    => self in g.charter_eligible,
         AlphaContest  => self.creature_type.is_packable,        // wolves handle this via combat cascade, not Claim
         AdHoc         => true,
       }

mask AcceptQuest{quest_id: q}
  where quest(q).type == Claim
  when self in quest(q).eligible_acceptors              // defender: members of quest(q).party.group
     && !self.in_any_active_claim_for(quest(q).party.group)   // can't back two claimants
     && !self.role_in(quest(q).party.group).is_some_and(|r| r == Outcast)

// 4. Cascade — leader death → vacancy; majority acceptance → succession; grace-period → dissolution.
physics leader_death_vacancy @phase(event) {
  on EntityDied{agent_id: dead} {
    for g in groups_where(leader_id == dead) {
      emit SetGroupLeader { group_id: g.id, new_leader: None }
      emit LeaderVacant {
        group_id: g.id,
        predecessor_id: dead,
        predecessor_tenure_ticks: now - g.leader_since_tick,
      }
    }
  }
}

physics claim_majority_resolution @phase(event) {
  on QuestCompleted{quest_id: qid, resolution: Majority}
    where quest(qid).type == Claim {
    let winner_candidate = quest(qid).target.unwrap_agent();  // target = Agent(claimant)
    let g = quest(qid).party.unwrap_group();
    emit SetGroupLeader { group_id: g, new_leader: Some(winner_candidate) }
    emit LeaderSucceeded {
      group_id: g, new_leader: winner_candidate,
      predecessor: g.last_predecessor,
      mechanism: match g.governance { Elective => Election, Hereditary => HereditaryLine, _ => Election },
    }
    // Cascade: losing claimants grieve.
    for other_qid in active_claim_quests(g).filter(|q| q != qid) {
      emit QuestCancelled { quest_id: other_qid, reason: SupersededBySuccessor }
      emit RecordMemory {
        observer: quest(other_qid).poster,
        payload: FactPayload::SuccessionLost { won_by: winner_candidate, tick: now },
        source: Witnessed, confidence: 1.0,
      }
    }
  }
}

physics decapitation_grace_timeout @phase(post) {
  on TickBoundary {
    for g in groups_where(!has_leader(_)) {
      if let Some(vt) = leader_vacated_tick(g.id)
         && now - vt > DECAPITATION_GRACE
         && active_claim_quests(g.id).is_empty() {
        emit GroupDissolved { group_id: g.id, reason: Decapitation }
      }
    }
  }
} @terminating_in(1)

// 5. Reward hooks — winning a Claim pays esteem + reputation; losing hurts.
reward {
  +3.0  on event(LeaderSucceeded{new_leader: self})
  -1.0  on event(QuestCancelled{quest_id, reason: SupersededBySuccessor})
         where quest(quest_id).poster == self
  +delta(self.emotions.pride)   * 0.1
  -delta(self.emotions.grief)   * 0.05
}

// 6. Observation — vacancy must be visible.
observation {
  slots memberships[K=8] from self.memberships sort_by relevance(self, _) {
    atom group_leader_vacant      = view::!has_leader(this.group_id)
    atom my_tenure_rank_in_group  = rank(self, this.group.officers_by_tenure) / count(this.group.officers)
    atom active_claim_count       = count(view::active_claim_quests(this.group_id))
  }
}
```

**Grammar fit.** Good. `Role(RoleTag)` in `QuestTarget` works — Claim targets `Role(Leader)`. `Resolution::Majority` is needed on the Claim quest — §9 D1 covers it. `AlphaContest` governance is listed as an enum variant but the actual resolution for a wolf pack happens via combat cascade (story H.48), not via Claim — the mask routes around it with the `is_packable` predicate → `creature_type.can_speak` + `Claim` gating fails for non-speakers.

**Gap found.** The spec's `governance` enum in §2.1 lists `{Hereditary, Elective, Council, Theocratic, AlphaContest, AdHoc}` — all good. But the **Claim eligibility predicate** needs a governance-conditional body, which currently requires a `match` expression inside a mask. Spec §2.5 says "No user-defined functions inside masks." The `match` branches over an enum are not a user function — they're pattern arms — but this needs to be explicit in the spec grammar. **Recommend** clarifying §2.5: match expressions over enum fields are permitted in masks.

---

## Story H.47 — Marriages by trust + compatibility

**Narrative outcome.** Agent A, single and with sufficient familiarity + shared groups with Agent B, proposes marriage. If B accepts, a Family Group forms, both become Founders, spouse_ids updates mutually. Mood spikes for both; surrounding witnesses memory-ring the event. Failed proposals grief A and cause personality drift.

### DSL implementation

```dsl
// 1. Events.
@replayable event MarriageProposed { proposer: AgentId, target: AgentId, quest_id: QuestId, tick: Tick }
@replayable event MarriageFormed   { a: AgentId, b: AgentId, family_group_id: GroupId, tick: Tick }
@replayable event MarriageRejected { proposer: AgentId, refused_by: AgentId, tick: Tick }

// 2. Views — compatibility + cultural eligibility.
@lazy
view compatibility_score(self: Agent, other: AgentId) -> f32 {
  let other_seen = self.known_actors.find(other);
  let perceived  = other_seen.map(|r| r.perceived_personality).unwrap_or(Personality::default());
  // Cosine similarity on 5-d personality, weighted by confidence.
  let sim = cosine(self.personality.as_vec5(), perceived.traits);
  sim * other_seen.map(|r| r.familiarity).unwrap_or(0.0)
}

@lazy
view marriage_eligibility(a: Agent, b: Agent) -> bool {
  a.creature_type == b.creature_type            // same species (v1)
  && forall g in a.memberships: match g.kind {
       Religion => b.memberships.contains(|m| m.group_id == g.group_id || m.group.kind != Religion),
       Faction  => !a.lead_group.standings.get(b.primary_faction).eq_some(Hostile),
       _        => true,
     }
}

@lazy
view has_active_marriage_quest(self: Agent) -> bool {
  self.active_quests.iter().any(|qid| quest(qid).type == Marriage && quest(qid).poster == self)
}

// 3. Mask.
mask PostQuest{type: Marriage, party: Individual(self), target: Agent(t), reward: Union}
  when self.creature_type.can_marry
     && count(self.spouse_ids) < self.creature_type.max_spouses         // D17 — polygamous-capable
     && !t.has_max_spouses()
     && t.alive
     && relationship(self, t).valence > MIN_MARRIAGE_TRUST
     && view::marriage_eligibility(self, t)
     && !view::has_active_marriage_quest(self)
     // distance NOT required — political marriages are arranged via cross-territory messages.

mask AcceptQuest{quest_id: q}
  where quest(q).type == Marriage
  when self in quest(q).eligible_acceptors       // eligible = [target only]
     && count(self.spouse_ids) < self.creature_type.max_spouses
     && view::marriage_eligibility(self, quest(q).poster)

// 4. Cascades — on propose, mark memory for both sides; on accept, form Family group; on expire, grieve.
physics marriage_propose_cascade @phase(event) {
  on QuestPosted{quest_id: qid, type: Marriage, poster: a, target: Agent(b)} {
    emit MarriageProposed { proposer: a, target: b, quest_id: qid, tick: now }
    emit RecordMemory {
      observer: b, payload: FactPayload::ReceivedMarriageProposal { from: a, tick: now, qid },
      source: Witnessed, confidence: 1.0,
    }
  }
}

physics marriage_accept_cascade @phase(event) {
  on QuestCompleted{quest_id: qid, resolution: MutualAgreement}
    where quest(qid).type == Marriage {
    let a = quest(qid).poster;
    let b = quest(qid).target.unwrap_agent();
    let family_id = next_id();
    emit FoundGroup {
      group_id: family_id, kind: Family,
      founders: [a, b],
      governance: AdHoc,
      recruitment_open: false,
    }
    emit AddSpouseLink { a: a, b: b }
    emit AddSpouseLink { a: b, b: a }
    emit AddMembership { agent: a, group_id: family_id, role: Founder }
    emit AddMembership { agent: b, group_id: family_id, role: Founder }
    emit MarriageFormed { a, b, family_group_id: family_id, tick: now }
    // Witness cascade — anyone in social range of either spouse remembers it.
    let witnesses = query::nearby_agents(a.pos, radius: 25.0).filter(|w| w != a && w != b && w.can_hear);
    for w in witnesses {
      emit RecordMemory {
        observer: w, payload: FactPayload::MarriageWitnessed { a, b, tick: now },
        source: Witnessed, confidence: 1.0,
      }
    }
  }
}

physics marriage_expire_cascade @phase(event) {
  on QuestExpired{quest_id: qid} where quest(qid).type == Marriage {
    let a = quest(qid).poster;
    let b = quest(qid).target.unwrap_agent();
    emit MarriageRejected { proposer: a, refused_by: b, tick: now }
    emit EmotionDelta { agent: a, dim: grief, delta: +0.15 }
    emit PersonalityDrift { agent: a, trait: social_drive, delta: -0.01 }
    emit RecordMemory {
      observer: a, payload: FactPayload::RejectedByMarriageTarget { by: b, tick: now },
      source: Witnessed, confidence: 1.0,
    }
  }
}

// 5. Reward shaping.
reward {
  +2.5  on event(MarriageFormed{a: self} | MarriageFormed{b: self})
  -1.0  on event(MarriageRejected{proposer: self})
  +delta(self.needs.social) * 0.1
}

// 6. Observation.
observation {
  slots known_actors[K=10] from query::known_actors(self) sort_by relevance(self, _) {
    atom compatibility_score = view::compatibility_score(self, other)
    atom is_married          = other.has_any_spouse
    atom shared_non_family_groups = count(overlap(self.memberships, other.memberships).filter(|g| g.kind != Family))
  }
}
```

**Grammar fit.** Clean. Polygamy is handled via `spouse_ids: SortedVec<AgentId, 4>` + `creature_type.max_spouses` config. `Resolution::MutualAgreement` works perfectly for 2-party proposal→accept. `FoundGroup` cascade output is a natural first-class event.

**Gap found.** `eligible_acceptors` for a Marriage quest is a singleton `[target]`. The spec's `QuestTarget::Agent(AgentId)` + the resolution `MutualAgreement` implicitly pairs them, but the **derivation of `eligible_acceptors` from `QuestTarget + PartyScope`** isn't spelled out. Proposed clarification in §3.3 table: for `PartyScope::Individual(poster)` + `QuestTarget::Agent(t)`, `eligible_acceptors = {t}`.

---

## Story H.48 — Monsters defend dens, hunt prey

**Narrative outcome.** A wolf pack hunts deer. The pack alpha's successful kill is witnessed by pack members, subtly biasing their learning toward cooperative hunting. When a human approaches the pack's den, proximity-to-den raises the wolves' anxiety and flips `is_hostile(wolf, human)` via the trespasser predicate — wolves attack. Pack members, seeing the alpha engage, join. An alpha's death triggers a fight-succession (AlphaContest governance).

### DSL implementation

```dsl
// 1. Entity.
entity WolfAgent : Agent {
  creature_type: Wolf,
  capabilities: Capabilities {
    can_speak: false, can_build: false, can_trade: false,
    can_hear: true, hearing_range: 80.0,
    is_predator: true, is_packable: true,
    max_spouses: 1,
  },
  default_personality: Personality { risk: 0.6, social: 0.7, ambition: 0.3, compassion: 0.2, curiosity: 0.3 },
  default_needs: Needs { hunger: 0.0, safety: 0.5, social: 0.4, shelter: 0.3 },
  hunger_drives: [HungerDriveKind::EatPrey],
  default_memberships: [GroupSpec { kind: Pack, role: Member }],
  predator_prey: { prey_of: [], preys_on: [Deer, Rabbit, Boar] },
}

entity Pack : Group {
  kind: Pack,
  eligibility_predicate: creature_type == Wolf,
  recruitment_open: false,
  governance: AlphaContest,
  default_standings: [(Pack, Neutral), (Faction, Neutral)],
  required_fields: [alpha_id, territory_centroid, den_pos, hunting_ground_cells],
}

// 2. Events.
@replayable event HuntSuccess  { hunter: AgentId, prey: AgentId, observers: [AgentId; 8], tick: Tick }
@replayable event TrespassDetected { agent: AgentId, by_group: GroupId, distance_from_core: f32, tick: Tick }
@replayable event AlphaDefeated { old_alpha: AgentId, new_alpha: AgentId, pack: GroupId, tick: Tick }

// 3. Views — the registry-driven predator_prey + trespasser + den.
@lazy
view predator_prey(a: CreatureType, b: CreatureType) -> bool {
  Registry::predator_prey_table.contains((a, b))
}

@lazy
view is_trespasser(observer: Agent, other: Agent) -> bool {
  forall p in observer.memberships.filter(|m| m.group.kind == Pack):
    let den = p.group.den_pos;
    distance(other.pos, den) < DEN_RADIUS
      && !p.group.members.contains(other)
}

@lazy
view is_hostile(self: Agent, other: Agent) -> bool {
  relationship(self, other).valence < HOSTILE_THRESH
  || groups_at_war(self, other)
  || predator_prey(self.creature_type, other.creature_type)
  || view::is_trespasser(self, other)
}

@lazy
view threats_sort_key(self: Agent, other: Agent) -> f32 {
  let d = distance(self, other);
  let velocity_toward_me = dot(other.velocity, (self.pos - other.pos).normalize());
  if predator_prey(other.creature_type, self.creature_type) {
    -(velocity_toward_me / d.max(0.1))   // reverse-predator ⇒ prey sees wolf as threat
  } else if view::is_hostile(self, other) {
    -(other.attack_damage / d.max(0.1))
  } else {
    +inf
  }
}

// 4. Mask — Attack already covered by is_hostile; no new predicate needed.
// 5. Cascade — kill → HuntSuccess → bias pack's action_outcomes EMA.
physics hunt_witness_cascade @phase(event) {
  on EntityDied{agent_id: prey, killer: Some(hunter), cause: Combat} {
    let pack_ids = hunter.memberships.iter().filter(|m| m.group.kind == Pack).map(|m| m.group_id);
    let observers = query::nearby_agents(hunter.pos, radius: 40.0)
                      .filter(|w| w != hunter && pack_ids.any(|p| w.is_member_of(p)))
                      .take(8);
    emit HuntSuccess { hunter, prey, observers, tick: now }
    for o in observers {
      emit RecordMemory {
        observer: o, payload: FactPayload::PackHuntSuccess { hunter, prey, tick: now },
        source: Witnessed, confidence: 1.0,
      }
    }
  }
}

physics trespass_detection_cascade @phase(post) {
  on MovementApplied{agent: intruder} {
    let nearby_packs = query::nearby_groups(intruder.pos, kind: Pack, radius: DEN_RADIUS);
    for p in nearby_packs {
      if !p.members.contains(intruder) {
        emit TrespassDetected {
          agent: intruder, by_group: p.group_id,
          distance_from_core: distance(intruder.pos, p.den_pos),
          tick: now,
        }
        // Every pack member in line of sight gets a memory — shapes their need.safety.
        for w in p.members.filter(|m| distance(m.pos, intruder.pos) < 100.0 && m.alive) {
          emit RecordMemory {
            observer: w, payload: FactPayload::IntruderNearDen { intruder, tick: now },
            source: Witnessed, confidence: 1.0,
          }
          emit EmotionDelta { agent: w, dim: anxiety, delta: +0.1 }
        }
      }
    }
  }
} @terminating_in(1)

physics alpha_succession_combat @phase(event) {
  on EntityDied{agent_id: dead} {
    for p in groups_where(alpha_id == dead && kind == Pack) {
      emit SetGroupLeader { group_id: p.id, new_leader: None }
      // AlphaContest governance: next tick, highest-stat wolf emits Attack on fellow contenders.
      // No Claim quest; combat resolves it.
    }
  }
}

physics alpha_contest_resolution @phase(event) {
  on EntityDied{agent_id: dead, killer: Some(killer), cause: Combat}
    where !has_leader(dead.last_leader_pack)
       && killer.is_member_of(dead.last_leader_pack)
       && killer.creature_type == Wolf {
    emit SetGroupLeader { group_id: dead.last_leader_pack, new_leader: Some(killer) }
    emit AlphaDefeated { old_alpha: dead, new_alpha: killer, pack: dead.last_leader_pack, tick: now }
  }
}

// 6. Reward shaping — wolves.
reward when self.creature_type == Wolf {
  +delta(self.needs.hunger) * (-0.2)                   // eating reduces hunger = reward
  +2.0 on event(HuntSuccess{hunter: self})
  +0.5 on event(HuntSuccess{observers.contains(self)}) // observation-only reward
  +3.0 on event(AlphaDefeated{new_alpha: self})
  -0.2 on event(TrespassDetected{by_group: self.lead_pack})  // anxiety source
}
```

**Grammar fit.** Strong. Reward conditioning on `self.creature_type` (via `reward when ...`) is the one clean way to specialise without a separate policy. Pack succession-via-combat uses the existing `EntityDied` cascade — no new "alpha_contest" mechanism needed.

**Gap found.** The spec §3.4 shows a global `reward { ... }` block. **Creature-conditional reward blocks** (`reward when self.creature_type == Wolf`) aren't explicit in the grammar. Options: allow `reward when <predicate> { ... }` alongside the unconditional block, additive; or require all creature-type differences to live in the same block under `match` pattern-arms. The former is cleaner for modding. **Recommend** adding to §3.4.

---

## Story H.49 — Some NPCs betray for personal gain

**Narrative outcome.** An ambitious, undervalued agent in Faction G_home emits `LeaveGroup(G_home)`, then `JoinGroup(G_rival)` (or accepts a Service quest from G_rival). Former comrades' relationships degrade; the defector accumulates a `Betrayed` memory in their rings. Under binding oaths, the agent must first `BreakOath`, paying a reputation penalty. Multi-membership can produce loyalty conflicts.

### DSL implementation

```dsl
// 1. Events.
@replayable event AgentLeftGroup     { agent: AgentId, group: GroupId, reason: LeaveReason, tick: Tick }
@replayable event AgentJoinedGroup   { agent: AgentId, group: GroupId, role: MembershipRole, tick: Tick }
@replayable event OathBroken         { agent: AgentId, oath: OathRef, target_group: GroupId, tick: Tick }
@replayable event LoyaltyConflictExpelled { agent: AgentId, expelled_from: GroupId, retained: GroupId, tick: Tick }

enum LeaveReason { Voluntary, Expelled, Dissolved, Oath_Broken }

// 2. Views.
@lazy
view binding_oaths_with(self: Agent, g: GroupId) -> [OathRef] {
  self.memory_events.iter()
    .filter(|e| e.kind == OathSworn && e.target_group == g)
    .filter(|e| !self.memory_events.any(|e2| e2.kind == OathBroken && e2.oath == e.oath_ref))
    .map(|e| e.oath_ref)
    .take(8)
}

@lazy
view n_conflicting_group_pairs(self: Agent) -> u32 {
  let mut count = 0;
  for (i, a) in self.memberships.enumerate() {
    for b in self.memberships[i+1..] {
      if standing(a.group_id, b.group_id) == Hostile { count += 1 }
    }
  }
  count
}

@lazy
view trust_drop_on_betrayal(ex_ally: Agent, defector: AgentId) -> f32 {
  let r = relationship(ex_ally, defector);
  if r.familiarity > 0.5 && r.valence > 0.0 { 0.3 }
  else if r.familiarity > 0.2 { 0.1 }
  else { 0.02 }
}

// 3. Masks.
mask LeaveGroup(g)
  when g in self.memberships
     && view::binding_oaths_with(self, g).is_empty()
     && !(self.role_in(g) == Founder && g.founders_alive_count == 1)
  // Leaders CAN leave but pay large reward penalty (see reward block).

mask JoinGroup(g)
  when g.recruitment_open
     && self.eligible_for(g)                  // uses g.eligibility_predicate
     && g.kind != Family                      // family joins happen via MarriageFormed or ChildBorn
     && !g.id in self.memberships.map(|m| m.group_id)

mask BreakOath(oath_ref: o)
  when o in view::binding_oaths_with(self, _any_)
     && self.personality.ambition > OATH_BREAK_AMBITION_THRESH

// 4. Cascades.
physics leave_group_cascade @phase(event) {
  on AgentLeftGroup{agent: defector, group: g, reason: r} {
    emit RemoveMembership { agent: defector, group_id: g }
    // Reputation cascade: every existing member with familiarity > 0 remembers.
    for member in g.members.filter(|m| m != defector) {
      let r_val = relationship(member, defector);
      if r_val.familiarity > 0.0 {
        let kind = if r_val.valence > 0.2 && r != Expelled { BetrayalWitnessed } else { AllyLeft };
        emit RecordMemory {
          observer: member,
          payload: FactPayload::AgentLeftGroupFact { defector, group: g, kind, tick: now },
          source: Witnessed, confidence: 1.0,
        }
        emit RelationshipDelta {
          a: member, b: defector,
          valence: -view::trust_drop_on_betrayal(member, defector),
        }
      }
    }
  }
}

physics join_group_welcome_cascade @phase(event) {
  on AgentJoinedGroup{agent: joiner, group: g, role: _} {
    emit AddMembership { agent: joiner, group_id: g, role: Member }
    for member in g.members.filter(|m| m != joiner) {
      emit RecordMemory {
        observer: member,
        payload: FactPayload::AgentJoinedGroupFact { joiner, group: g, tick: now },
        source: Witnessed, confidence: 1.0,
      }
      emit RelationshipDelta {
        a: member, b: joiner,
        valence: +0.05, familiarity: +0.05,
      }
    }
  }
}

physics break_oath_cascade @phase(event) {
  on BreakOath{agent: a, oath: o} {
    emit OathBroken { agent: a, oath: o, target_group: o.target_group, tick: now }
    emit ReputationDelta { agent: a, delta: -OATH_BREAK_PENALTY }
    // Chronicle.
    for member in o.target_group.members {
      emit RecordMemory {
        observer: member, payload: FactPayload::OathBrokenBy { a, oath: o, tick: now },
        source: Witnessed, confidence: 1.0,
      }
    }
  }
}

// Optional passive resolution for prolonged multi-membership conflict.
physics loyalty_conflict_expulsion @phase(post) {
  on TickBoundary where now % LOYALTY_CHECK_CADENCE == 0 {
    for a in query::alive_agents() {
      if view::n_conflicting_group_pairs(a) > 0 {
        let conflict_age = now - a.last_conflict_resolution_tick;
        if conflict_age > LOYALTY_GRACE {
          let (g_keep, g_drop) = pick_loyalty_by_standing_and_tenure(a);
          emit AgentLeftGroup { agent: a, group: g_drop, reason: Expelled, tick: now }
          emit LoyaltyConflictExpelled { agent: a, expelled_from: g_drop, retained: g_keep, tick: now }
        }
      }
    }
  }
} @terminating_in(1)

// 5. Reward — leaving when Leader is costly; successful JoinGroup brings welcome bump.
reward {
  -0.5 on event(AgentLeftGroup{agent: self, group: g}) where self.was_role_in(g) == Leader
  -1.5 on event(OathBroken{agent: self})
  +0.3 on event(AgentJoinedGroup{agent: self})
  -delta(relationship(self, avg_ex_ally).valence) * 0.02    // aggregate grief from reputation cascade
}

// 6. Observation.
observation {
  atom n_conflicting_group_pairs = view::n_conflicting_group_pairs(self)
  slots memberships[K=8] from self.memberships sort_by relevance(self, _) {
    atom my_standing_rank_in_group = view::standing_rank(self, this.group)
    atom has_binding_oath_to_group = !view::binding_oaths_with(self, this.group_id).is_empty()
    atom group_activity_ewma       = this.group.activity_log_ewma
  }
}
```

**Grammar fit.** Clean. Multi-membership conflict is handled with an explicit cascade (`loyalty_conflict_expulsion`) for the extreme case; normal pressure goes through rewards. `BreakOath` is a distinct micro primitive (listed in spec §3.2? — see gap below).

**Gap found.** Spec §3.2's `micro_kind` enum list doesn't currently include `BreakOath`. Options: (a) add as a top-level micro primitive (bumps schema hash); (b) synthesise via `verb BreakOath(oath_ref) = action Announce(...)`. (a) is cleaner because BreakOath is a fundamental action with unique cascade consequences. **Recommend** adding `BreakOath` to the `micro_kind` enum (append-only, bumps schema hash per §4).

---

## Story H.50 — Renowned characters emerge from reputation

**Narrative outcome.** Reputation is continuous, derived from chronicle mentions + witnessed deeds × recency. High-reputation agents scale combat damage, can petition charters, intimidate effectively. A "legendary hero" is just the top-K reputation agent; no flag. Their death triggers widespread grief because their deeds are widely-witnessed.

### DSL implementation

```dsl
// 1. Views — reputation as a first-class continuous view, not stored state.
@materialized(on_event = [ChronicleEntry, MemoryEvent])
view reputation_log(agent: AgentId) -> f32 {
  initial: 0.0,
  on ChronicleEntry{entity_ids} where entity_ids.contains(agent) {
    self += chronicle_weight(e.category) * recency_decay(now - e.tick)
  }
  on MemoryEvent{subject} where subject == agent {
    self += e.emotional_impact * recency_decay(now - e.tick) * 0.1
  }
  clamp: [-REPUTATION_MAX, REPUTATION_MAX],
}

@lazy
view reputation_known_in_settlement(self: AgentId, s: SettlementId) -> f32 {
  // Local bonus — hometown heroes have higher perceived reputation.
  if agent(self).home_settlement_id == s {
    view::reputation_log(self) * LOCAL_REPUTATION_MULTIPLIER
  } else {
    view::reputation_log(self)
  }
}

@lazy
view intimidate_strength(self: AgentId) -> f32 {
  (view::reputation_log(self)).abs()       // negative rep (infamy) also intimidates
}

// 2. Ability scaling — reputation enters ability formulas.
ability Attack {
  effect Damage {
    target: Agent
    magnitude: self.attack_damage * (1.0 + 0.05 * view::reputation_log(self.id).clamp(-10, 10))
  }
  conditions: view::is_hostile(self, target)
}

ability Intimidate {
  effect Fear {
    target: Agent
    magnitude: base + 0.1 * view::intimidate_strength(self.id)
    duration_ticks: 30
  }
  conditions: relationship(self, target).trust < 0 || target.creature_type.can_speak
}

// 3. Masks — reputation-gated charters / claims / auction visibility.
mask PostQuest{type: Charter, party: Individual(self), target: Settlement(s)}
  when view::reputation_log(self.id) > CHARTER_REPUTATION_THRESH
     || self in s.charter_eligible_by_peerage

mask PostQuest{type: Claim, party: Group(g), target: Role(Leader)}
  when view::senior_member(self, g)
     && view::reputation_log(self.id) > MIN_CLAIM_REPUTATION.get_or(g.governance, 0.0)
     && !has_leader(g)

mask Bid{auction_id: a, payment: p, conditions: _}
  when auction(a).visibility.includes_self_or_rep(self, view::reputation_log(self.id))
     && self.gold >= auction(a).reserve_price.gold_amount_equiv()

// 4. Cascades — cosmetic legend threshold; no mechanical flag.
physics legend_ascended_chronicle @phase(post) {
  on MaterializedViewChanged{view: reputation_log, agent: a, old_value: v0, new_value: v1}
    where v0 < LEGENDARY_REPUTATION_THRESH && v1 >= LEGENDARY_REPUTATION_THRESH {
    emit ChronicleEntry {
      category: Legendary,
      entity_ids: [a, 0, 0, 0],
      template_id: Chronicle::LEGEND_ASCENDED,
      tick: now,
    }
  }
} @terminating_in(1)

// Death of a legendary agent propagates widespread grief via the standard Witnessed cascade —
// no special-case code; the memory ring + emotion view handle it.
// 5. Reward — continuous reputation growth is incentive.
reward {
  +delta(view::reputation_log(self.id)) * 0.02
}

// 6. Observation.
observation {
  self.reputation_log = view::reputation_log(self.id)
  self.is_legendary_tier = self.reputation_log > LEGENDARY_REPUTATION_THRESH
  slots known_actors[K=10] from query::known_actors(self) sort_by relevance(self, _) {
    atom other_reputation_visible = view::reputation_known_in_settlement(
      other.id, self.home_settlement_id
    ).clamp(-10, 10) / 10.0
  }
  slots nearby_actors[K=12] from query::nearby_agents(self, radius: 50.0) sort_by distance(self, _) {
    atom other_reputation_visible = view::reputation_known_in_settlement(
      other.id, self.home_settlement_id
    ).clamp(-10, 10) / 10.0
  }
}
```

**Grammar fit.** Strong. Ability-side reputation scaling uses existing ability DSL expressions (`.ability` files). `@materialized` reputation view auto-updates on chronicle + memory events; no hand-coded decay cascade — the view's fold handles decay via `recency_decay()` evaluated at read time.

**Gap found.** The `MaterializedViewChanged` event firing on threshold crossings isn't explicit in spec §2.3. The spec says `@materialized` triggers event-dispatch handlers that *write* the view; it doesn't say it *emits* a change event. To implement the "legend ascended" chronicle entry, we either (a) add a `@emits_on_threshold(field, value)` annotation to the view, or (b) convert to a cascade that polls `reputation_log` at `@phase(post)` and checks crossings. Option (a) is cleaner; option (b) is achievable with the current grammar. **Recommend** clarifying that either works.

---

## Story H.51 — Long wars affect economy

**Narrative outcome.** A Conquest quest starts (story H.45). Over the ensuing 2000+ ticks: raids loot treasuries, caravans reroute to avoid AtWar territory, scarce commodities spike in price, NPCs migrate or starve, morale drops, defections rise. Eventually the losing leader sues for peace (`PostQuest{Peace}`) or is overthrown. Every link is an existing action, cascade, or derived view — no dedicated "war economy system."

### DSL implementation

```dsl
// 1. Events.
@replayable event TreasuryLooted  { settlement: SettlementId, looter_group: GroupId, amount: f32, tick: Tick }
@replayable event BuildingDamaged { building_id: ItemId, damage: f32, tick: Tick }
@replayable event CaravanRerouted { caravan: AgentId, old_path: PathId, new_path: Option<PathId>, reason: RerouteReason, tick: Tick }

enum RerouteReason { PathUnsafe, DestinationHostile, TravelCostTooHigh }

// 2. Views — treasury trend, safe paths, population, prices, war exhaustion.
@materialized(on_event = [TreasuryLooted, TreasuryDeposit, TreasuryWithdraw])
view treasury_trend(settlement: SettlementId) -> f32 {
  initial: 0.0,
  on TreasuryLooted{settlement: sid, amount} where sid == settlement { self -= amount }
  on TreasuryDeposit{settlement: sid, amount} where sid == settlement { self += amount }
  on TreasuryWithdraw{settlement: sid, amount} where sid == settlement { self -= amount }
  decay: self *= 0.995 per tick,                  // 200-tick half-life smoothing
}

@lazy
view safe_path(path: PathId, for_agent: AgentId) -> bool {
  let mem = for_agent.primary_faction;
  forall cell in path.cells:
    cell.owning_group == None
    || standing(cell.owning_group, mem) != Hostile
}

@lazy
view population(s: SettlementId) -> u32 {
  count(a in query::alive_agents() where a.home_settlement_id == s)
}

@materialized(on_event = [ItemHarvested, ItemProduced, ItemConsumed])
view stockpile(s: SettlementId, c: Commodity) -> f32 {
  initial: s.initial_stockpile[c],
  on ItemProduced{settlement: sid, commodity: ci, amount} where sid == s && ci == c { self += amount }
  on ItemConsumed{settlement: sid, commodity: ci, amount} where sid == s && ci == c { self -= amount }
  clamp: [0.0, +inf],
}

@lazy
view commodity_price(s: SettlementId, c: Commodity) -> f32 {
  let pop = view::population(s).max(1) as f32;
  Registry::base_price[c] / (1.0 + view::stockpile(s, c) / (pop * Registry::halflife[c]))
}

@materialized(on_event = [EntityDied, TreasuryLooted, BuildingDamaged, QuestPosted])
view war_exhaustion(g: GroupId) -> f32 {
  initial: 0.0,
  on EntityDied{agent_id: a} where a.primary_faction == g {
    self += CASUALTY_WEIGHT * recency_decay(now - e.tick)
  }
  on TreasuryLooted{looter_group, amount} where looter_group != g && owning_group(e.settlement) == g {
    self += amount * TREASURY_WEIGHT
  }
  on QuestPosted{type: Conquest, party: Group(gp)} where gp == g {
    self += DURATION_WEIGHT     // ongoing-war tick contribution
  }
  decay: self *= 0.998 per tick,
  clamp: [0.0, 1.0],
}

// 3. Masks — caravan travel requires safe path; peace quest requires exhaustion.
mask Travel(path: p)
  when view::safe_path(p, self)
     && self.needs.safety > MIN_TRAVEL_SAFETY

mask PostQuest{type: Peace, party: Group(g), target: Group(t)}
  when g in self.leader_groups
     && g.standings[t] == Hostile
     && view::war_exhaustion(g) > PEACE_EXHAUSTION_THRESH

// 4. Cascades — raids loot treasuries + damage buildings; peace resolves war.
physics battle_end_loot_cascade @phase(event) {
  on BattleEnded{grid_id: gid, victor: v, tick} where v.is_some() {
    let settlement = settlement_at(gid);
    if let Some(s) = settlement {
      // Loot proportional to battlefield control.
      let loot = (s.treasury * LOOT_FRACTION).min(v.unwrap().carry_capacity());
      emit TreasuryLooted {
        settlement: s.id, looter_group: v.unwrap().primary_faction,
        amount: loot, tick: now,
      }
      emit TransferGold { from: s.treasury, to: v.unwrap().primary_faction.treasury, amount: loot }
      // Building damage proportional to battle intensity.
      for b in s.buildings {
        emit BuildingDamaged { building_id: b.id, damage: battle_intensity(gid) * 0.1, tick: now }
      }
    }
  }
}

physics caravan_reroute_cascade @phase(event) {
  on SetStandingEvent{setter, target, kind: Hostile} {
    let affected_routes = query::trade_routes_through(setter.territory)
                       + query::trade_routes_through(target.territory);
    for route in affected_routes {
      let caravans = query::caravans_on(route);
      for c in caravans {
        let new_path = query::best_safe_path(c.pos, route.destination, c);
        emit CaravanRerouted {
          caravan: c, old_path: route.id, new_path,
          reason: PathUnsafe, tick: now,
        }
      }
    }
  }
}

physics peace_acceptance_cascade @phase(event) {
  on QuestCompleted{quest_id: qid, resolution: MutualAgreement}
    where quest(qid).type == Peace {
    let g_a = quest(qid).party.unwrap_group();
    let g_b = quest(qid).target.unwrap_group();
    emit SetStandingEvent { setter: g_a, target: g_b, kind: Neutral }
    emit SetStandingEvent { setter: g_b, target: g_a, kind: Neutral }
    emit ChronicleEntry {
      category: WarEnded,
      entity_ids: [g_a.leader_id, g_b.leader_id, 0, 0],
      template_id: Chronicle::PEACE_SIGNED,
      tick: now,
    }
  }
}

// 5. Reward — war exhaustion is policy signal, not a reward; peace completion pays.
reward {
  +5.0 on event(QuestCompleted{type: Peace, party.includes(self)})
  -0.1 on event(EntityDied{agent_id: a}) where a.primary_faction == self.primary_faction
  -0.05 * view::war_exhaustion(self.primary_faction) per tick     // continuous exhaustion drag
}

// 6. Observation additions.
observation {
  self.war_exhaustion_of_my_lead_group = view::war_exhaustion(self.lead_group.id)
  self.treasury_trend_of_my_settlement = view::treasury_trend(self.home_settlement_id)
  slots known_groups[K=6] from query::known_groups(self) sort_by relevance(self, _) {
    atom peer_war_exhaustion = view::war_exhaustion(other.id)
    atom peer_treasury_trend = view::treasury_trend(other.home_settlement_id)
  }
}
```

**Grammar fit.** Strongest validation of the spec so far. Every cascade is local (`BattleEnded → loot settlement`, `SetStanding → reroute caravans`); every view composes cleanly. The continuous exhaustion drag in reward (`-0.05 * view::war_exhaustion(...) per tick`) is a clean way to incentivize peace without hand-coded fatigue logic.

**Gap found.** The `per tick` suffix on reward terms that integrate a view over time isn't in spec §3.4's reward grammar examples. Currently §3.4 shows `delta(X) × 0.1` (implicitly per tick) and `+1.0 on event(...)`. A continuous reward on a view needs explicit `per tick` or an alternative like `+X * 0.1`. **Recommend** spec §3.4 add `per_tick <scalar_expr> × <coefficient>` as a first-class form.

---

## Story H.52 — Children inherit parental traits

**Narrative outcome.** Two married agents in a Family group co-located → gestation (500 ticks) → `ChildBorn`. The child spawns with blended personality + cultural bias + class seed, starts level 0, inherits Family membership and home_settlement, has parents in its `known_actors` with high trust. Adult-gated actions (Marriage, Conquest, Bid) unlock at `level > ADULT_LEVEL`.

### DSL implementation

```dsl
// 1. Events.
@replayable event ChildConceived { parents: [AgentId; 4], family_group: GroupId, tick: Tick }
@replayable event ChildBorn      { parents: [AgentId; 4], child_id: AgentId, settlement: SettlementId, tick: Tick }

// 2. Views.
@lazy
view can_conceive(a: Agent, b: Agent) -> bool {
  a.creature_type == b.creature_type
  && a.alive && b.alive
  && distance(a, b) < CONCEPTION_RANGE
  && a.needs.social > 0.5 && b.needs.social > 0.5
  && (a.spouse_ids.contains(b.id) || b.spouse_ids.contains(a.id))   // must be married
  && now - view::last_conception_tick([a.id, b.id]) > CONCEPTION_COOLDOWN
}

@lazy
view blend_personality(parents: [AgentId; 4]) -> Personality {
  let alive = parents.filter(|p| p != 0 && agent(p).alive);
  let avg = Personality {
    risk:       mean(alive.map(|p| agent(p).personality.risk)),
    social:     mean(alive.map(|p| agent(p).personality.social)),
    ambition:   mean(alive.map(|p| agent(p).personality.ambition)),
    compassion: mean(alive.map(|p| agent(p).personality.compassion)),
    curiosity:  mean(alive.map(|p| agent(p).personality.curiosity)),
  };
  avg + gaussian_noise(mean: 0.0, std: 0.1, stream: rng::child_personality(parents, now))
        .clamped(0.0, 1.0)
}

@lazy
view class_bias_seed(parents: [AgentId; 4]) -> SortedVec<(TagId, f32), 16> {
  merge_top_k(
    parents.filter(|p| p != 0).flat_map(|p| agent(p).behavior_profile),
    k: 16,
    decay: 0.5,    // child doesn't fully inherit, just biased
  )
}

// 3. Cascades — conception + birth (two-phase to honor gestation delay).
physics conception_cascade @phase(event) {
  on TickBoundary where now % CONCEPTION_CHECK_CADENCE == 0 {
    // Probabilistic: per-family, check conception eligibility.
    for family in query::groups(kind: Family) {
      let members = family.members.filter(|m| m.creature_type.can_bear_children);
      if members.len() >= 2 {
        let (a, b) = (members[0], members[1]);      // for MVP, first two
        if view::can_conceive(a, b) {
          if rng::per_agent_bool(a.id, now, CONCEPTION_PROB) {
            emit ChildConceived {
              parents: [a.id, b.id, 0, 0], family_group: family.id, tick: now,
            }
          }
        }
      }
    }
  }
} @terminating_in(1)

physics birth_cascade @phase(event) {
  on ChildConceived{parents, family_group, tick: conception_tick} {
    let birth_tick = conception_tick + GESTATION_TICKS;
    emit ScheduleEvent {
      at_tick: birth_tick,
      event: ChildBorn {
        parents, child_id: next_id(), settlement: family_group.home_settlement, tick: birth_tick,
      }
    }
  }
}

physics child_spawn_cascade @phase(event) {
  on ChildBorn{parents, child_id: cid, settlement: s, tick} {
    let blended = view::blend_personality(parents);
    let class_bias = view::class_bias_seed(parents);
    let cultural = mean(parents.filter(|p| p != 0).map(|p| agent(p).cultural_bias));
    emit SpawnAgent {
      agent_id: cid,
      creature_type: parents[0].creature_type,
      pos: settlement(s).home_pos,
      level: 0,
      personality: blended,
      needs: Needs::newborn(),
      memberships: [Membership { group_id: parents[0].primary_family, role: Member, joined_tick: tick }],
      home_settlement_id: s,
      parents: parents,
      children: [],
      class_bias: class_bias,
      cultural_bias: cultural,
      spouse_ids: [],
      movement_mode: Walk,
    }
    // Seeded relationships — child trusts parents.
    for p in parents.filter(|p| p != 0) {
      emit SeedRelationship { a: cid, b: p, valence: +0.7, familiarity: +0.8, kind: Kin }
      emit AddChild { parent: p, child: cid }
      emit EmotionDelta { agent: p, dim: joy, delta: +0.4 }
    }
    // Settlement-wide chronicle.
    emit ChronicleEntry {
      category: Birth,
      entity_ids: [cid, parents[0], parents[1], 0],
      template_id: Chronicle::CHILD_BORN,
      tick,
    }
  }
}

// 4. Masks — adult-gated actions.
mask PostQuest{type: Marriage, ...}
  when self.level > ADULT_LEVEL && self.creature_type.can_marry && ...

mask PostQuest{type: Conquest, ...}
  when self.level > ADULT_LEVEL && ...

mask Bid{...}
  when self.level > COMMERCE_LEVEL && ...

// Children's Attack is level-gated.
mask Attack(t)
  when self.level > CHILD_COMBAT_LEVEL       // prevents wolf pups from mass-killing
     && view::is_hostile(self, t)
     && distance(self, t) < AGGRO_RANGE

// 5. Leveling — automatic age-based, not XP-gated, for MVP.
@lazy
view level(a: AgentId) -> u32 {
  let age = now - agent(a).spawn_tick;
  (age / LEVEL_UP_TICKS).min(MAX_LEVEL)
}
```

**Grammar fit.** Good. Multi-parent blending via `parents: [AgentId; 4]` handles monogamous (2 entries, 2 zeros) and polygamous (3-4) uniformly. `ScheduleEvent` for gestation delay is a clean pattern; see gap below.

**Gap found.** `ScheduleEvent { at_tick, event }` — emitting an event scheduled for a future tick — is used for gestation but **not in spec §2.2's event grammar**. The spec doesn't describe deferred event emission. Options: (a) add `@deferred(at_tick)` event annotation; (b) allow `emit <Event> { ... } at <tick>` syntax; (c) maintain a cascade that polls each tick for due conceptions (more grammar-compatible, less clean). **Recommend** add `emit <Event> { ... } at <tick_expr>` to §2.4 cascade grammar.

---

## Story H.53 — Emotional reactions to witnessed events

**Narrative outcome.** A murder in the town square triggers grief + fear in witnesses (bystanders within sight-range + socially-linked distant agents via gossip). Grief decays slowly; fear decays faster. Productivity drops for grieving NPCs; a high-grief soldier may refuse combat orders. Indelible memories (high `emotional_impact`) persist past the ring-buffer cap.

### DSL implementation

```dsl
// 1. Events.
@replayable @observation_visible
event Witnessed {
  observer: AgentId,
  subject: AgentId,
  kind: WitnessKind,
  location: vec3,
  entity_ids: [AgentId; 4],
  emotional_impact: f32,
  tick: Tick,
}

enum WitnessKind { FriendDied, Murder, Battle, Theft, Birth, Marriage, Coronation, TradedSuccess, BetrayalWitnessed, OathSworn, LegendAct }

// 2. Views — emotion folds.
@materialized(on_event = [Witnessed, PersonalityDrift])
view emotions(agent: AgentId) -> Emotions {
  initial: Emotions::zero(),
  on Witnessed{observer: a, kind, emotional_impact, tick}
    where a == agent {
    let r = Registry::emotion_response_kernel[kind];
    self.joy       += r.joy       * emotional_impact * recency_decay(now - tick);
    self.anger     += r.anger     * emotional_impact * recency_decay(now - tick);
    self.fear      += r.fear      * emotional_impact * recency_decay(now - tick);
    self.grief     += r.grief     * emotional_impact * recency_decay(now - tick);
    self.pride     += r.pride     * emotional_impact * recency_decay(now - tick);
    self.anxiety   += r.anxiety   * emotional_impact * recency_decay(now - tick);
  }
  decay: self *= Emotions::per_tick_decay per tick,       // grief ×0.995, fear ×0.98, etc.
  clamp: Emotions::bounds(),
}

@lazy
view emotional_impact(observer: Agent, event: EventRef) -> f32 {
  match event.kind {
    FriendDied => relationship(observer, event.subject).valence.max(0.0) * 0.8,
    Murder     => 0.6 * (1.0 + relationship(observer, event.subject).familiarity),
    Battle     => 0.4,
    Theft      => 0.3,
    Marriage   => 0.2 * relationship(observer, event.entity_ids[0]).familiarity,
    _          => 0.2,
  }
}

// 3. Cascades — every first-class event fans out to witnesses.
physics witness_emission_cascade @phase(post) {
  on EntityDied{agent_id: victim, killer: k, cause: c, tick: death_tick} {
    let kind = match c {
      Combat if k.map(|k_a| relationship(victim, k_a).valence < -0.5).unwrap_or(false) => Murder,
      Combat => Battle,
      _      => FriendDied,
    };
    let witnesses = query::nearby_agents(victim.last_pos, radius: WITNESS_RADIUS)
                      .filter(|w| w.alive && w.id != victim && w.id != k.unwrap_or(0));
    for w in witnesses {
      let impact = view::emotional_impact(w, EventRef::EntityDied { victim, killer: k });
      emit Witnessed {
        observer: w.id, subject: victim, kind,
        location: victim.last_pos, entity_ids: [victim, k.unwrap_or(0), 0, 0],
        emotional_impact: impact, tick: death_tick,
      }
    }
    // Gossip propagation: distant allies learn later.
    let distant_allies = query::alive_agents()
                           .filter(|a| relationship(a, victim).familiarity > 0.5)
                           .filter(|a| distance(a.pos, victim.last_pos) > WITNESS_RADIUS);
    for a in distant_allies {
      let delay = ((distance(a.pos, victim.last_pos) - WITNESS_RADIUS) / GOSSIP_SPEED) as Tick;
      emit ScheduleEvent {
        at_tick: death_tick + delay,
        event: Witnessed {
          observer: a.id, subject: victim, kind: FriendDied,
          location: victim.last_pos, entity_ids: [victim, 0, 0, 0],
          emotional_impact: view::emotional_impact(a, EventRef::EntityDied { victim, killer: k }) * 0.5,
          tick: death_tick + delay,
        }
      }
    }
  }
}

// Apply the same pattern for MarriageFormed, QuestCompleted{Conquest}, BehaviorThresholdCrossed, etc.
// Each @observation_visible event gets a witness_emission cascade per @phase(post).

// 4. Memory ring management — indelible slots for high-impact events.
physics memory_record_cascade @phase(post) {
  on Witnessed{observer: o, kind, emotional_impact: impact, ...} {
    let indelible = impact > 0.8;
    emit RecordMemory {
      observer: o,
      payload: FactPayload::WitnessPayload {
        kind, subject: e.subject, location: e.location,
        entity_ids: e.entity_ids, tick: e.tick,
      },
      source: Witnessed,
      confidence: 1.0,
      indelible: indelible,
    }
  }
}

// 5. Reward — emotions feed into shaping.
reward {
  -0.5 on event(Witnessed{observer: self, kind: FriendDied})
  -0.3 on event(Witnessed{observer: self, kind: Murder})
  +0.2 on event(Witnessed{observer: self, kind: Marriage, entity_ids.contains(self.spouse_id)})
  -0.01 * self.emotions.anxiety per tick                     // continuous anxiety penalty
  +0.005 * self.emotions.joy per tick
}

// 6. Observation — emotions + raw memory slots.
observation {
  block self.emotions {
    from view::emotions(self.id) as vec6       // joy, anger, fear, grief, pride, anxiety
  }
  slots recent_memory_events[K=12] from latest(self.memory_events) {
    atom type      = one_hot(e.kind, EVENT_TYPE_VOCAB)
    atom age       = log1p(now - e.tick) / 10.0
    atom target_in_nearby = index_in_slots(e.entity_ids[0], nearby_actors)
    atom impact    = e.emotional_impact
    atom source    = one_hot(e.source, 6)
    atom indelible = if e.indelible { 1.0 } else { 0.0 }
  }
  atom n_indelible_memories = count(self.memory_events.filter(|e| e.indelible))
}
```

**Grammar fit.** Very clean. `@materialized` emotions view folds events with a decay parameter; reward grammar handles both event-triggered bumps (`on event(...)`) and continuous-state drags (`per tick`).

**Gap found.** The `ScheduleEvent { at_tick, event }` pattern shows up again (gossip delay). Ratified under story H.52's gap recommendation.

**Gap found.** `indelible: bool` on `RecordMemory` cascade — the spec mentions "up to 5 indelible slots reserved for `emotional_impact > 0.8`" in `state.md`, but the cascade-level grammar for flagging an emit as indelible isn't shown in `spec.md` §2.4. **Recommend** documenting that `RecordMemory` accepts an `indelible: bool` parameter that routes to the indelible portion of the ring buffer.

---

## Story H.54 — Faction alliance via bidding

**Narrative outcome.** A weaker faction posts a Diplomatic auction with `AllianceTerms{obligations: MutualDefense(vs=rival), duration: 5000}`. Multiple neighbors bid mixed payments (Gold + Service + Reputation). Seller picks one (MutualAgreement) or forms a coalition (Coalition{min_parties=3}). When the rival attacks, allies auto-receive a Defend quest; breaking alliance costs reputation.

### DSL implementation

```dsl
// 1. New auction item variant.
enum AuctionItem {
  Commodity { commodity: Commodity, quantity: f32 },
  Item { item_id: ItemId },
  ServicePledge { scope: ServiceScope, duration: Tick },
  AllianceTerms { obligations: AllianceObligation, duration: Tick },
  CharterTerms { terms: CharterTerms },
}

enum AllianceObligation {
  MutualDefense { vs: GroupId },
  TradeExclusive { against: GroupId },
  VassalCommitment { duration: Tick },
  ProductionShare { commodity: Commodity, rate: f32 },
}

// 2. Payment extended per §9 D10 (material + immaterial Combination).
enum Payment {
  None,
  Gold(f32),
  Item(ItemId),
  Commodity { commodity: Commodity, quantity: f32 },
  Service { scope: ServiceScope, duration: Tick },
  Promise(Box<Reward>, Tick),
  Reputation(f32),                               // stakes reputation on completion
  Combination { material: Box<Payment>, immaterial: Box<Payment> },
}

enum Visibility {
  Public,
  Groups(SortedVec<GroupId, 16>),
  ReputationAtLeast(f32),
}

// 3. Events.
@replayable event AuctionPosted  { auction_id: AuctionId, kind: AuctionKind, item: AuctionItem, seller: AuctionParty, visibility: Visibility, deadline: Tick, resolution: Resolution, tick: Tick }
@replayable event BidPlaced      { auction_id: AuctionId, bidder: AgentId, payment: Payment, tick: Tick }
@replayable event AuctionResolved { auction_id: AuctionId, winners: SortedVec<AgentId, 16>, kind: AuctionKind, tick: Tick }
@replayable event AllianceFormed  { members: SortedVec<GroupId, 16>, obligations: AllianceObligation, duration: Tick, tick: Tick }
@replayable event AllianceBroken  { breaker: GroupId, obligation: AllianceObligation, reason: AllianceBreakReason, tick: Tick }

enum AllianceBreakReason { ExplicitBreak, FailureToHonor, MemberDeath, AllianceExpired }

// 4. Views — subjective valuation per §9 D10.
@lazy
view intrinsic_value(observer: AgentId, payment: Payment) -> f32 {
  match payment {
    Payment::Gold(g)        => g * observer.gold_valuation,
    Payment::Commodity{c,q} => q * view::commodity_price(observer.home_settlement_id, c),
    Payment::Service{scope, duration} => scope.expected_value(observer) * duration as f32 * observer.time_discount,
    Payment::Reputation(r)  => r * observer.reputation_weight,
    Payment::Combination{material, immaterial} =>
      view::intrinsic_value(observer, *material)
      + view::intrinsic_value(observer, *immaterial) * observer.immaterial_preference,
    Payment::Promise(r, when) => reward_intrinsic_value(observer, *r)
                                 * recency_decay(when - now).max(0.0),
    Payment::None           => 0.0,
    Payment::Item(i)        => item_value(observer, i),
  }
}

@lazy
view believed_intrinsic_value(self: AgentId, other: AgentId, payment: Payment) -> f32 {
  // Theory-of-mind: how would the OTHER party value this payment?
  let proxy = self.known_actors.find(other).map(|r| r.perceived_valuation).unwrap_or(default_valuation());
  match payment {
    Payment::Gold(g) => g * proxy.gold_weight,
    _ => view::intrinsic_value(other, payment)         // fallback — assume common valuation
  }
}

@lazy
view alliance_active(g_a: GroupId, g_b: GroupId) -> Option<AllianceObligation> {
  query::active_alliances()
    .filter(|al| al.members.contains(g_a) && al.members.contains(g_b))
    .map(|al| al.obligations)
    .first()
}

// 5. Masks.
mask PostQuest{type: Diplomacy, ...} when false      // Diplomacy now goes through PostAuction.

mask PostAuction{kind: Diplomatic, item: AllianceTerms{obligations: o, duration: d}, seller: Group(g_self), ...}
  when g_self in self.leader_groups
     && g_self.kind == Faction
     && (view::alliance_active(g_self, _).is_none() || !conflicts_with(o, _))
     && d <= MAX_ALLIANCE_DURATION

mask Bid{auction_id: a, payment: p, conditions: _}
  where auction(a).kind == Diplomatic
  when auction(a).visibility.includes(self)
     && self.leader_groups.len() > 0
     && view::intrinsic_value(self, p) > 0
     // Can't ally with a current enemy.
     && self.lead_group.standings[auction(a).seller.group()] != Hostile

// 6. Cascade — resolve alliance auction, wire up obligations.
physics alliance_auction_resolution @phase(event) {
  on AuctionResolved{auction_id: aid, winners, kind: Diplomatic, tick} {
    let a = auction(aid);
    let seller_g = a.seller.group();
    let winner_groups: [GroupId] = winners.iter().map(|w| agent(w).primary_faction).collect();
    let all_members = [seller_g] ++ winner_groups;
    // Update standings to Allied-tier for every pair.
    for i in 0..all_members.len() {
      for j in (i+1)..all_members.len() {
        emit SetStandingEvent { setter: all_members[i], target: all_members[j], kind: Friendly }
        emit SetStandingEvent { setter: all_members[j], target: all_members[i], kind: Friendly }
      }
    }
    emit AllianceFormed {
      members: SortedVec::from(all_members),
      obligations: a.item.unwrap_alliance_terms().obligations,
      duration: a.item.unwrap_alliance_terms().duration,
      tick,
    }
    // Schedule the expiration.
    emit ScheduleEvent {
      at_tick: tick + a.item.unwrap_alliance_terms().duration,
      event: AllianceExpired { members: all_members, tick: 0 },
    }
    emit ChronicleEntry {
      category: AllianceFormed, entity_ids: all_members.pad(4),
      template_id: Chronicle::ALLIANCE_SIGNED, tick,
    }
  }
}

// 7. Cascade — MutualDefense auto-kick-in: when a war is declared against an ally, ally gets a Defend quest.
physics mutual_defense_kick_in @phase(event) {
  on WarDeclared{aggressor, defender, tick} {
    for ally in query::allies_with_obligation(defender, AllianceObligation::MutualDefense { vs: aggressor }) {
      let defend_qid = next_id();
      emit QuestPosted {
        quest_id: defend_qid, type: Defend,
        poster: ally.leader_id, party: Group(ally),
        target: Location(defender.home_pos),
        reward: Reciprocal,
        deadline: tick + DEFAULT_DEFEND_WINDOW,
      }
      emit ChronicleEntry {
        category: AllianceActivated, entity_ids: [ally, defender, aggressor, 0],
        template_id: Chronicle::DEFEND_ALLY, tick,
      }
    }
  }
}

// 8. Cascade — failure to honor breaks alliance after N retries.
physics alliance_failure_cascade @phase(event) {
  on QuestExpired{quest_id: qid} where quest(qid).type == Defend {
    let defender = quest(qid).target.unwrap_location_owner();
    let ally = agent(quest(qid).poster).primary_faction;
    let obligation = AllianceObligation::MutualDefense { vs: defender.recent_aggressor };
    if query::alliance_active(ally, defender).is_some_and(|o| o == obligation) {
      emit RecordMemory {
        observer: defender.leader_id,
        payload: FactPayload::AllyFailedToDefend { ally, obligation, tick: now },
        source: Witnessed, confidence: 1.0,
      }
      // On first failure, just remember; after 2 failures, break.
      if count_failures(ally, obligation) >= 2 {
        emit AllianceBroken {
          breaker: ally, obligation,
          reason: FailureToHonor, tick: now,
        }
        emit SetStandingEvent { setter: defender, target: ally, kind: Tense }
        emit SetStandingEvent { setter: ally, target: defender, kind: Tense }
        emit ReputationDelta { agent: ally.leader_id, delta: -ALLIANCE_BREAK_REP_COST }
      }
    }
  }
}

// 9. Reward — policy learns to honor alliances (reputation + standing benefits).
reward {
  +5.0 on event(AllianceFormed{members.contains(self.lead_group.id)})
  +2.0 on event(QuestCompleted{type: Defend, party.includes(self)})
  -3.0 on event(AllianceBroken{breaker: self.lead_group.id, reason: FailureToHonor})
  -1.0 on event(AllianceBroken{breaker: self.lead_group.id, reason: ExplicitBreak})
}

// 10. Observation — active_auctions slot alongside active_quests.
observation {
  slots active_auctions[K=4] from self.visible_auctions sort_by deadline_asc {
    atom kind_one_hot           = one_hot(e.kind, 6)
    atom seller_group_id_hash   = hash(e.seller.group_id_if_group) / u32::MAX as f32
    atom my_believed_seller_val = view::believed_intrinsic_value(self, e.seller.leader_id, best_payment_for(self, e)) / INTRINSIC_MAX
    atom deadline_ticks_log     = log1p(e.deadline - now) / 10.0
    atom n_current_bids         = log1p(e.current_bids_count) / 10.0
    atom my_eligible            = mask_passes(Bid{auction_id: e.id, ...}, self)
  }
}
```

**Grammar fit.** `PostAuction` is needed as a macro head distinct from `PostQuest` — or `PostQuest{kind: Diplomacy}` could alias to posting an auction. Spec §3.3 says "`PostAuction` aliases `PostQuest{kind: Diplomacy | Charter | Service}`" — so the implementation uses the alias. `Resolution::Coalition{min_parties}` enables multi-party alliances naturally. Subjective valuation via `intrinsic_value` + `believed_intrinsic_value` handles heterogeneous payment scoring.

**Gap found.** `ScheduleEvent` (appears a third time here for alliance expiration) — see H.52 recommendation.

**Gap found.** `active_auctions[K=4]` observation slot uses sort-key `deadline_asc` which is not in spec §3.1's list of slot sort forms (it shows `sort_by distance(self, _)`, `sort_by relevance(self, _)`). **Recommend** generalizing: `sort_by <scalar_expr> [asc|desc]`.

---

## Consolidated grammar-gap report

Derived from implementing 10 stories against `spec.md`'s grammar. Each gap has: severity (blocking / important / cosmetic), proposed resolution, which story surfaced it.

| # | Gap | Severity | Story | Proposed resolution |
|---|-----|----------|-------|---------------------|
| G1 | `match <expr> { ... }` inside mask predicates (for governance-conditional Claim eligibility) | important | H.46 | Clarify §2.5: `match` over enum fields is permitted in masks, alongside `contains` / `in` / etc. |
| G2 | Derivation of `eligible_acceptors` from `QuestTarget + PartyScope` | important | H.47 | §3.3 table: `PartyScope::Individual(p) + QuestTarget::Agent(t) → eligible_acceptors = {t}`; `PartyScope::Group(g) → eligible_acceptors = members_of(g)`; etc. |
| G3 | Creature-type-conditional `reward` blocks (`reward when self.creature_type == Wolf { ... }`) | important | H.48 | §3.4: allow multiple conditional reward blocks; components additive across matching predicates. |
| G4 | `BreakOath` micro primitive missing from `micro_kind` enum | blocking | H.49 | §3.2: append `BreakOath` to `micro_kind` (schema-hash bump, append-only). |
| G5 | Materialized-view change events (for "legend ascended" cosmetic chronicle) | cosmetic | H.50 | Either `@emits_on_threshold(field, value)` annotation or document that cascade polling at `@phase(post)` achieves the same. |
| G6 | Continuous per-tick reward terms (`-0.01 * self.emotions.anxiety per tick`) | important | H.51, H.53 | §3.4: add `per_tick <expr> × <coeff>` alongside `delta(...)` and `+X on event(...)`. |
| G7 | Deferred event emission (`emit <Event> at <tick>` for gestation, gossip propagation, alliance expiration) | blocking | H.52, H.53, H.54 | §2.4: add `emit <Event> { ... } at <tick_expr>` to cascade grammar; compiler emits a pending-event priority queue in the runtime. |
| G8 | `indelible: bool` on `RecordMemory` cascade | cosmetic | H.53 | Document in `state.md` under `MemoryEvent` / RecordMemory cascade output. |
| G9 | Generalized slot sort keys (`sort_by deadline_asc`) | cosmetic | H.54 | §3.1: `sort_by <scalar_expr> [asc|desc]` where `<scalar_expr>` is any view/field producing a scalar. |

### Non-gaps (things the grammar already handles well)

- **Universal action vocabulary.** Every story routes through `PostQuest`, `AcceptQuest`, `Bid`, `Announce`, or a micro primitive — none needed a new macro head (except `PostAuction` alias already settled).
- **Multi-group membership + standings cascade.** Defection, succession, alliance all use per-pair `Relationship` + `Group.standings` cleanly.
- **`SetStanding` macro** collapses DeclareWar / MarkHostile / ProposeAlliance / Vassalize into one primitive with emergent "alliance" semantics.
- **Subjective valuation** (`intrinsic_value`, `believed_intrinsic_value`) handles cross-type payment comparison in diplomatic auctions without hand-coded valuation tables.
- **Materialized views with event-fold + decay** produce all the continuous state that looks like "emotions," "war_exhaustion," "reputation," "treasury_trend" — no stored-aggregate drift code.
- **Reward shaping by aggregate views** (`-0.05 * war_exhaustion(g) per tick` — pending G6) is the cleanest way to incentivize behavior that scales with a continuous state signal.

### Story-level verdicts

| Story | Implementable today (post-fixes)? | Blocking gaps |
|-------|-----------------------------------|---------------|
| H.45  | Yes | — |
| H.46  | Yes | G1 (important, not blocking) |
| H.47  | Yes | — |
| H.48  | Yes | G3 (important) |
| H.49  | **Blocked** on G4 (BreakOath). | G4 |
| H.50  | Yes | — |
| H.51  | Yes | G6 (important) |
| H.52  | **Blocked** on G7 (deferred emit). | G7 |
| H.53  | **Blocked** on G7. | G7 |
| H.54  | **Blocked** on G7. | G7 |

**Critical path**: resolve G4 (BreakOath micro primitive) and G7 (deferred event emission). The other seven gaps are clarifications / minor additions that don't change architectural shape.

---

## Exercise conclusions

1. **The DSL surface holds up.** 9 / 10 Batch H stories are implementable today with minor clarifications. The one true blocker (G7, deferred events) is mechanical rather than architectural — a runtime priority queue is straightforward; only the grammar needs extending.
2. **`SetStanding` is load-bearing.** It showed up in H.45, H.49, H.54 — war, defection, alliance. Replaces what would have been 5+ special-case macros.
3. **Subjective valuation + theory-of-mind views** (`intrinsic_value`, `believed_intrinsic_value`) — the H.54 auction bidding shows these do real work. They were settled in §9 D10 partly on their own merit; this exercise confirms they're not speculative.
4. **Materialized views + `per tick` rewards** — H.51 and H.53 show the pattern: a continuous state aggregate (war_exhaustion, anxiety) drives a continuous reward gradient that produces the player-facing behavior (suing for peace, seeking calm) without any hand-coded decision rule. This pattern appeared 4+ times and merits first-class reward-grammar support (G6).
5. **Cascade-centric design** keeps the imperative logic localized. No story needed a cross-cutting "system" — just on-event reactions that emit further events or mutate views.
