# DSL Design Decisions Log

Standing log of design decisions. For the at-a-glance summary and schema impact, see `spec.md` §9 — this doc carries the rationale and reversal criteria.

Each entry:
- **Decision** — what we chose.
- **Alternatives considered** — what we rejected, and why.
- **Why this, not that** — the load-bearing argument.
- **Reversal criteria** — what new evidence would make us revisit.

## Index

Decisions are numbered to match `spec.md` §9. New decisions append to the end.

### Action / quest mechanics

- [D1 — Auction state machine](#d1-auction-state-machine)
- [D2 — Macro head firing rate](#d2-macro-head-firing-rate)
- [D3 — Macro credit assignment](#d3-macro-credit-assignment)
- [D4 — Quest discovery push/pull hybrid](#d4-quest-discovery-pushpull-hybrid)
- [D5 — Slot K tuning](#d5-slot-k-tuning)
- [D6 — Cross-entity mask index design](#d6-cross-entity-mask-index-design)
- [D7 — Concurrent quest membership](#d7-concurrent-quest-membership)
- [D8 — Reward delivery on long quests](#d8-reward-delivery-on-long-quests)
- [D9 — Cancellation / amendment](#d9-cancellation--amendment)
- [D10 — Bid currency parity](#d10-bid-currency-parity)
- [D15 — Nested quest cancellation](#d15-nested-quest-cancellation)
- [D17 — Polygamy / cross-species / multi-parent](#d17-polygamy--cross-species--multi-parent)
- [D18 — Mercenary / service payment direction](#d18-mercenary--service-payment-direction)
- [D19 — Alliance obligation enforcement](#d19-alliance-obligation-enforcement)
- [D20 — Group-level vs agent-level invites](#d20-group-level-vs-agent-level-invites)

### Runtime / infrastructure

- [D11 — LlmBackend distillation pipeline](#d11-llmbackend-distillation-pipeline)
- [D12 — Per-agent RNG streams](#d12-per-agent-rng-streams)
- [D13 — Materialized-view restoration on load](#d13-materialized-view-restoration-on-load)
- [D14 — Event log storage compression](#d14-event-log-storage-compression)
- [D21 — Chronicle prose side-channel lifecycle](#d21-chronicle-prose-side-channel-lifecycle)
- [D22 — Probe default episode count](#d22-probe-default-episode-count)
- [D23 — Off-policy vs on-policy training dispatch](#d23-off-policy-vs-on-policy-training-dispatch)
- [D24 — Utility backend retirement milestone](#d24-utility-backend-retirement-milestone)
- [D25 — 3D spatial hash structure](#d25-3d-spatial-hash-structure)
- [D26 — Overhear confidence decay](#d26-overhear-confidence-decay)
- [D28 — believed_knowledge decay rate](#d28-believed_knowledge-decay-rate)

### Schema / memory

- [D27 — Document trust_score authoring](#d27-document-trust_score-authoring)
- [D29 — FactRef ownership after memory eviction](#d29-factref-ownership-after-memory-eviction)

### Modding

- [D16 — Mod event-handler conflict resolution](#d16-mod-event-handler-conflict-resolution)

---

## Format note

This doc is currently a stub pointing into `spec.md` §9. Per-decision rationale will be filled in as implementation forces revisits or as new decisions land. The intent is to capture:

1. The *why* (so future-self can judge whether reversal is warranted).
2. The rejected alternatives (so we don't re-litigate).
3. What would force a revisit (so drift is detectable).

When adding a new decision, create a new top-level entry `DN` (continuing numbering from 29), add an index row above, and reference it from `spec.md` §9.

---

## Detailed entries

*(Per-decision sections will be filled in on demand. For now, `spec.md` §9 carries the canonical decision text and one-line rationale.)*

### D1 — Auction state machine

**Decision.** `Resolution` enum = `{HighestBid, FirstAcceptable, MutualAgreement, Coalition{min_parties: u8}, Majority}`. `PostAuction` aliases `PostQuest{kind: Diplomacy | Charter | Service}`; no separate macro head. Cadence is per-world config.

**Why this, not that.** `Coalition` is needed for story §H.54 (multi-party diplomatic pacts); `Majority` for story §H.46 (contested succession). Aliasing vs introducing a new head keeps the action vocabulary stable at 7 macros.

**Reversal criteria.** If implementation reveals that `Coalition` / `Majority` need fundamentally different bidding semantics (not just resolution rules), split into a separate macro head.

---

*(Remaining decisions D2–D29 to be filled in as implementation progresses; see `spec.md` §9 for current statements.)*
