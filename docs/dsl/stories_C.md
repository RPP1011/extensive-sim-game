# Category C: Observation Schema (ML Engineer) — Story Analysis

Stories 13–18 from `user_stories_proposed.md`. Each verdict is relative to the
current state of `proposal_policy_schema.md` (§2.1 observation, §3 DSL surface,
§4 versioning). Where the proposal as written conflicts with user intent on
story 15 (no backwards compat), this document overrides the proposal.

---

### Story 13: Inspect schema as JSON

**Verdict:** PARTIAL

**User's framing note:** "Essential." ML pipelines need a machine-readable
description of the observation tensor to build correctly-shaped models.

**How the DSL supports this:**
- `proposal_policy_schema.md` §3 lists "JSON schema dump for ML pipeline tooling"
  as a compiler output, alongside "Packed struct with named field offsets
  (debug tools decode bytes back to fields)." That bullet is the hook story 13
  depends on.
- §2.1 decomposes the ~1655-float tensor into nameable groups (self atomic,
  self contextual, group memberships, spatial slots, non-spatial slots, context
  blocks). Those groupings are what the JSON schema must expose.
- §3's DSL pseudocode gives every field a name (`self.hp_pct`,
  `self.psychological`, `nearby_actors[i].relationship_valence`, ...). The
  named form maps 1:1 onto JSON schema entries.

**Implementation walkthrough:**
A workable JSON schema shape — not specified in the proposal but consistent
with the pseudocode — is:

```json
{
  "schema_hash": "sha256:...",
  "total_floats": 1655,
  "groups": [
    {
      "name": "self.atomic",
      "offset": 0,
      "size": 55,
      "fields": [
        { "name": "hp_pct",        "offset": 0,  "dtype": "f32",
          "norm": { "kind": "identity", "range": [0,1] } },
        { "name": "max_hp_log",    "offset": 1,  "dtype": "f32",
          "norm": { "kind": "log1p", "scale": 1.0 } },
        ...
      ]
    },
    {
      "name": "spatial.nearby_actors",
      "offset": 295,
      "size": 360,
      "slot_count": 12,
      "slot_size": 30,
      "slot_fields": [
        { "name": "relative_pos_x", "offset": 0, "dtype": "f32",
          "norm": { "kind": "scale", "denom": 50 } },
        { "name": "creature_type_one_hot", "offset": 2, "dtype": "f32[8]",
          "norm": { "kind": "one_hot", "vocab_size": 8 } },
        ...
      ]
    }
  ],
  "one_hot_vocabularies": {
    "creature_type":  ["Human","Elf","Dwarf","Wolf","Goblin","Dragon",...],
    "group_kind":     ["Faction","Family","Guild","Religion","Party","Pack",
                       "Settlement","Other"],
    "relationship_kind": ["Spouse","Kin","Mentor","Apprentice","Friend",
                          "Rival","SwornEnemy","Stranger"],
    "event_type":     [...]
  },
  "slot_repeats": {
    "nearby_actors": { "k": 12, "sort_key": "distance" },
    "known_actors":  { "k": 10, "sort_key": "relevance" }
  },
  "action_vocabulary": {
    "macro_kind": ["NoOp","PostQuest","AcceptQuest","Bid"],
    "micro_kind": ["Hold","MoveToward","Flee","Attack","Cast","UseItem",
                   "Harvest","Eat","Drink","Rest","PlaceTile","PlaceVoxel",
                   "HarvestVoxel","Converse","ShareStory"]
  }
}
```

Tools consume this by:
1. Loading `schema.json` at PyTorch model build time.
2. Declaring input shape `[batch, total_floats]` and stashing `schema_hash`
   into the checkpoint's metadata.
3. Using `groups[i].offset/size` to slice the tensor for grouped encoders
   (e.g., one sub-MLP for self atomic, a transformer over spatial slots).
4. Exposing `one_hot_vocabularies` to the replay-buffer decoder (story 17).

The stability contract: once `schema_hash` is emitted, it is a
**content-addressed fingerprint** of the observation layout. Changing *any*
field's offset, dtype, or normalization changes the hash. Tools must refuse
to load a checkpoint whose hash disagrees with the current DSL build
(story 15).

**Gaps / open questions:**
- The proposal does not specify where the normalization constants live. The
  pseudocode hints at `log1p`, `/50`, `/100` scaling in individual atoms, but
  never commits to a canonical set of normalization kinds. A `norm` taxonomy
  must be fixed early (identity / log1p / scale / clamp / one_hot / bitset)
  because all three of story 13, 14, and 17 consume it.
- Action space, mask shape, and reward declaration are also part of the
  training contract. §6 open question 17 already flags this: "Should the
  schema hash cover the action vocabulary?" — if yes, the JSON schema has to
  emit action + reward sections too. Recommend yes; checkpoint compatibility
  is all-or-nothing.
- The DSL compiler that produces this JSON does not yet exist; §3 only
  sketches the surface. The schema emitter is the first concrete output the
  compiler owes the ML pipeline.

**Related stories:** 14 (append-only growth must preserve the emitted shape),
15 (hash is the mismatch detector), 17 (decoder CLI reads this file).

---

### Story 14: Add a new observation feature

**Verdict:** SUPPORTED

**User's framing note:** "Essential." Day-to-day ML work. The engineer edits
the DSL to add `self.fame_log`, rebuilds, and the model gets the new signal.

**How the DSL supports this:**
- §4.3 "Append-only schema growth": "when adding features, append; never
  reorder existing slots. Removing a feature is a breaking change requiring
  model retraining."
- `self.fame_log` is already in §2.1.2's Self contextual list — the block
  exists. Adding a brand-new scalar outside the existing blocks is the
  interesting case.
- §3 pseudocode shows how an atom is declared:
  ```
  self.fame_log = log1p(view::fame(self))
  ```
  Or inside a block:
  ```
  block self.social_standing {
    from view::fame(self)       as f32 via log1p
    from view::reputation(self) as f32
  }
  ```

**Implementation walkthrough:**

The append-only workflow, assuming `view::fame` already exists (if not, it
gets declared in a `view` block and its source events/fields must already be
in `state_npc.md`):

1. **Edit DSL.** Add `atom fame_log = log1p(view::fame(self))` to the relevant
   observation block. Source references must resolve to declared entity
   fields, views, or events — §3 explicitly states this validation rule.
2. **Rebuild.** The DSL compiler:
   - Re-orders feature offsets so the new atom is appended after all existing
     ones in the same block. "Appended" means *within the block*; the
     aggregate tensor has per-block contiguous regions, so a new atom in
     `self.contextual` lives at the end of that region, not at position 1655.
   - Recomputes the schema hash. Any append changes the hash.
   - Re-emits `schema.json` with the new field and its `offset/size/norm`.
   - Regenerates the Rust packing kernel (new line that writes the value).
   - Regenerates the GPU packing kernel if the block is `@gpu_kernel`.
3. **Bump the checkpoint.** §4.4 "CI guard — observation schema changes that
   touch existing slots fail CI unless accompanied by a model checkpoint
   bump." An append-only change must still bump the hash, but does not edit
   any existing offset. CI check: old offsets + types unchanged under the new
   schema's field list.
4. **Train.** The ML engineer re-initializes the input layer (or widens it to
   match new total_floats) and trains. The append-only rule means old
   trajectories from pre-change runs can be re-padded with zeros at the new
   tail positions *if and only if the engineer opts in* — but per story 15,
   we are not building automatic padded-zero migration (see below).

**What changes for the model:**
- `total_floats` increases by 1 (or however many floats the new feature
  contributes).
- The input layer widens. For a linear input projection this is a trivial
  re-init; for a grouped encoder where the new atom joins an existing block,
  the block's sub-MLP widens.
- All downstream shapes are stable because the new feature is appended.

**Gaps / open questions:**
- Per-slot append is trickier than per-atom append. If we add a field to
  `nearby_actors` slots (currently 30 floats × 12 slots), every slot grows.
  That's still append-only (the new slot field goes at offset 30 within the
  slot), but the total footprint change is `K × new_bytes`, not
  `1 × new_bytes`. Compiler must handle slot-internal appends identically.
- There is no story for **removing** a feature. §4.3 says removal is a
  breaking change. We should explicitly fail compilation on an attempted
  removal unless a `@deprecated` path was declared (and even then, story 15's
  no-backcompat stance means we just retrain from scratch).
- View/event references introduce an indirection: adding `self.fame_log`
  requires `view::fame` to be backed by a concrete `state_npc.md` field
  (currently `reputation_summary` is listed there but `fame_log` is not
  explicitly — it needs a source field or a view over `legendary_deeds`
  events).

**Related stories:** 13 (new field appears in JSON schema), 15 (the schema
hash bump is what makes the old checkpoint refuse to load), 19 (bootstrap
trajectories recorded before the append cannot be replayed verbatim against
the new model).

---

### Story 15: Schema versioning

**Verdict:** GAP (the proposal's answer is wrong for this project)

**User's framing note:** **Re-scoped.** The proposal's §4 prescribes three
versioning tools (schema hash, `@since` annotations, padded-zero migration
tables). The user explicitly rejects two of those: "We do not want to have v1
and v2 and v3 in the same codebase. Git can be used to store versions, it is
nothing but waste to support backwards compatibility in a solo project like
this."

The correct design is **fail loud; no coexisting versions; no padded-zero
migration.** The schema hash stays — as a fingerprint, not as a migration
hinge.

**How the DSL supports this (after re-scoping):**

Drop from the proposal:
- `@since(v=1.1)` field annotations — implies two schemas coexist. Delete.
- "Migration tables" in §4.2 — implies automatic padded-zero fill. Delete.
- "rejected-with-explanation when the model would need fields that no longer
  exist" — only half wrong: the rejection is right, the prose implying the
  other branch (zero-pad when fields were *added*) is wrong. Replace with
  unconditional rejection.

Keep from the proposal:
- **Schema hash.** SHA256 over the canonical (sorted/normalized) observation
  layout + normalization constants + action vocabulary + reward declaration.
  Burned into every checkpoint at save time.
- **Append-only growth.** Same mechanical rule as story 14.
- **CI guard.** Breaking changes (reorder, remove, type change, norm change)
  must not merge without a corresponding checkpoint bump.

**Implementation walkthrough:**

The error UX when a mismatched checkpoint meets a rebuilt DSL:

```
error: policy checkpoint schema mismatch
  checkpoint: generated/npc_v3.bin (trained 2026-04-10)
  checkpoint schema_hash: sha256:a1b2c3...7890
  current DSL schema_hash: sha256:e4f5g6...2345
  diff:
    + appended: self.fame_log  (offset 1655, size 1, norm log1p)
    + appended: nearby_actors[].has_quest_conflict
                (slot-internal offset 30, size 1, norm identity)
  action: retrain from current DSL, or git-checkout the commit whose
          schema_hash matches the checkpoint.
```

Key UX properties:
1. **Hard fail on mismatch.** No auto-padding, no opt-in flag, no warning
   mode. The loader refuses and exits with a nonzero code.
2. **Explain the drift.** The error prints a diff of what changed (requires
   keeping the checkpoint's schema JSON alongside its weights, which is
   cheap — a few KB per checkpoint).
3. **Suggest git.** The canonical remediation is either retrain or checkout
   an older commit. Git is the version control; the codebase is not.
4. **Single live schema.** The compiler only emits one schema at a time.
   There is no "v1" or "v2" path in the Rust code. If two developers (or
   two branches) disagree, their checkpoints are mutually incompatible and
   that is fine — they live on different branches.

**Is the schema hash still valuable as a fingerprint?** Yes, unambiguously.
Without it we have no way to detect silent corruption. A checkpoint without a
hash cannot be safely loaded — if someone trains on schema A, commits schema
B locally without noticing, and loads the checkpoint, the tensor layout is
garbage but the model happily produces actions. The hash turns this
silent-corruption case into the loud-error case.

The hash also covers:
- **Action vocabulary.** Per §6 open question 17 — yes, it should. Adding a
  new `ActionKind` enum variant changes the output head's shape; the model is
  incompatible even if the observation didn't move.
- **Reward DSL (optional).** Reward changes don't break the trained model
  mechanically, but they break dataset-level comparability (replay buffer
  rewards were computed under a different rule). Recommend hashing reward
  too, but exposing it as a separate `reward_hash` so observation-only
  compatibility checks can ignore reward drift.

**What FAIL LOUD prevents that migration tables would permit:**
- A field gets renamed. Under padded-zero migration the old slot reads as
  "this used to be the value, now it's zero" and the model wastes capacity
  on a dead feature. Under fail-loud, we retrain with the renamed field
  present from the start.
- A one-hot vocabulary grows a category. Padded-zero would leave the new
  category permanently unselectable in old checkpoints. Fail-loud forces us
  to notice.
- A normalization constant changes (someone tightens the clamp range). Under
  migration the model's learned weights assume the old scale. Fail-loud
  catches it.

**Gaps / open questions:**
- Where does the schema hash physically live in a checkpoint? Proposal does
  not specify. Minimum viable: a `meta` dict in the checkpoint (`.bin` or
  `.json` sibling file) with `schema_hash`, `action_vocab_hash`, commit SHA,
  and training date.
- Do we hash the normalization constants or just the structure? **Must hash
  the constants.** Changing `log1p` to `scale(1/100)` on the same field is a
  silent model-breaker otherwise.
- How does the hash interact with the `one_hot_vocabularies` section of
  story 13's JSON schema? Vocabularies are load-bearing; any reorder or
  insert changes the hash.
- `[OPEN]` from proposal §4: action-vocabulary hashing scope. Recommend
  resolve to "yes, one combined hash; observation + action + reward as
  separate component hashes too for more granular diffing."

**Related stories:** 13 (JSON schema is what the hash is computed over), 14
(every append bumps it), 19 (bootstrap trajectories are now hash-tagged and
rejected against newer schemas).

---

### Story 16: Per-tick training data emission

**Verdict:** SUPPORTED

**User's framing note:** "Great." Standard RL/IL plumbing; the only question
is tuple format and buffer file shape.

**How the DSL supports this:**
- §2.5 names the mechanism: "Logging hooks for (observation, action, reward)
  tuples → training dataset emission" as a reward-compiler output.
- §2.1 observation packing and §2.2 action emission already run every tick;
  adding a log sink is a tee, not new computation.
- `SimState` tick loop (existing `step()` harness in `src/ai/core/simulation.rs`
  per CLAUDE.md) already produces events each tick; the terminal flag is
  derivable from existing `SimEvent::EntityDied` (agent-terminal) or the
  scenario-end event (episode-terminal).

**Implementation walkthrough:**

**Per-tick tuple, per agent:**

```
TrainingTuple {
  tick:          u64
  agent_id:      u32
  episode_id:    u64        // for GAE / return computation
  schema_hash:   [u8; 32]   // fingerprint (story 15)

  observation:   [f32; OBS_DIM]    // packed, exactly what the policy saw
  action:        PackedAction       // multi-head: macro_kind, micro_kind,
                                    // target_slot_idx, pos_delta, magnitude,
                                    // quest_type, party_scope, reward_type,
                                    // payment_type
  mask_snapshot: [bool; NUM_HEADS × MAX_CHOICES]   // optional, expensive

  log_prob:      f32         // log π(a | s), summed across heads
  value_est:     f32         // V(s) from critic (if actor-critic)
  reward:        f32         // from the reward DSL this tick
  terminal_flag: bool        // agent died or episode ended this tick
}
```

Sizing: `1655 × 4 = 6620` observation bytes + ~80 bytes other fields ≈ 6.7 KB
per agent per tick. At 20K agents × 10 Hz sim = 1.3 GB/s raw, unworkable as
flat JSONL. See volume estimates below.

**Buffer file format:**

Recommend **length-prefixed flatbuffers or msgpack frames** written to a
rolling file (`buffer/ep_00042.bin`):

- Fixed header per file: schema_hash, action_vocab_hash, episode_id,
  tick_start, sim version.
- Body: concatenated frames, one per tick, each frame = length-prefixed
  batch of tuples for all live agents that tick.
- Optional float16 compression of the observation tensor for 2× savings;
  keep action/log_prob/reward in float32.

Do **not** use JSONL for the observation (O(7 KB) stringified is absurd).
Reserve JSONL for debug single-agent dumps (story 17).

**Replay buffer integration:**

The file format already matches what a standard PyTorch `IterableDataset`
wants: open file, seek to frame, yield a batch of tuples. Two buffer-shape
options:

- **Episode buffer** (one file per episode): trivial for on-policy
  algorithms (REINFORCE, PPO). Re-read to compute GAE with known
  terminals. What we'd start with.
- **Prioritized replay buffer**: sharded by `episode_id`, with a priority
  index over macro actions (rare events under §2.2 and the "prioritized
  replay buffer with rare-event up-weighting" mitigation in §7). Needed
  later for macro-action credit assignment; defer.

**Per-tick volume estimates:**

| Scale                   | Agents | Tuple bytes | Per tick | Per second (10 Hz) |
|-------------------------|--------|-------------|----------|--------------------|
| Combat scenario         | 20     | 6700        | 134 KB   | 1.3 MB/s           |
| Small world sim         | 500    | 6700        | 3.3 MB   | 33 MB/s            |
| Full world sim target   | 20000  | 6700        | 134 MB   | 1.3 GB/s           |

The full-world case cannot write flat uncompressed tuples in real time. Need:
- Downsample (log every Nth tick, or only on non-NoOp action).
- Compress observations to float16 → 670 MB/s, still a lot.
- Separate the observation from the action; the action stream is O(100
  bytes) per agent per tick. Log actions always, observations only on
  ticks flagged "interesting" (non-NoOp macro, or reward magnitude >
  threshold, or 1-in-N uniform sample).

**Gaps / open questions:**
- "Terminal flag" has two meanings: agent death vs. episode end. Proposal
  doesn't distinguish. Buffer tuple should carry both as separate booleans.
- Mask snapshot is expensive; the number of heads and slots means ~100s of
  bools per tuple. Recommend omit by default, add a `--log-masks` flag for
  debugging why a policy picked what it picked.
- Value estimate only exists if the backend is actor-critic; needs to be
  `Option<f32>` or sentinel NaN.
- Log-prob must be the sum across all heads the action actually used (macro,
  micro, target, continuous) — compiler must emit a helper that returns the
  summed log-prob given the PackedAction.

**Related stories:** 13 (tuple shape is derived from the observation schema),
19 (bootstrap trajectories reuse this exact format), 20 (reward DSL is the
tuple's reward producer).

---

### Story 17: Single agent observation decoder CLI

**Verdict:** SUPPORTED

**User's framing note:** "Great." Debug tool for "what did the model see?"
Essential when the model does something surprising.

**How the DSL supports this:**
- §3 explicitly outputs "Packed struct with named field offsets (debug tools
  decode bytes back to fields)." That is exactly story 17's dependency.
- Story 13's JSON schema contains the offset table — it *is* the named-offset
  table story 17 consumes.
- Story 16's buffer format stores the packed observation tensor alongside
  `tick` and `agent_id`. The decoder just correlates the two.

**Implementation walkthrough:**

CLI shape, consistent with existing `xtask` subcommands:

```
cargo run --bin xtask -- obs decode \
    --buffer generated/buffer/ep_00042.bin \
    --agent 17 \
    --tick 1200 \
    [--schema generated/schema.json]   # default: current DSL schema
    [--format pretty | json | flat]
    [--slice self.atomic]              # restrict to a group
    [--show-zeros]                     # default: suppress zero slot rows
```

Example output (`pretty`):

```
tick=1200 agent=17 episode=42 schema_hash=sha256:e4f5g6...2345

self.atomic (55 floats, offset 0)
  hp_pct              = 0.73
  max_hp_log          = 5.30   (raw max_hp ≈ 200)
  shield_pct          = 0.00
  ...

self.contextual (120 floats, offset 55)
  aspiration.need_vector = [hunger=0.2 safety=0.8 ...]
  ...

spatial.nearby_actors[12, 30 each, offset 295]
  slot 0  exists=1 dist_rank=0
    relative_pos       = (-3.2, +1.1)
    creature_type      = Wolf [one_hot_arg=3]
    hp_pct             = 0.45
    relationship_valence = -0.8  (hostile)
    n_shared_groups    = 0
  slot 1  exists=1 dist_rank=1
    ...
  slot 2  exists=0 (suppressed; rerun with --show-zeros to see)

action taken
  macro_kind = NoOp
  micro_kind = Attack
  target     = spatial.nearby_actors[0]  (the Wolf above)
  log_prob   = -0.34
  reward     = -0.50
```

Mechanics:
1. Load `schema.json` (from flag or by deriving from the DSL's current
   schema_hash).
2. Compare `schema_hash` in the buffer header to the schema's hash. **On
   mismatch, fail loud** (story 15) — print both hashes and the
   git-remediation hint; exit nonzero. Do not attempt to decode.
3. Seek to `tick` frame, find tuple with matching `agent_id`.
4. For each `group` in the schema, slice `observation[offset .. offset+size]`,
   then apply the group's field layout. For slot arrays, iterate `slot_count`
   times at stride `slot_size`.
5. Invert the `norm` spec to recover raw values when possible (log1p →
   expm1, scale → multiply; identity and one_hot pass through).
6. Render one-hot fields as their winning category name using
   `one_hot_vocabularies`.

The tool is ~300 lines of Rust; the payoff is every future "why did the
model do X?" investigation uses it.

**Gaps / open questions:**
- Need a companion `obs diff --tick A --tick B --agent 17` to see what
  changed frame-to-frame. Trivial once the single-tick decoder works.
- Need a companion `obs grep --predicate "hp_pct < 0.2"` to find all
  low-HP snapshots. Story 18's probes idea (below) generalizes this.
- The buffer file format from story 16 must keep tick + agent_id indexable
  without a full scan. Minimum viable: a sidecar `buffer/ep_00042.idx` with
  `(tick, agent_id) → byte_offset`.
- Handling of `known_actors[K]` slots that backref `nearby_actors` (§2.1.5):
  the decoder should resolve `in_nearby_actors_slot_idx` and print the
  linked slot's contents inline.

**Related stories:** 13 (schema source), 15 (hash check), 16 (buffer source),
18 (probes are a batched generalization of this tool).

---

### Story 18: Compare two policy backends + probes on a known dataset

**Verdict:** PARTIAL (comparison: supported; probes: genuine GAP, needs new
DSL surface)

**User's framing note:** "Great." Plus the extension: "I would like a way of
evaluating probes on a known dataset as well." The probes piece is the more
interesting half and is not in the current proposal.

**How the DSL supports this:**

The comparison half is already expressible:
- §2.4 defines `PolicyBackend` as a trait with one method
  (`evaluate_batch(obs, mask) -> ActionBatch`). The trait makes Utility vs
  Neural vs LLM structurally interchangeable.
- Determinism contract (CLAUDE.md: "All simulation randomness flows through
  `SimState.rng_state`") guarantees that a seeded scenario with fixed agent
  spawns produces identical observations across backend swaps. Any decision
  divergence is purely from the policies.
- Story 17's decoder handles the single-agent rendering; an A/B diff is the
  same tool called twice then diffed.

CLI shape (comparison):

```
cargo run --bin xtask -- policy compare \
    --scenario scenarios/basic_4v4.toml \
    --seed 42 \
    --backend-a utility \
    --backend-b neural:generated/npc_v3.bin \
    --output generated/compare_v3.jsonl
```

Per-tick output row: `{tick, agent_id, action_a, action_b, agree: bool,
rationale_a, rationale_b}`. Summary: action-agreement rate, distribution of
disagreements by macro/micro head, cases where one backend moved while the
other held, etc.

**Probes — the real work:**

A **probe** is a small, hand-authored scenario plus an expected behavioral
assertion. Not "did the model pick action X at tick Y" (brittle), but "does
the distribution of decisions on this scenario satisfy the stated
property?" Probes live in-repo as regression fixtures and run every
checkpoint eval.

Proposed probe DSL surface:

```
probe LowHpFlees {
  scenario "probes/low_hp_1v1.toml"
  seed 42
  ticks 200
  backend neural:generated/npc_v3.bin

  // A behavioral claim the policy must satisfy.
  assert {
    // Over all ticks where agent 0 has hp_pct < 0.3,
    // the chosen action is Flee or MoveAway > 80% of the time.
    pr[ action.micro_kind in {Flee, MoveToward_away_from_threat}
      | self.hp_pct < 0.3 ]
      >= 0.80
  }

  tolerance 0.02  // allow 2% absolute slack for stochastic policies
}

probe LeaderPostsQuestsUnderThreat {
  scenario "probes/threatened_settlement.toml"
  backend neural:generated/npc_v3.bin

  assert {
    count[ action.macro_kind == PostQuest
           ∧ action.quest_type == Defense
         | self.is_leader_anywhere == 1
           ∧ settlement.threat_level > 0.6 ]
    >= 1  // at least one defense quest gets posted
  }
}

probe NoSpouseAttacks {
  scenario "probes/random_family_5agents.toml"
  backend neural:generated/npc_v3.bin

  assert {
    pr[ action.micro_kind == Attack
      | action.target.is_spouse == 1 ]
    == 0.0
  }
}
```

**Where probes live:** `probes/` at repo root (sibling to `scenarios/`),
with probe `.probe` files referencing seed scenarios in `probes/scenarios/`.
Keep them small (1–10 agents, short tick budgets) so the full suite runs in
seconds.

**How probes integrate with the training pipeline:**

1. **As regression tests.** Every `generated/npc_v*.bin` checkpoint runs the
   probe suite via `xtask policy probe generated/npc_v3.bin probes/`.
   Checkpoint fails to be "released" if any essential probe fails.
2. **As eval metrics.** The probe pass rate is a scalar per checkpoint,
   tracked alongside episode return. "94.2% probe pass, 0.73 avg return."
3. **As training-time signal.** Probes can gate curriculum steps ("don't
   advance curriculum until AttackSpouse probe passes"). Longer-term, a
   probe's `assert` expression can be converted to an auxiliary loss term
   (soft version of the pass condition).
4. **As a comparison target.** For story 18's core compare: run both
   backends against the same probe suite, report which probes each passes.
   This is the quantitative version of "side-by-side decision diff."

**Probe assertion grammar (minimum viable):**

```
assert_expr := count_expr | prob_expr | mean_expr
count_expr  := "count" "[" filter_expr "]" comparator scalar
prob_expr   := "pr"    "[" action_expr "|" filter_expr "]" comparator prob
mean_expr   := "mean"  "[" scalar_expr "|" filter_expr "]" comparator scalar
filter_expr := boolean over observation fields + action fields + derived
               facts (settlement.*, relationship(self, target).*, ...)
```

All names on the left side of `|` are **action fields** from story 16's
tuple; all names on the right side are **observation fields** resolved via
story 13's schema plus a few cross-reference helpers (target_of_action,
relationship, settlement_of).

**Why this is the right extension of story 18:**

- A raw tick-by-tick diff between two backends is high-noise; 50% disagreement
  doesn't tell you if either is wrong.
- Named probes ("Does low-HP agent flee?" "Does leader post defense
  quests?") are *interpretable* and *stable* — they stay valid as long as
  the observation schema contains `hp_pct` and `micro_kind` has `Flee`.
- They double as sanity tests *and* as comparison oracles. A new backend
  that passes fewer probes is a regression regardless of its episode return.
- The assert syntax is intentionally close to SQL-over-trajectories; no new
  ML framework needed, just a trajectory query engine.

**Gaps / open questions:**
- The probe grammar is new DSL surface; not written yet. It overlaps the
  reward DSL (§2.5) in shape — both filter over (observation, action,
  event) tuples. Consider unifying the two grammars; a probe is a reward
  that asserts instead of accumulates.
- Who authors probes? The ML engineer for behavioral regressions; the game
  designer for "don't attack spouses" style cultural constraints. Need a
  clear convention for which probes are essential (CI-gating) vs. advisory.
- Running probes against stochastic policies requires enough seeded episodes
  to get stable probability estimates. Decide on a default N (e.g., 32
  episodes per probe) and an explicit `seeds [42, 43, ...]` override.
- How do probes interact with schema versioning? A probe references
  `self.hp_pct` by name; if the observation drops `hp_pct` (breaking
  change), the probe fails to compile. Fail-loud, same as everywhere else.
  A probe that references a field that was appended under story 14 works
  automatically once the probe's target schema_hash matches the checkpoint.
- Probes for macro actions require long scenarios (Conquest resolves 2000+
  ticks after PostQuest). Budget accordingly, or split probes by horizon
  (short / medium / long).

**Related stories:** 13 (probes reference schema field names), 14 (a new
observation feature enables new probe predicates), 15 (probes run against a
specific schema_hash; mismatch fails loud), 16 (probes are queries over the
stored replay tuples — probes can be run post-hoc on saved buffers, not just
live), 17 (probes that fail should emit failing ticks to the decoder for
inspection), 19–20 (training on bootstrap trajectories: probes are the
"did bootstrap give us sensible behavior?" check).

---

## Cross-story summary

| Story | Verdict  | Blocking gap                                           |
|-------|----------|--------------------------------------------------------|
| 13    | PARTIAL  | Canonical normalization taxonomy unspecified; compiler does not yet exist |
| 14    | SUPPORTED| Slot-internal append semantics need spelling out       |
| 15    | GAP      | Proposal's migration-table design is wrong for this project; re-scope to hash + fail-loud |
| 16    | SUPPORTED| Volume at 20K agents needs downsampling strategy       |
| 17    | SUPPORTED| Buffer file format needs an index sidecar              |
| 18    | PARTIAL  | Probe DSL and runner are new surface, not in proposal  |

The single biggest piece of work across category C is **story 18's probe
DSL + runner**. It unblocks sanity-checking every checkpoint, serves as the
comparison surface for backend swaps, and doubles as a cultural-constraint
test bed ("no spouse-killing"). Stories 13/17 fall out of the DSL
compiler's JSON emitter (already listed as a compiler output in §3). Story
15's correction (drop `@since`, drop migration tables) is pure
deletion/simplification, not new work. Story 16 is plumbing with known
volume tradeoffs.
