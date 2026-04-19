# Category D — Training Pipeline (Trainer) — Story Analysis

Per-story analysis of stories 19-24 from `user_stories_proposed.md`. Cites `proposal_policy_schema.md` (policy/obs/action/reward/backend schema) and `proposal_universal_mechanics.md` (action vocabulary) as design anchors.

Overall posture: the DSL is **very strong** on observation + mask + single-backend interchange, **moderately strong** on reward declaration, and **under-specified** on the trainer-side plumbing — replay buffer format, curriculum stages, telemetry hooks. The interchangeable `PolicyBackend` trait (§2.4) and declarative `reward { ... }` block (§2.5) give us the right primitives for IL bootstrap and on-policy RL; actor-critic, curriculum, and rare-event upweighting are additive extensions that need to be written into the DSL surface explicitly.

---

### Story 19: Bootstrap from utility trajectories
**Verdict:** SUPPORTED

**User's framing note:** "Great"

**How the DSL supports this:**
The single-backend commitment (README "Settled" bullet 3 / proposal §2.4) already makes the Utility backend a first-class `PolicyBackend` implementation — same observation packing, same masks, same action space. Running the sim with `UtilityBackend` emits the same `(observation, action)` tuples the Neural backend expects to consume. Story 16's per-tick training-data emission (marked "Great") fills in the logging side: `(observation, action, log_prob, reward, terminal_flag)` per agent per tick to a buffer file. Put those together and trajectory bootstrap is a single pipeline — run sim under utility, drain the emitted replay buffer, warm-start the neural weights on it.

**Implementation walkthrough:**
1. Trainer authors a policy block with `backend Utility { rules: "npc_utility.rules" }`. Compiler emits the utility argmax-over-masked-candidates implementation (proposal §2.4 comment: "declarative scoring rules from DSL").
2. Trainer enables per-tick trajectory logging (story 16 hook, likely a `@emit_training_data` annotation on the policy block or a CLI flag).
3. Run N ticks × M agents. Each agent's (observation, mask, action, reward, terminal) tuple is flushed to an append-only file. Utility backend's `log_prob` is degenerate (argmax → 1.0; softmax-with-temperature over utility scores if the trainer wants calibrated behavior-cloning targets) — this is an open detail.
4. Behavior cloning loop (offline, outside the sim): load replay buffer → train neural weights to minimize cross-entropy against the utility's categorical choices per head (macro_kind, micro_kind, each parameter head), MSE against the continuous heads (pos_delta, magnitude). Masks are part of the observation record, so the BC loss excludes masked-off actions automatically.
5. Swap `backend Utility` → `backend Neural { weights: "npc_v0_bc.bin", h_dim: 256 }` in the policy block. Same DSL file, different backend line. Story 23 handles the hot-swap.

Because observation packing, masks, action space, and reward are declared once in the DSL and consumed by both backends, there is no schema drift risk between bootstrap and training — the schema hash is computed once per policy block.

**Gaps / open questions:**
- Utility `log_prob` semantics. BC is cleanest when utility emits calibrated probabilities (softmax over utility scores with a temperature). Needs specification — is `softmax_temperature` a property of the utility backend or a per-policy annotation? The DSL should declare it so that trajectories carry correct log-probs for later off-policy correction (importance sampling if BC later transitions to off-policy RL).
- Trajectory format is open (README "low-priority / defer" item 20: "Trainer integration — replay buffer format, episode boundary semantics"). Minimum schema: `{ schema_hash, tick, agent_id, obs_bytes, mask_bytes, action, log_prob, reward, done, terminal }`. Should be emitted as columnar parquet or NDJSON with a sidecar manifest declaring schema_hash for loud failure on mismatch.
- Episode boundary semantics for zero-player worldgen. Agents live indefinitely — when does an "episode" end for reward discounting? Proposals: death-as-terminal + rolling windows, or fixed N-tick horizons.
- Utility → Neural swap path is actually Utility → BC-warm Neural → RL fine-tune. Story 20's reward block governs the second half.

**Related stories:** 16 (per-tick trajectory emission), 18 (backend comparison), 20 (reward block for RL fine-tune), 22 (curriculum), 23 (checkpoint deploy).

---

### Story 20: Declare reward in DSL (with actor-critic extension)
**Verdict:** PARTIAL

**User's framing note:** "Great, but we will also want to support actor critic"

**How the DSL supports this:**
Proposal §2.5 gives us a declarative reward block, shown below verbatim from the spec:

```
reward {
  delta(self.needs.satisfaction_avg)         × 0.1
  delta(self.hp_frac)                        × 5
  +1.0  on event(EntityDied{killer=self ∧ target.team ≠ self.team})
  -1.0  on event(EntityDied{target ∈ self.close_friends})
  +0.05 per behavior_tag accumulated this tick
  +2.0  on event(QuestCompleted{quest.party_member_ids ∋ self})
  ...
}
```

The compiler emits a per-tick reward kernel that diffs pre/post-tick observations and scans events involving `self`, plus logging hooks for (observation, action, reward) tuples, plus validation that every term references a declared view or event (§2.5 "Compiler emits"). This covers vanilla REINFORCE cleanly: per-step reward → discounted return → policy gradient.

**Actor-critic is where the current schema stops short.** Proposal §2.5 only specifies a scalar reward stream. Nothing in the current DSL declares a **value function head**, an **advantage estimator**, or **PPO clip parameters**. These need to be added, and they're natural additions because the policy block already has a `backend Neural { weights: ..., h_dim: ... }` declaration — the value head is just another output projection off the same trunk.

**Implementation walkthrough (proposed actor-critic extension):**

Proposed extension to §2.5 and §2.4 — the policy block grows `value` and `training` sub-blocks:

```
policy NpcDecision {
  observation { ... }
  action { ... }          // heads: macro_kind, micro_kind, target, pos_delta, magnitude, quest_type, ...
  mask { ... }
  reward { ... }          // scalar reward stream (unchanged from §2.5)

  value {
    // Separate head off the shared trunk. Scalar V(s).
    head scalar v_pred                  // shared trunk → Linear(h_dim, 1)
    // Or: separate trunk (decoupled A/C). Default shared; override per-block.
    trunk shared                        // shared | separate
    loss mse                            // mse | huber
    clip_range 10.0                     // optional reward clipping
  }

  advantage {
    kind gae                            // gae | nstep | montecarlo
    gamma   0.99
    lambda  0.95                        // GAE-λ; ignored for nstep/MC
    normalize per_batch                 // per_batch | per_agent | none
  }

  training {
    algorithm ppo {
      clip_epsilon       0.2
      vf_coef            0.5
      entropy_coef       0.01
      n_epochs           4
      minibatch_size     4096
      target_kl          0.02           // optional early stop
    }
    // or: algorithm reinforce { baseline: v_pred }
    // or: algorithm bc { loss: cross_entropy }
    optimizer adamw { lr: 3e-4, beta2: 0.98, weight_decay: 1.0 }
    grokfast ema { alpha: 0.98, lamb: 2.0 }   // optional (MEMORY.md pattern)
  }

  backend Neural { weights: "npc_v3.bin", h_dim: 256 }
}
```

Concrete answers to the prompt's actor-critic questions:

- **Does the policy block need a `value` head separate from action heads?** Yes. It's declared in a `value` sub-block that sits alongside `action`. Default trunk is shared with the action heads (one encoder, multi-head output); `trunk separate` produces a decoupled critic when divergence hurts training (see MEMORY.md note: "PPO collapses — bad value head" — that issue came from a shared-trunk bootstrap that the action gradients destabilized; having `trunk shared|separate` as a declared toggle lets the trainer recover without code changes).
- **Where does the critic train?** Same place as the policy — the compiler emits a per-tick dataset row `(obs, action, log_prob, reward, v_pred)` where `v_pred = model.value_head(obs)`. The offline trainer computes GAE advantages (δ_t = r_t + γ V(s_{t+1}) − V(s_t), Â_t = δ_t + γλ Â_{t+1}) per the declared `advantage { kind=gae, gamma=0.99, lambda=0.95 }` block. Critic loss is MSE(V(s), R_t) where R_t is the discounted return or TD(λ)-target. Both policy and value losses backprop through the shared trunk (unless `trunk separate`).
- **What's the reward block syntax that supports both REINFORCE and PPO/AC training?** The `reward` block is identical — it always declares the per-step scalar reward. What **differs** is the `advantage` and `training` blocks. REINFORCE uses `training { algorithm reinforce { baseline: v_pred | none } }`; PPO uses `training { algorithm ppo { clip_epsilon: 0.2, ... } }`. Behavior cloning uses `training { algorithm bc }` (value + advantage blocks ignored). This separation of concerns (reward = environment semantics, advantage = credit assignment, training = optimizer recipe) keeps the reward block the same across all algorithms while making the switch declarative.
- **PPO clip parameters?** Explicit: `clip_epsilon` (policy ratio clip, default 0.2), `vf_coef` (value loss weight, default 0.5), `entropy_coef` (exploration bonus, default 0.01), `n_epochs` (PPO epochs per rollout, default 4), `minibatch_size`, optional `target_kl` for early stopping. All declared, all validated at compile time against the chosen `algorithm` (REINFORCE rejects `clip_epsilon`).

The hierarchical action heads (§2.2 macro_kind × micro_kind × pointer × continuous) each need their own `log_prob` and `entropy` computed and summed — this is a codegen concern but the DSL already declares the heads.

**Gaps / open questions:**
- **Credit assignment for macro actions.** Proposal §2.5 flags this as `[OPEN]`: "Reward shaping for rare macro actions (Conquest reward arrives 2000+ ticks after the PostQuest decision)." GAE with γ=0.99 has an effective horizon of ~100 ticks — far too short. Options: per-head γ (macro head uses γ_macro=0.999 ≈ 1000-tick horizon; micro head uses γ_micro=0.99), n-step returns with explicit goal-conditioned bootstrap, or separate trainers per head. The DSL should allow `advantage { macro: {gamma: 0.999}, micro: {gamma: 0.99} }` overrides.
- **Shared vs separate trunk default.** MEMORY.md records that PPO collapsed on shared-trunk. Need an empirical rule for when to decouple — probably "decouple if entropy collapse observed within N minibatches" — but that's a training-time heuristic not a DSL concern.
- **Value clipping (PPO's vf_clip trick).** Not in the current proposal. Should probably be a `value { clip_range: 0.2 }` knob.
- **Off-policy vs on-policy dispatch.** PPO/REINFORCE are on-policy; BC is off-policy. When the utility backend emits trajectories for BC, log_probs are mis-calibrated for importance correction. Need either a strict "BC only for utility trajectories" rule or importance-sampling machinery (V-trace / Retrace). Defer.
- **Reward shaping per-role.** A leader's reward shouldn't be the same as a commoner's (cascade-impact rewards skew everything). Proposal doesn't distinguish. Could declare reward overrides per role via observation-conditional terms in the reward block — but this is speculative.

**Related stories:** 16 (training data emission — must log `v_pred` and `log_prob` alongside action), 19 (BC as the `algorithm bc` training variant), 21 (rare-action upweighting interacts with advantage normalization), 22 (curriculum stages swap the `training` block over time).

---

### Story 21: Up-weight rare actions in training
**Verdict:** PARTIAL

**User's framing note:** "Essential"

**How the DSL supports this:**
The hierarchical macro/micro action decomposition (proposal §2.2) is half of the answer — it already separates rare (macro: PostQuest/AcceptQuest/Bid) from common (micro: Hold/Move/Attack/...) at the head level. Each head can, in principle, get its own learning rate or loss weight. Proposal §7 "Risks" acknowledges rare-action training is hard and lists "prioritized replay buffer with rare-event up-weighting" as a mitigation. But the DSL surface for declaring which actions are rare, and at what weight, isn't specified.

**Implementation walkthrough (proposed):**

Add a `training_weights` annotation to action heads, or a per-action tag in the action block, that the replay sampler honors:

```
action {
  head categorical macro_kind: enum {
    NoOp,                              @training_weight(1.0)
    PostQuest { @training_weight(50.0) @rare },
    AcceptQuest { @training_weight(20.0) },
    Bid { @training_weight(10.0) },
  }

  head categorical micro_kind: enum {
    Hold,                              @training_weight(0.1)  // very common, downweight
    MoveToward, Flee,                  @training_weight(1.0)
    Attack, Cast, UseItem,             @training_weight(2.0)
    Harvest, Eat, Drink, Rest,         @training_weight(1.0)
    PlaceTile, PlaceVoxel, HarvestVoxel, @training_weight(5.0)  // rare
    Converse, ShareStory,              @training_weight(3.0)
  }

  head categorical quest_type: enum QuestType {
    Hunt, Escort, ...,                 // defaults to 1.0
    Conquest @training_weight(100.0),   // only a handful per campaign
    Marriage @training_weight(50.0),
    Found    @training_weight(100.0),
  }
}
```

Two consumers of these weights:

1. **Prioritized replay buffer**: when offline training samples minibatches, rows get probability proportional to `max(training_weight[head_k] for each head k with nonzero log_prob)`. Rows where the agent chose `Conquest` are ~100× more likely to appear in a minibatch than rows where it chose `Hold`. Standard prioritized-experience-replay math (Schaul et al.) with importance-sampling correction `w_i = (N × P_i)^{-β}` to debias the policy gradient.
2. **Per-head loss scaling** at the training loss level: the cross-entropy loss for the `macro_kind` head is multiplied by `5.0` if the sampled macro was rare, so even in a uniformly sampled batch the rare macros get gradient signal.

The sampler also needs **action-frequency telemetry** to detect when weights need tuning — running counts per `(head, action)` tuple, exposed via the same telemetry hooks story 24 uses. If `count(PostQuest) / count(NoOp) < 10^-4` after N steps, emit a warning.

**Gaps / open questions:**
- Whether to annotate weights at the action definition site (shown above) or in a separate `training { ... }` block (cleaner separation, but loses locality). Probably the latter to keep the action vocab language-only.
- Replay buffer format is unspecified (README defer-list item 20). Minimum: a priority-tree or sum-tree index over the trajectory file. `sample_weighted(batch_size)` returns rows with IS-correction weights.
- Pointer-head rare-target upweighting is harder — pointer targets are slot indices, not fixed verbs. Rare-target emphasis is an advantage-estimator concern (e.g. give higher advantage to selecting the rarely-selected slot 9 when rewarded), not a replay-priority concern.
- Interaction with curriculum (story 22): in stage 1 `Conquest` is masked out, so `training_weight(100)` is wasted. Weights should be **stage-scoped**: stage 1 declares `Conquest @training_weight(0)`, stage 5 declares `Conquest @training_weight(100)`. See story 22.
- MEMORY.md note: V3 pointer action space has "BC alone: 2.9% (model collapses to hold without engagement heuristic)" — the `Hold` downweight above is the DSL-level prophylactic for exactly that collapse.

**Related stories:** 20 (training block where weights live), 22 (stage-scoped weight overrides), 24 (action-frequency telemetry feeds back to weight tuning).

---

### Story 22: Curriculum / staged training
**Verdict:** PARTIAL

**User's framing note:** "Essential"

**How the DSL supports this:**
The mask language (§2.3) is the right primitive — masking out unreached verbs is already the declarative mechanism for role gating, and curriculum is just training-time role gating. The README "Settled" bullet 4 explicitly ties role power to mask: "Role power = mask + cascade, not a smarter policy." Stage N's curriculum is literally a training-time mask override: stage 1's mask zeroes every action except `Hunt` and `Eat`; stage 5's mask zeroes nothing. Because the mask is per-head and compiled to a boolean tensor, overriding it per stage is cheap.

What's missing: a **stage declaration surface** and **transition criteria**.

**Implementation walkthrough (proposed):**

Add a `curriculum` block to the policy:

```
policy NpcDecision {
  ...

  curriculum {
    stage Foraging {
      mask_override {
        // Allow only these micro_kinds; everything else training-time-masked to 0
        micro_kind allow [Hunt, Eat, Drink, Rest, MoveToward, Hold]
        macro_kind allow [NoOp]
      }
      training_weights {
        // Stage-specific upweights
        micro_kind { Hunt: 5.0, Eat: 3.0 }
      }
      transition_when {
        metric action_entropy(micro_kind)  >= 1.2
        metric mean_episode_reward         >= 2.0
        min_steps 50_000
      }
      reward_override {
        // Emphasize survival during foraging stage
        +5.0 × delta(self.hunger_frac)   // was 0.1 × delta(needs.satisfaction)
      }
    }

    stage Combat {
      inherits Foraging
      mask_override {
        micro_kind allow_additional [Attack, Cast, Flee]
      }
      transition_when {
        metric win_rate_vs_baseline >= 0.4
        min_steps 100_000
      }
    }

    stage Social {
      inherits Combat
      mask_override {
        micro_kind allow_additional [Converse, ShareStory]
      }
      ...
    }

    stage Macro {
      inherits Social
      mask_override {
        macro_kind allow_additional [PostQuest, AcceptQuest, Bid]
        quest_type allow [Hunt, Escort]   // start narrow, expand
      }
      ...
    }

    stage Full {
      inherits Macro
      mask_override { macro_kind allow_all; micro_kind allow_all; quest_type allow_all }
    }
  }
}
```

Mechanics:

1. **`mask_override` composes with runtime mask.** The runtime mask (§2.3 predicates: `Attack(t) when is_hostile ∧ distance < AGGRO_RANGE`, etc.) is AND-ed with the stage mask. Runtime rules still hold ("can't attack a non-hostile target") but the stage further restricts ("can't attack at all, even hostile, during Foraging"). Implementation: compiler emits `stage_mask_i[N × NUM_ACTIONS]` static tensor per stage; final mask = `runtime_mask & stage_mask_i`.
2. **`transition_when` block** evaluates against emitted training telemetry (story 24). Criteria are AND-ed: entropy ≥ threshold AND mean episode reward ≥ threshold AND min steps elapsed. When all true, move to next stage. No auto-regression — once advanced, stays advanced (unless the trainer manually rolls back by editing the stage pointer in the checkpoint).
3. **`inherits`** lets later stages additively expand masks without redeclaring base.
4. **`reward_override` and `training_weights`** are stage-scoped, overriding the base `reward` and per-action weights.
5. The compiler validates that every action mentioned in a stage's mask_override actually exists in the action block, and that `transition_when` metrics are declared in the telemetry surface.

Transition criteria could also be **action-distribution-driven** — e.g. transition out of Combat once the model reliably selects `Attack` when hostile targets are in range (precondition coverage ≥ 90%). This needs a declared metric hook; see story 24.

**Gaps / open questions:**
- **Stage state.** Which stage are we in? Lives in the model checkpoint metadata alongside weights, schema hash, training step count. Loading a checkpoint resumes at the stored stage.
- **Stage pointer vs stage weights.** Do we keep one set of weights per stage (stage 1 weights used once, frozen) or continuously finetune the same weights across stages? Default: continuous — the mask widening is what drives behavior change, not a reset. The Grokfast EMA / AdamW state also persists.
- **Curriculum and BC bootstrap interact.** If utility trajectories were generated under Stage 5 (full action space), BC on stage 1 is awkward — we'd be dropping all the rare-macro data. Solution: filter BC data by the stage mask, or generate stage-specific utility trajectories.
- **Reward scheduling (`reward_override` above)** is an actor-critic correction. Changing the reward function mid-training is a non-stationary-environment problem; the critic V(s) becomes stale at every stage transition. Mitigation: re-warm the critic by freezing the policy for M steps after each transition and letting V(s) catch up.
- **Stage transitions and early stopping.** If `target_kl` (from PPO training block) fires before `transition_when` is satisfied, stage is stuck. Need explicit fail-forward or abort semantics.

**Related stories:** 20 (training block's reward/algorithm are what curriculum overrides), 21 (stage-scoped action weights), 24 (entropy + action-distribution metrics feed transition criteria).

---

### Story 23: Deploy a model checkpoint
**Verdict:** SUPPORTED

**User's framing note:** "Essential"

**How the DSL supports this:**
Proposal §2.4 declares `backend Neural { weights: "npc_v3.bin", h_dim: 256 }` as a file path on the policy block. The schema-hash guard (§4) is specified: "Every model checkpoint stores its training-time hash. Loading a model whose hash mismatches the current DSL is a hard error." Story 15's user annotation is explicit: "WE DO NOT WANT TO HAVE V1 AND V2 AND V3 in the same codebase. Git can be used to store versions." So schema mismatch **FAILS LOUD** — no migration tables, no pad-zero fallback, no auto-upgrade path. The §4 proposal's migration-table language is superseded by the story 15 constraint.

**Implementation walkthrough:**

Checkpoint file layout:
```
npc_v3.bin
├── header (fixed offset, 128 bytes):
│   ├── magic:            "NPCPOL\0\0"          (8 bytes)
│   ├── format_version:   u32                   (4 bytes)
│   ├── schema_hash:      [u8; 32]              (32-byte SHA256 of obs+action+mask schema)
│   ├── policy_name:      [u8; 32]              ("NpcDecision\0...")
│   ├── training_step:    u64                   (for provenance)
│   ├── stage_name:       [u8; 16]              (curriculum stage from story 22)
│   ├── reserved:         [u8; 12]
│   └── weights_offset:   u64                   (offset to weights section)
├── weights section (contiguous tensor blobs with named offsets)
└── footer: CRC32 over weights section
```

Hot-swap semantics:

1. Trainer drops `npc_v3.bin` into the configured weights directory. Either the path is hot-polled (watcher) or explicitly signaled via an admin command.
2. Runtime reads the header **before** loading weights.
3. **Schema hash check (FAILS LOUD):** `if header.schema_hash != compiled_dsl.schema_hash { abort_with_error("...") }`. Per story 15, there is no fallback, no migration table, no pad-zero behavior. The error message shows the expected hash, the loaded hash, and points to the DSL commit that produced each. The current tick's decisions continue under the old weights; the new weights are rejected entirely.
4. **Magic + format_version check (FAILS LOUD):** reject non-matching magic or unrecognized format_version with a clear error.
5. **CRC check (FAILS LOUD):** detect truncated/corrupt file.
6. **Shape check (FAILS LOUD):** every tensor declared by the neural backend's architecture (`h_dim: 256`, head dimensions derived from action block) is matched against the weights file. Missing tensors, wrong shapes → abort load.
7. **Atomic swap:** weights load into a staging buffer; on success, a single pointer swap (RCU or mutex) makes the next tick use the new weights. Tick N uses old, tick N+1 uses new — no partial state. In-flight forward passes finish against the weights they started on.
8. **Graceful fallback when no weights at all:** if `backend Neural` is declared but no file exists, falls back to `backend Utility` (if declared as the bootstrap in the same policy) or refuses to start. This is the one graceful path; all others fail loud.

What fails gracefully vs catastrophically:

| Failure mode                          | Behavior                                    |
|---------------------------------------|---------------------------------------------|
| File doesn't exist at startup         | Graceful: fall back to Utility if declared  |
| File doesn't exist at hot-swap        | Graceful: keep current weights              |
| Magic mismatch                        | **LOUD ABORT**: wrong file type             |
| Format version mismatch               | **LOUD ABORT**: codebase too old/new        |
| Schema hash mismatch                  | **LOUD ABORT** (per story 15): no migration |
| Tensor shape mismatch                 | **LOUD ABORT**: reject swap, keep old       |
| CRC mismatch                          | **LOUD ABORT**: file corrupt                |
| Stage name in checkpoint unknown      | **LOUD ABORT**: curriculum mismatch         |
| Out-of-memory during stage load       | **LOUD ABORT**: reject swap, keep old       |

Story 15 constraint is load-bearing here: "Git can be used to store versions, it is nothing but waste to support backwards compatibility." Every code path that might silently pad-zero, upcast, or remap channels is **explicitly rejected**. The only "smart" behavior is the atomic pointer swap + rollback-on-error; everything else is a bright-line check with a clear error.

**Gaps / open questions:**
- Whether the schema hash covers action vocabulary + reward, or just observation. Proposal §4 [OPEN] item 17. Recommendation given story 15's strictness: hash **all three** (observation + action + reward) — changing any of them changes what the model learned to do, and silent success is the failure mode we're preventing.
- Does the curriculum `stage_name` participate in the schema hash? Probably not — stage is training provenance, not input/output layout. But the runtime should warn if loaded stage doesn't match the DSL-declared expected production stage.
- How to handle rolling deploys / canarying. Not in the current proposal. Could add `backend Neural { weights: "npc_v3.bin", canary_weights: "npc_v4.bin", canary_frac: 0.05 }` — 5% of agents use v4 for A/B.
- Weight file format: raw packed tensors vs safetensors vs custom. Recommendation: **safetensors-style** with named tensor offsets so the header can be inspected without loading the whole file.
- Hot-swap during an in-flight training step is UB unless training and inference use separate weight buffers. Trainer writes to a staging path; atomic rename triggers the runtime watcher.

**Related stories:** 14 (schema hash bump on obs change), 15 (hash mismatch FAILS LOUD — constraint-defining), 22 (stage metadata in checkpoint).

---

### Story 24: Detect mode collapse
**Verdict:** PARTIAL

**User's framing note:** "Essential"

**How the DSL supports this:**
The runtime already has everything needed to compute action-distribution telemetry — the `PolicyBackend::evaluate_batch` call (§2.4) produces `ActionBatch`, which is a typed per-agent action containing head choices. Counting those choices per tick gives raw distributions. What the DSL doesn't yet declare is **which metrics to emit, how to emit them, and alert thresholds**. That's a `telemetry { ... }` block that needs to exist.

**Implementation walkthrough (proposed):**

Add a `telemetry` block to the policy:

```
policy NpcDecision {
  ...

  telemetry {
    // Per-head entropy — primary mode-collapse detector.
    metric entropy_macro_kind  = entropy_of(action.macro_kind)
           window 1000                    // rolling window size in ticks
           emit_every 100                 // tick cadence for emission
           alert when value < 0.3         // log2(1.35) — model picked one verb >70% of the time

    metric entropy_micro_kind  = entropy_of(action.micro_kind)
           window 1000
           alert when value < 0.5         // tighter — more options, less collapse allowed

    metric entropy_quest_type  = entropy_of(action.quest_type)
           conditioned_on action.macro_kind = PostQuest
           window 10000                   // rare — need longer window
           alert when value < 0.8

    // Per-head action frequencies — individual-action detection (not whole-head).
    metric freq_micro          = histogram(action.micro_kind) window 1000 emit_every 100
           alert when max_bin > 0.85      // any one action > 85% of tick choices

    // Parameter-head coverage.
    metric pointer_slot_coverage = coverage(action.target, n_slots=42) window 5000
           alert when value < 0.3         // using <30% of available slot types

    metric continuous_pos_delta_stddev = stddev(action.pos_delta) window 1000
           alert when value < 0.05        // collapsed to near-constant movement

    // Reward + value head diagnostics (feeds back to story 20 actor-critic tuning).
    metric mean_reward         = mean(reward) window 1000
    metric value_error         = mse(value.v_pred, montecarlo_return) window 1000
           alert when value > 10.0        // critic has exploded

    // Mask-coverage: which masked-in actions does the model actually use?
    metric mask_util           = mean(argmax_action_in_mask_top_k(k=3)) window 1000
           alert when value < 0.7         // model picking low-probability (per its own logits) actions
  }
}
```

Mechanics:

1. **Runtime computes metrics in a tick-end hook.** Because `evaluate_batch` returns the full per-agent `ActionBatch`, histogram updates are a single parallel scatter per head. Cost is O(N agents × N heads) per tick, ~trivial.
2. **Rolling windows** implemented as ring buffers per metric, sized by `window`. Entropy and stddev computed from running sums (Welford for numerically stable variance).
3. **Emission** writes to a structured log (JSON lines, one record per emit cadence) with `{ metric_name, value, tick, policy_name }`. Training infrastructure consumes this for dashboards and alert routing.
4. **Alert criterion** is declared inline. When `value` hits the `alert when` predicate, the runtime emits a `METRIC_ALERT` event (loud: stderr + log + optional webhook). It does not halt training — the trainer decides.
5. **Conditional metrics** (e.g. `entropy_quest_type conditioned_on macro_kind=PostQuest`) only sample on tick when the condition fires. Essential for rare-macro-action telemetry: measuring `quest_type` entropy over all ticks is meaningless when >99% of ticks emit NoOp for macro_kind.
6. **Curriculum integration:** `transition_when` criteria in the curriculum block (story 22) can reference declared metrics directly: `metric entropy_micro_kind >= 1.2`.

Alert criteria — specific thresholds for mode-collapse detection:

- **Entropy-based (primary):** `H(action) < ε` where `ε ≈ log(K) × 0.3` for a K-way categorical head. A K=12 micro_kind head has `H_max = log2(12) ≈ 3.58`; alert at `< 0.5`. For K=4 macro_kind, `H_max = 2`; alert at `< 0.3`. These are starting points; trainer tunes.
- **Max-bin frequency:** `max_k P(action = k) > 0.85` — single-action dominance.
- **Zero-frequency floor:** `min_k P(action = k) < 0.001` in a stage where action k is unmasked — model ignoring a valid verb.
- **Pointer-slot coverage:** fraction of `[0, NUM_SLOTS)` ever selected over the window — catches collapse to a single target.
- **Continuous-head stddev:** too-low stddev on pos_delta / magnitude indicates collapsed continuous policy.
- **Critic-specific (story 20):** value-error explosion (`mse(v, R) > 10`) or value clip saturation — catches bad-critic issues MEMORY.md already documented ("PPO collapses — bad value head").

Reference point from MEMORY.md: V2 actor-critic "12% win rate (HvH-specialized, doesn't transfer)" and BC-alone "model collapses to hold" (2.9% win rate) — these are exactly the patterns entropy + max-bin alerts should have caught in real time.

**Gaps / open questions:**
- **Where does the alert go?** Runtime stderr + log is cheap; webhook/email is trainer-specific. Probably keep to structured log and let external infra decide.
- **Cross-agent vs per-agent distributions.** Declared metrics above are aggregated across all agents per tick. Per-role / per-stage distributions might also be useful — needs a `group_by role` clause. Defer unless empirically needed.
- **Historical baseline.** "Entropy is low" is absolute; "entropy is 3σ below the training-window mean" is relative. Current proposal is absolute thresholds; extending to z-score-style thresholds is a later refinement.
- **Interaction with curriculum.** When the curriculum widens the mask (Stage 1 → Stage 2), entropy should *spike* (more options) — an entropy-too-low alert firing right before transition is expected, not a bug. Alerts should be suppressed for N ticks after stage transition.
- **Action-sequence telemetry.** Mode collapse sometimes manifests as cycles (`Hold, Move, Hold, Move, ...`) rather than a single action dominating. Detecting this needs n-gram distributions, more expensive. Defer.

**Related stories:** 20 (value head diagnostics feed into the critic loss tuning), 21 (action-frequency histograms are the feedback signal for weight tuning), 22 (metrics drive `transition_when` curriculum criteria), 23 (checkpoint metadata could include last-known telemetry snapshot for provenance).

---

## Cross-story summary

| Story | Verdict  | Key gap                                             |
|-------|----------|-----------------------------------------------------|
| 19    | SUPPORTED | utility `log_prob` calibration; trajectory format  |
| 20    | PARTIAL   | needs `value` + `advantage` + `training` sub-blocks (proposed) |
| 21    | PARTIAL   | needs per-action `training_weight` annotations + replay sampler hook |
| 22    | PARTIAL   | needs `curriculum { stage { ... } }` block with `mask_override` + `transition_when` |
| 23    | SUPPORTED | hash mismatch FAILS LOUD per story 15; atomic swap; graceful only for "file missing" |
| 24    | PARTIAL   | needs `telemetry { metric ... alert when ... }` block |

**Cross-cutting observation:** the proposal is strongest where ML contracts meet the runtime (obs packing, masks, action heads, schema hashing, backend trait). It's thinnest where trainers meet the DSL (reward-to-algorithm pipeline, curriculum staging, telemetry emission, replay buffer format). README "low-priority / defer" item 20 ("Trainer integration — replay buffer format, episode boundary semantics") is no longer low-priority if stories 19-24 are essential; the next proposal revision should promote it and spec out `training { ... }`, `curriculum { ... }`, `telemetry { ... }` blocks as first-class policy sub-blocks alongside `observation / action / mask / reward / backend`.

The actor-critic extension (story 20) is the most consequential addition: it introduces the `value` head, `advantage` block, and `training { algorithm ppo { ... } }` surface, all of which then become the substrate stories 21-24 extend (stage-scoped training weights, per-stage algorithm overrides, value-error telemetry).
