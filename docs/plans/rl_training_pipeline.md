# RL Training Pipeline for V6 Actor-Critic

Single-GPU (4090) training plan for the 2.4M-parameter V6 actor-critic. The central theme: **stop handrolling RL infrastructure and use rl4burn**, which already provides the algorithms that were reimplemented with bugs in `training.rs` and `impala_train.rs`.

---

## What we have now

**Model (V6):** EntityEncoderV5 (539K) → SpatialCrossAttention (68K) → LatentInterface/ELIT (399K) → CfC temporal cell (427K) → dual decision heads (position Gaussian + combat pointer). AbilityTransformer (596K) encodes DSL tokens per ability, cross-attended to entity tokens. Two-headed value (attrition + survival) for curriculum. Total ~2.4M params.

**Action space:** Continuous position head (μ,σ Gaussian over normalized world coords) + discrete combat pointer head (10 combat types × pointer attention over entity tokens). V4 dual-head format: 9 movement directions + 10 combat types with pointer targeting.

**rl4burn library (largely unused):**

| rl4burn provides | Currently used? | Status |
|------------------|----------------|--------|
| `SyncVecEnv`, `Env` trait | Yes (combat-trainer) | Working |
| `masked_ppo_collect` / `masked_ppo_update` | Yes (combat-trainer) | Working |
| `vtrace_targets()` | Yes (re-exported) | Working |
| `normalize()`, `clip_grad_norm()`, `value_loss()` | Yes (utilities) | Working |
| `League`, `PfspMatchmaking`, `SelfPlayPool` | **No** | Available, untouched |
| `imagine_rollout()`, `lambda_returns()` | **No** | Available, untouched |
| `bc_loss_discrete`, `bc_loss_multi_head` | **No** | Handrolled instead (with passivity bug) |
| `actor_learner_collect()` with behavior log_probs | **No** | Handrolled instead (with temperature bug) |
| `InferenceChannel` (centralized GPU batching) | **No** | Available, untouched |
| `ActionDist::MultiDiscrete`, `ActionDist::Continuous` | **No** | Available, untouched |
| `ReplayBuffer` | Yes (impala_train) | Working |
| `TensorBoardLogger` | Yes | Working |

**Handrolled code that introduced bugs:**
- `training.rs` `compute_loss()`: custom IMPALA loop, custom BC loss, custom log_prob computation
- `impala_train.rs`: custom actor-learner collection, custom trajectory handling
- `rl_episode.rs`: temperature scaling applied during collection but not recoverable during training → PPO KL=3M
- `training.rs` `compute_bc_loss()`: no inverse-frequency weighting → 97.5% accuracy on "hold", 2.9% win rate

**What's been tried:**
- BC on oracle data: 97.5% type accuracy, 98.3% pointer accuracy — but only 2.9% HvH win rate (passive agent)
- REINFORCE: unstable, pg_loss grows exponentially after 3-4 iterations
- PPO: broken due to temperature scaling mismatch in handrolled log_probs (KL diverged to 3M)
- V3 pointer: ~30% HvH win rate (BC + engagement heuristic + REINFORCE)
- V2 flat 14-action: 54.4% HvH with REINFORCE (worked because it used rl4burn's simpler path)
- Next-state prediction pretraining: works at short horizons, overfits at long horizons with large models

---

## Stage 0 — Migrate V6 training onto rl4burn (2-3 days engineering)

The three known training bugs all come from handrolled code that reimplements what rl4burn already provides. Rather than patching the handrolled code, replace it.

### 0a. V6 adapter trait for rl4burn

The reason V6 training was handrolled: rl4burn's `MaskedActorCritic` trait returns `(logits: [B, N], values: [B])` — flat discrete only. V6 has three heterogeneous heads. The fix is a thin adapter, not a full reimplementation.

**Add to rl4burn or to tactical_sim:**

```rust
pub struct HybridAction {
    pub position: [f32; 2],        // continuous
    pub combat_type: usize,        // discrete, 10 classes
    pub target_idx: usize,         // pointer into entity tokens
}

pub struct HybridOutput<B: Backend> {
    pub position_mu: Tensor<B, 2>,     // [B, 2]
    pub position_log_std: Tensor<B, 1>, // [2]
    pub combat_logits: Tensor<B, 2>,   // [B, 10]
    pub pointer_logits: Tensor<B, 2>,  // [B, S] (variable-length)
    pub values: Tensor<B, 1>,         // [B]
}

pub trait HybridActorCritic<B: Backend> {
    fn forward_hybrid(&self, ...) -> HybridOutput<B>;
}
```

**Log-prob computation** (single canonical implementation, used by both collection and training):

```rust
fn hybrid_log_prob(output: &HybridOutput, action: &HybridAction) -> f32 {
    let lp_pos = gaussian_log_prob(output.position_mu, output.position_log_std, action.position);
    let lp_combat = categorical_log_prob(output.combat_logits, action.combat_type);
    let lp_pointer = categorical_log_prob(output.pointer_logits, action.target_idx);
    lp_pos + lp_combat + lp_pointer  // independent heads, sum log-probs
}
```

This eliminates the temperature mismatch bug by construction — there's one log_prob function, used everywhere. No separate "collection" vs "training" code paths.

### 0b. Replace handrolled collection with `actor_learner_collect`

rl4burn's `actor_learner_collect()` already:
- Tracks `behavior_log_probs` per step (for V-trace importance weights)
- Handles episode boundaries and resets
- Returns `Trajectory` structs ready for V-trace

Adapt it to use `HybridActorCritic` instead of `DiscreteActorCritic`. The trajectory struct needs `HybridAction` instead of `i32` actions — either generalize rl4burn's `Trajectory` or wrap it.

### 0c. Replace handrolled BC with `bc_loss_multi_head` + continuous loss

rl4burn's `bc_loss_multi_head` handles the discrete heads (combat type + pointer). Add a continuous MSE term for the position head. Critically, add **inverse-frequency weighting** to the discrete loss — rl4burn's BC doesn't do this by default, but it's a one-line modification to the cross-entropy call.

This fixes the passivity bug: hold actions dominated oracle data at ~60% of samples. With inverse-frequency weighting, the loss contribution from rare actions (attack, ability) is amplified ~10×.

### 0d. Replace handrolled IMPALA loop with rl4burn primitives

The current `compute_loss()` in `training.rs` (lines 294-360) manually computes:
- Policy loss with log-ratio clipping
- Value loss
- Entropy bonus
- Action masking

rl4burn already provides all of these as composable functions. The custom `train_step()` can be replaced with:
1. `actor_learner_collect()` → trajectories
2. `vtrace_targets()` → advantages (already used)
3. PPO-clipped policy loss using `ActionDist` sampling + the hybrid adapter
4. `value_loss()` (already used)
5. Standard backward pass with `clip_grad_norm()` (already used)

The only truly custom piece is the multi-head loss weighting (position vs combat vs pointer) — keep that as a thin wrapper around rl4burn's loss functions.

---

## Stage 1 — Value alignment before RL (2-4 hours)

After BC pretraining, the policy is reasonable but the value network is random. Starting PPO immediately produces garbage advantages that destroy the policy. This is the single most impactful missing piece.

### Protocol

1. **Freeze entire policy** (position head, combat pointer head, ability cross-attention, CfC cell, encoders)
2. **Train only the value head** for 50K environment steps against Combined (squad AI) opponents
   - The two-headed value (attrition + survival) in `value_head.rs` is already designed for this
   - Use rl4burn's `actor_learner_collect()` (from Stage 0b) with frozen policy to gather trajectories
   - Compute V-trace targets, update only value head parameters
3. **Validate**: value predictions should correlate >0.6 with actual episode returns before proceeding

### Implementation

With the rl4burn migration from Stage 0, this is just a config flag: pass a parameter group with LR=0 for all modules except `value_head`. No custom freezing logic needed — rl4burn's optimizer handles parameter groups.

---

## Stage 2 — Graduated unfreezing into PPO (8-16 hours)

Layer-by-layer unfreezing prevents catastrophic forgetting of BC-learned representations.

### Phase 2a — Policy heads only (100K steps, ~1 hour)

- Unfreeze position head + combat pointer head
- Keep encoder, latent interface, CfC cell, ability transformer frozen (LR=0 in their param groups)
- PPO with `clip_eps=0.1` (tight), `entropy_coef=0.02`
- KL penalty toward BC policy: `β=1.0` (strong anchor)
- LR: `1e-4` for heads only

### Phase 2b — CfC temporal cell (200K steps, ~2 hours)

- Unfreeze `cfc_cell` with discriminative LR: `1e-5` (10× lower than heads)
- Reduce KL penalty: `β=0.5`

### Phase 2c — Latent interface + spatial cross-attention (500K steps, ~4 hours)

- Unfreeze `latent_interface` and `spatial_cross_attn` at `1e-5`
- The ELIT bottleneck's learned latent tokens are the most sensitive — they compress the entire game state
- Reduce KL penalty: `β=0.1`, increase `clip_eps=0.2`

### Phase 2d — Full fine-tuning (remaining budget)

- Unfreeze entity encoder and ability transformer at `1e-6` (100× lower than heads)
- KL penalty: `β=0.05` (maintained throughout, never zero)
- Cosine LR annealing

### Implementation

Per-module learning rate groups via Burn's optimizer API. A `--phase {1,2a,2b,2c,2d}` flag selects which modules have non-zero LR. Each phase loads the previous phase's checkpoint.

---

## Stage 3 — Parameterized opponent diversity (ongoing)

The current scenario gen creates diverse compositions but the AI behavior is uniform (squad AI with default personality). This limits what the policy can learn.

### Parameterized scripted opponents

Extend `SquadAiState` / `Personality` with a parameter vector sampled per episode:

| Parameter | Range | Effect |
|-----------|-------|--------|
| aggression | 0.0–1.0 | threshold for engaging vs retreating |
| focus_fire | 0.0–1.0 | probability of targeting lowest-HP enemy vs nearest |
| ability_eagerness | 0.0–1.0 | urgency threshold for using abilities |
| formation_tightness | 0.0–1.0 | spacing between units |
| retreat_threshold | 0.0–0.5 | HP% at which units disengage |

This creates ~100 meaningfully different opponents from the existing squad AI at zero GPU cost.

### Prioritized Level Replay (PLR)

After each episode, score the (scenario, opponent_params) pair by TD-error magnitude from the value head. Maintain a priority queue of 500 configurations. Sample training episodes proportionally to priority scores with temperature τ=0.5.

---

## Stage 4 — Dyna-style world model augmentation (4-8 hours to train, then ongoing)

The next-state prediction work (`pretrain_nextstate.py`) already proved entity-based world models work at short horizons. Repurpose as a Dyna augmenter.

### Use rl4burn's imagination infrastructure

rl4burn provides `imagine_rollout()` which takes an RSSM world model and an actor closure, producing `ImaginedTrajectory` with predicted rewards and continuation flags. It also provides `lambda_returns()` for computing targets on imagined data.

### World model architecture

Adapt the `EntityEncoderDecomposed` from `pretrain_nextstate.py`:

- **Input**: entity features (34-dim × 7 slots) + action taken (hybrid action encoding)
- **Output**: predicted next entity features (delta prediction)
- **Size**: ~500K params (d=64, 4 heads, 2 layers) — small to avoid overfitting
- **Training data**: real transitions from episode generation, accumulated continuously

### Dyna loop

After collecting each batch of real episodes:

1. Sample 1000 real states from rl4burn's `ReplayBuffer`
2. Use `imagine_rollout()` for 3-step rollouts (actor closure = current policy via hybrid adapter)
3. Compute `lambda_returns()` on imagined trajectories
4. Mix into training batch at 70% imagined / 30% real

**Rollout truncation**: ensemble of 3 world models (cheap at 500K each). Truncate when prediction disagreement > 0.5 L2.

### Why short rollouts suffice

The combat sim runs at 100ms fixed ticks. Meaningful tactical decisions play out over 5-15 ticks. A 3-step rollout at `step_interval=3` covers 9 real ticks — enough to evaluate ability usage outcomes.

---

## Stage 5 — Self-play bootstrap (1-3 days, then ongoing)

### Use rl4burn's league and self-play infrastructure

rl4burn provides exactly the components needed:

- **`SelfPlayPool`**: ring buffer of model checkpoints with `add_snapshot()` / `sample()`
- **`PfspMatchmaking`**: tracks per-opponent win rates, samples harder opponents proportionally
- **`League`**: orchestrates main agents + exploiters with role-based checkpointing

### Setup

```rust
let mut pool = SelfPlayPool::new();
let mut pfsp = PfspMatchmaking::new();

// Before each episode batch:
let opponent = match rng.random_range(0..100) {
    0..=4   => Policy::Combined,                    // 5% squad AI baseline
    5..=19  => pool.sample(&mut rng).unwrap(),      // 15% past checkpoint
    _       => current_model.clone(),                // 80% self-play
};

// After each episode:
pfsp.record_result(opponent_id, won, drew);

// Every 200K steps, if win rate > 55%:
pool.add_snapshot(&model, step);
pfsp.add_opponent(step);
```

Enemy policy runs on CPU via `NdArray` backend (already supported in `inference.rs`). Hero policy trains on GPU.

### Transition schedule

| Training step | Self-play % | Squad AI % | Past checkpoint % |
|---------------|-------------|------------|--------------------|
| 0–500K | 0% | 70% | 30% (from Stage 2 checkpoints) |
| 500K–1M | 30% | 40% | 30% |
| 1M–2M | 60% | 20% | 20% |
| 2M+ | 80% | 5% | 15% |

### Anti-collapse

- **KL anchor**: `β=0.05` penalty toward Stage 2d final policy throughout self-play
- **Entropy floor**: if policy entropy < 0.5 nats, double `entropy_coef` temporarily
- **PFSP weighting**: `pfsp.sample_opponent()` automatically up-weights opponents the agent loses to, preventing strategy cycling
- **Win rate monitoring**: if win rate vs buffer drops below 40%, revert to last good checkpoint

### Mini-exploiter (Stage 5b)

Use rl4burn's `League` with `AgentRole::MainExploiter`:

```rust
league.add_agent(fork_of_main, LeagueAgentConfig {
    role: AgentRole::MainExploiter,
    checkpoint_interval: 50_000,
    reset_threshold: 200_000,
});
```

Every 500K steps, the exploiter trains exclusively against the main agent's recent checkpoints. When it finds a weakness, it gets frozen into the `SelfPlayPool`. The `reset_threshold` reverts it to the main agent's current weights periodically so it doesn't drift into irrelevance.

---

## Stage 6 — Macro-controller for multi-room campaigns (future)

For the campaign layer (turn-based overworld → multi-room dungeons → combat), add a lightweight macro-controller:

- **Input**: room graph embedding, party state summary, dungeon progress
- **Output**: 15-20 discrete goals (enter room X, rest, use consumable, retreat)
- **Size**: ~400K params (2-layer MLP, 256→128)
- **Training**: rl4burn's `masked_ppo_update` (discrete goals are a simple masked action space that fits natively)

The macro-controller is the h-DQN pattern: meta-controller picks goals every room transition, V6 micro-executor handles 100ms tick decisions. Two levels only.

---

## What gets deleted

After the rl4burn migration in Stage 0, these handrolled components can be removed or reduced:

| Handrolled code | Replaced by | Lines saved (est.) |
|-----------------|-------------|-------------------|
| `training.rs` `compute_loss()` | rl4burn PPO loss + hybrid adapter | ~70 |
| `training.rs` `compute_bc_loss()` | `bc_loss_multi_head` + continuous MSE | ~40 |
| `training.rs` `train_step()` / `train_step_bc()` | Standard Burn backward + rl4burn grad clip | ~50 |
| `training.rs` `predict_values_and_logprobs()` | Hybrid adapter's `log_prob()` | ~80 |
| `training.rs` `rescore_replay_buffer()` | `actor_learner_collect` tracks log_probs natively | ~60 |
| `impala_train.rs` custom collection loop | `actor_learner_collect()` | ~100 |
| `impala_train.rs` custom trajectory packing | rl4burn `Trajectory` struct | ~40 |
| Self-play opponent management | `SelfPlayPool` + `PfspMatchmaking` | new code avoided |
| World model rollout loop | `imagine_rollout()` + `lambda_returns()` | new code avoided |

The only truly custom code that remains is:
1. **`HybridActorCritic` adapter** (~100 lines): maps V6's three-head output to rl4burn's interfaces
2. **`CombatEnv`** (~200 lines): implements `rl4burn::Env` for the tactical sim (already working)
3. **Multi-head loss weighting** (~20 lines): relative weights for position vs combat vs pointer losses
4. **Inverse-frequency BC weighting** (~10 lines): per-class weight computation from dataset statistics

---

## Timeline estimate

| Stage | Duration | Prerequisites |
|-------|----------|---------------|
| 0: rl4burn migration | 2-3 days | None |
| 1: Value alignment | 2-4 hours | Stage 0 |
| 2: Graduated unfreezing | 8-16 hours | Stage 1 |
| 3: Opponent diversity | 1 day engineering | Stage 0 |
| 4: World model | 4-8 hours training + 1 day integration | Stage 2 |
| 5: Self-play | 1-3 days training | Stage 2 |
| 6: Macro-controller | Future | Stage 5 stable |

Stages 3 and 4 can run in parallel with Stage 2. The critical path is 0 → 1 → 2 → 5.

---

## Key metrics to track

| Metric | Where | Target |
|--------|-------|--------|
| PPO KL divergence | TensorBoard | < 0.05 per update |
| Value prediction correlation | Validation episodes | > 0.6 |
| Policy entropy | TensorBoard | > 0.5 nats |
| HvH win rate (204 scenarios) | `rl_eval` | > 60% (Stage 2), > 75% (Stage 5) |
| Win rate vs squad AI (28 attrition) | `rl_eval` | > 90% (match combined v2 baseline) |
| Episode return variance | TensorBoard | Decreasing over training |
| World model ensemble disagreement | Custom logging | < 0.5 L2 for 3-step rollouts |
| `old_lp == recomputed_lp` assertion | Training loop | Always passes (Stage 0 guarantee) |
