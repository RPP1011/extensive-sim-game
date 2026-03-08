# Per-Decision Credit Assignment via Actor-Critic with Entity Encoder Value Function

## Problem Statement

The current training pipeline relies on oracle rollouts to score ability decisions. This is expensive: each candidate ability requires running the simulation forward to completion to measure its impact. The result is coarse, episode-level credit assignment -- a decision at tick 50 gets the same outcome label as one at tick 300, even though the tick-50 decision may have been pivotal while the tick-300 one was irrelevant.

The entity encoder v2, pretrained on fight outcome prediction (99.1% win accuracy, 0.037 HP MAE), already functions as a value function: given a 210-dim game state, it outputs `P(hero_wins)`. This plan uses that value function for dense, per-decision temporal difference (TD) credit assignment in an actor-critic framework.

## 1. How TD Learning Works Here

### Value Function

The entity encoder serves as the critic. For any game state `s`:

```
V(s) = sigmoid(entity_encoder.win_head(pool(encode_entities(s))))
```

This is already trained and accurate. It maps the 210-dim state (7 entities x 30 features) to a win probability via self-attention over entity tokens.

### Per-Decision Advantage

When a hero unit makes an ability decision at tick `t`:

1. Extract game state `s_t` (210-dim) from the deciding unit's perspective
2. The actor (ability transformer + cross-attention) selects action `a_t`
3. The sim advances `k` ticks (one decision interval, likely 10 ticks / 100ms)
4. Extract game state `s_{t+k}`
5. Compute TD advantage:

```
A(s_t, a_t) = V(s_{t+k}) - V(s_t) + terminal_bonus
```

Where `terminal_bonus`:
- `+1.0 - V(s_t)` if hero team wins during the interval (reward the gap to certainty)
- `-V(s_t)` if hero team loses during the interval (penalize remaining confidence)
- `0` if fight continues

This is TD(0). The advantage is positive when the action improved the predicted win probability, negative when it worsened it. No discounting is needed because fights are short (typically 200-500 ticks) and the value function already accounts for temporal position via the features it sees (HP levels, cooldown states, cumulative damage).

### Why Not TD(lambda)?

TD(0) is the right starting point because:
- The value function is already very accurate (99.1%), so bias from bootstrapping is minimal
- Variance from single-step estimation is acceptable given the high V(s) accuracy
- Implementation simplicity matters for a first iteration
- TD(lambda) can be added later if variance is problematic (just accumulate eligibility traces in the episode buffer)

### Multi-Agent Consideration

Each hero unit has its own perspective in the 210-dim state (self slot = that unit). When unit A acts and unit B also acts in the same interval, the advantage for unit A includes the effect of unit B's action. This is acceptable because:
- The entity encoder was trained on full-state snapshots that implicitly include all agents
- Shared credit is fine for cooperative teams -- if ally coordination improves win probability, both units should be reinforced
- Individual contribution isolation would require counterfactual baselines (COMA), which is overkill for 3v3/4v4

## 2. Training Loop

### Episode Collection

```
for each scenario in training_scenarios:
    init sim from scenario TOML
    episode_buffer = []

    while fight not over:
        for each hero unit with a pending decision:
            s_t = extract_game_state(sim, unit)       # 210-dim
            V_t = critic.forward(s_t)                  # frozen entity encoder

            # Actor: ability transformer processes DSL tokens + cross-attends to entity tokens
            urgency, target_logits = actor.forward(ability_tokens, s_t)
            action = sample_action(urgency, target_logits)  # or argmax during eval

            episode_buffer.append({
                state: s_t,
                action: action,
                ability_tokens: tokens,
                V_t: V_t,
                unit_id: unit.id,
                tick: sim.tick,
            })

        advance sim by decision_interval ticks

        for each pending decision in episode_buffer where V_{t+k} not yet computed:
            s_{t+k} = extract_game_state(sim, unit)
            V_{t+k} = critic.forward(s_{t+k})
            decision.advantage = V_{t+k} - V_t + terminal_bonus_if_applicable

    # Terminal: label any remaining decisions
    for each decision without advantage:
        terminal_bonus = (+1 - V_t) if hero_wins else (-V_t)
        decision.advantage = terminal_bonus
```

### Policy Gradient Update

Using PPO-clip (simpler than full PPO, no KL penalty to tune):

```
for epoch in range(ppo_epochs):  # 3-4 epochs per batch
    for mini_batch in episode_buffer.shuffle().chunks(batch_size):
        # Re-evaluate actor on stored states
        new_urgency, new_target_logits = actor.forward(batch.ability_tokens, batch.state)

        # Log probability of the taken action under current policy
        log_prob_new = compute_log_prob(new_urgency, new_target_logits, batch.action)

        # Ratio for PPO clipping
        ratio = exp(log_prob_new - batch.log_prob_old)
        clipped = clamp(ratio, 1 - epsilon, 1 + epsilon)

        # Advantage is already computed, normalize per batch
        adv = (batch.advantage - mean) / (std + 1e-8)

        policy_loss = -min(ratio * adv, clipped * adv).mean()

        # Entropy bonus to encourage exploration
        entropy = compute_entropy(new_urgency, new_target_logits)

        loss = policy_loss - entropy_coeff * entropy
        loss.backward()
        optimizer.step()
```

### Action Parameterization

The actor outputs two things:
1. **Urgency** (sigmoid, 0-1): probability of using this ability vs. falling through to student model
2. **Target logits** (3-class softmax for unit-targeted abilities): which enemy/ally to target

The action is a tuple `(use_ability: bool, target_idx: int)`. Log probability:

```
if action.use_ability:
    log_prob = log(urgency) + log(softmax(target_logits)[target_idx])
else:
    log_prob = log(1 - urgency)
```

This naturally handles the 6%/94% ability-eval vs. student-model decision split from the current system.

## 3. What Needs to Be Built

### New Files

| File | Purpose |
|------|---------|
| `training/train_actor_critic.py` | Main training script: episode collection + PPO updates |
| `training/episode_runner.py` | Python wrapper around sim_bridge for collecting episodes with game state extraction |
| `training/advantage.py` | TD advantage computation, terminal bonus logic |

### Modifications to Existing Files

| File | Change |
|------|--------|
| `src/bin/sim_bridge/main.rs` | Add `game_state` emission mode: output 210-dim state vector alongside condensed state at each decision point. Add ability to receive and execute specific ability actions (not just personality/squad overrides). |
| `src/bin/sim_bridge/types.rs` | New message types: `GameStateMessage` (210-dim vector), `AbilityActionMessage` (unit_id, ability_idx, target_idx) |
| `training/model.py` | Add `log_prob()` and `entropy()` methods to `AbilityTransformerDecision`. Add value head option for future end-to-end critic training. |

### Sim Bridge Protocol Extension

Current protocol:
```
init -> state -> decision -> state -> decision -> ... -> done
```

Extended protocol for actor-critic:
```
init(mode="actor_critic") ->
  state_with_game_vec(tick, units, game_states: {unit_id: [210 floats]}) ->
  ability_action(unit_id, ability_idx, target_idx) ->
  state_with_game_vec(...) ->
  ...
  done(winner, final_game_states)
```

Key change: the sim bridge must emit the raw 210-dim game state vector per hero unit at each decision point, and accept specific ability actions (not just personality weight adjustments).

### Replay Buffer

A simple episode buffer is sufficient (not a full replay buffer):

- Collect full episodes (one fight = one episode, typically 200-500 decisions)
- Batch 8-16 episodes before each PPO update
- No experience replay across updates (on-policy PPO)
- Episode storage: ~500 decisions x (210 state + 256 tokens + 5 scalars) = ~300KB per episode
- 16 episodes = ~5MB per batch, easily fits in memory

GAE (Generalized Advantage Estimation) is not needed initially since we use TD(0) and the value function is already accurate.

## 4. Risks and Mitigations

### Risk 1: Value Function Accuracy Degrades Under New Policy

**Problem**: The entity encoder was trained on outcomes from the *default AI* policy. As the actor learns a better policy, the value function's predictions may become miscalibrated -- it might underestimate win probability for states the default AI would never reach.

**Severity**: Medium. The value function learned game state features (HP ratios, position, cooldowns), not policy-specific patterns. States reachable by a better policy are likely still well-characterized by these features.

**Mitigation**:
- Monitor V(s) calibration during training: bin predictions into deciles, check actual win rate per bin
- If calibration drifts beyond 5% per bin, fine-tune the critic on episodes from the current policy (periodic critic update, every N policy updates)
- The critic fine-tuning is cheap: same architecture, same loss, just new data from recent episodes

### Risk 2: Reward Sparsity at Decision Granularity

**Problem**: Most ability decisions produce tiny V(s') - V(s) differences (e.g., 0.001). The signal-to-noise ratio per decision may be too low for stable learning.

**Severity**: Medium-high. This is the biggest practical risk.

**Mitigation**:
- Normalize advantages per batch (already in the PPO loop above)
- Use the HP remaining head as a secondary reward signal: `A = w1*(V_win(s') - V_win(s)) + w2*(V_hp(s') - V_hp(s))`. The HP signal is denser because even small fights cause HP changes every few ticks.
- Increase decision interval from 10 ticks to 20-30 ticks for training (more state change per decision = clearer signal), while keeping 10-tick inference
- If still noisy, switch to TD(lambda=0.95) to propagate end-of-fight rewards backward more aggressively

### Risk 3: Exploration

**Problem**: PPO with a pretrained actor may converge to a local optimum near the current policy without exploring alternative ability usage patterns.

**Severity**: Low-medium. The current system already has good coverage (92.9% win rate on 28 scenarios), so the policy starting point is strong.

**Mitigation**:
- Entropy bonus (0.01-0.05 coefficient) in the policy loss
- Epsilon-greedy exploration layered on top: with probability epsilon, sample a random ability instead of the actor's choice
- Curriculum: start training on scenarios the current system loses or times out on (knight, shaman), where exploration is most valuable

### Risk 4: Sim Bridge Throughput

**Problem**: Each training episode requires running the Rust sim via subprocess (sim_bridge). Episode collection may bottleneck training.

**Severity**: Low. The sim runs at ~100K ticks/sec in headless mode. A 500-tick fight takes ~5ms. 16 episodes for a PPO batch takes ~80ms. Training the network is slower than data collection.

**Mitigation**:
- Run 8-16 sim_bridge processes in parallel (one per episode)
- Batch collection: collect all episodes first, then do PPO updates (already the plan)
- If subprocess overhead dominates, consider compiling the sim as a shared library callable from Python via ctypes/PyO3

### Risk 5: Multi-Agent Credit Assignment Noise

**Problem**: When multiple heroes act in the same interval, each unit's advantage includes the effects of other units' actions. This adds noise to individual credit assignment.

**Severity**: Low. Cooperative teams benefit from shared credit. The 3v3/4v4 scale makes this manageable.

**Mitigation**: Not needed for v1. If individual credit matters later, implement COMA (counterfactual multi-agent) baselines.

## 5. Complexity Estimate and Dependencies

### Phase 1: Sim Bridge Extension (2-3 days)
- Add `game_state` emission to sim_bridge (Rust)
- Add ability action input protocol (Rust)
- Test with manual Python client

**Dependencies**: None. Can start immediately.

### Phase 2: Episode Runner + Advantage Computation (2-3 days)
- Python episode runner wrapping sim_bridge subprocess
- TD advantage computation with terminal bonuses
- Parallel episode collection (multiprocessing)
- Verify advantage distribution looks reasonable on existing scenarios

**Dependencies**: Phase 1 (sim bridge extension).

### Phase 3: PPO Training Loop (3-4 days)
- Add `log_prob()` and `entropy()` to AbilityTransformerDecision
- PPO-clip implementation
- Training script with logging (CSV metrics, advantage histograms)
- Hyperparameter sweep: lr, epsilon, entropy_coeff, batch_size, ppo_epochs

**Dependencies**: Phase 2 (episode runner), existing pretrained entity encoder and ability transformer checkpoints.

### Phase 4: Evaluation + Critic Calibration (2-3 days)
- Run trained actor on 28-scenario benchmark
- Compare win rate vs. current combined system (92.9% baseline)
- Monitor value function calibration, implement critic fine-tuning if needed
- Ablation: TD(0) vs. TD(0.95), decision interval 10 vs. 20 vs. 30

**Dependencies**: Phase 3 (trained model).

### Total: 9-13 days

### Dependency Graph

```
Entity encoder pretrained (DONE, 99.1% acc)
    |
    v
Phase 1: Sim bridge extension --------------------------+
    |                                                    |
    v                                                    |
Phase 2: Episode runner + advantage                      |
    |                                                    |
    v                                                    |
Phase 3: PPO training loop  <-- Ability transformer      |
    |                           pretrained (DONE)        |
    v                                                    |
Phase 4: Evaluation  <-----------------------------------+
```

### Key Decision Points

1. **After Phase 2**: If advantage distributions are degenerate (all near zero), increase decision interval or add HP reward signal before proceeding to Phase 3.
2. **After Phase 3 initial training**: If win rate drops below 85% (current system is 92.9%), the value function may need critic fine-tuning. Implement periodic critic updates before further actor training.
3. **After Phase 4**: If the actor-critic system matches or exceeds 92.9% win rate, it validates dense credit assignment. The next step would be scaling to the full 3300+ generated scenario corpus for training diversity.
