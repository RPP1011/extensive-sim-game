# Monte Carlo Episode Scoring — Future Plan

## Context

The current ability evaluator uses rollout-based oracle scoring: clone state, try each action, simulate 10 ticks, score by outcome delta. This gives clean per-decision labels but has a 10-tick horizon — it can't capture delayed payoffs (setup combos, abilities that create kill windows 30+ ticks later).

## Approach: Policy Gradient / REINFORCE

Run full episodes where each hero uses a softmax-sampled policy (not always "best" — sometimes random). Record every ability decision + episode outcome. Then:

```
ability_value(state, ability) = E[outcome | used ability in state]
                              - E[outcome | didn't use ability in state]
```

### Training Data Generation

1. Run N complete episodes per scenario (target: 100K total)
2. Each hero samples from softmax over available abilities (temperature-controlled exploration)
3. Record per-tick: unit_id, ability_index, category features, action taken, state summary
4. Episode outcome: win/loss, total damage dealt, total damage taken, time to resolution

### Credit Assignment

Two options:

**Simple: Episode-level reward**
- Every ability decision in a winning episode gets reward +1, losing gets -1
- Noisy but simple. Needs ~50K+ episodes for signal.

**Better: Temporal difference with value baseline**
- Train a value function V(state) predicting episode outcome from any mid-game state
- Advantage = actual_outcome - V(state_when_decided)
- Much cleaner signal, needs fewer episodes (~10K)

### Advantage Over Current Oracle

| | Rollout Oracle | Monte Carlo |
|---|---|---|
| Horizon | 10 ticks | Full episode |
| Other units | Default AI (fixed) | Also exploring (realistic) |
| Credit assignment | Exact counterfactual | Statistical (noisy) |
| Cost per sample | ~10 rollouts × 10 ticks | 1 episode run (amortized) |
| Combo detection | Can't see setup plays | Naturally captures them |
| Data needed | ~50K samples sufficient | ~100K episodes for clean signal |

### Compute Estimate

- Sim throughput: ~100K ticks/sec single-threaded, ~500K with rayon
- 100K episodes × 500 ticks avg = 50M ticks
- Wall time: ~10-20 minutes on local machine (rayon across episodes)
- Training: seconds on RTX 4090 (same tiny models)

### Hybrid Approach (Recommended)

Use both signals:
1. Oracle urgency labels for immediate tactical value (keep existing)
2. MC episode value for strategic adjustment (new)
3. Combined: `urgency = alpha * oracle_urgency + (1-alpha) * mc_advantage`

This lets the oracle handle "CC the casting healer NOW" while MC handles "save this cooldown for the fight 20 ticks from now."

### Implementation Steps

1. Add exploration policy to squad AI (softmax temperature over ability evaluators)
2. Add episode recorder that logs (state_features, action, tick) per decision
3. Run 100K episodes across scenario set with exploration
4. Train value function V(state) on episode outcomes
5. Compute advantages, merge with oracle urgency labels
6. Retrain ability evaluators on combined signal

### GPU Sim Consideration

Not needed for this scale. CPU handles 100K episodes in minutes. GPU sim (Isaac Gym style) only becomes worth the engineering for continuous self-play RL (millions of episodes). If we go that route later, a simplified GPU kernel (fixed unit count, subset of effects) would give 100-1000x speedup for oracle rollouts.

### When to Pursue

After the current ability evaluator pipeline is validated end-to-end:
- Terrain-aware features trained and tested
- Evaluators integrated into runtime squad AI
- Win rate improvement measured on attrition scenarios

If we hit a ceiling where the 10-tick oracle can't capture important ability interactions, that's the signal to add MC scoring.
