# IMPALA V4 Training — Experiment Journal

## Context

Training V4 dual-head actor-critic (9-way movement + pointer combat) via IMPALA on 474 HvH scenarios (`dataset/scenarios/`). BC checkpoint starts at ~7% on dataset/scenarios/ and ~37% on hvh/ eval (204 scenarios).

**Best result so far:** 43.4% on dataset/scenarios/ at iter 2 (original simple config, 3 epochs, no regularization). Decays to 15% by iter 5 — every run peaks at iter 2-3 then crashes.

**Root cause:** 3 training epochs per iteration cause massive policy drift (KL reaches 7-14 by epoch 3). V-trace importance ratios become meaningless. Entropy explodes. Policy randomizes.

**What works:**
- Original simple IMPALA (no PPO clip, no KL, no per-traj norm) gets highest peak
- Per-trajectory advantage normalization extended improvement window (5 iters vs 3) but introduced survival bias
- Target entropy regularization controlled entropy but didn't prevent decline
- PPO clipping kept KL stable but capped peak (~30%)
- CPU-side PreTensorizedData avoids OOM on large datasets

**Key files:**
- `training/impala_learner.py` — main training loop
- `training/model.py` — AbilityActorCriticV4
- `src/bin/xtask/oracle_cmd/transformer_rl.rs` — episode generation
- `training/gpu_inference_server.py` — GPU shared memory inference

## Template Command

```bash
PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/impala_learner.py \
    --scenarios dataset/scenarios/ \
    --checkpoint generated/actor_critic_v4_full_unfrozen.pt \
    --output-dir generated/impala_E{N} \
    --embedding-registry generated/ability_embedding_registry.json \
    --external-cls-dim 128 \
    --threads 64 --sims-per-thread 64 \
    --episodes-per-scenario 5 \
    --gpu --iters 20 --temperature 1.0 --batch-size 1024 \
    --eval-scenarios scenarios/hvh/ --eval-every 10 \
    {EXPERIMENT-SPECIFIC FLAGS} \
    > /tmp/impala_E{N}.log 2>&1
```

---

## E1: Baseline reproduction (1 epoch)

**Hypothesis:** 1 epoch instead of 3 reduces per-iteration drift while keeping the same gradient signal.

**Flags:**
```
--train-epochs 1 --lr 1e-4 --value-coeff 0.5 --entropy-coeff 0.01
--reward-scale 1.0 --kl-coeff 0.0 --max-train-steps 0
```

**Expected:** Similar peak (~40%) but slower decay. Should plateau rather than crash.

**Code changes:** None (existing flags).

**Status:** COMPLETE

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| 1 | 6.4% | — | 1.698 | 1.99 | 0.0113 | BC baseline |
| 2 | **41.3%** | — | 2.068 | 1.11 | 0.0073 | **Peak** |
| 3 | 37.2% | — | 2.768 | 2.06 | 0.0030 | |
| 4 | 35.1% | — | 3.289 | 5.86 | 0.0011 | KL spike |
| 5 | 32.1% | — | 2.707 | 4.11 | 0.0023 | |
| 6 | 31.8% | — | 3.236 | 6.00 | 0.0014 | |
| 7 | 30.1% | — | 3.958 | 8.39 | 0.0007 | |
| 8 | 31.8% | — | 3.610 | 3.58 | 0.0045 | |
| 9 | 27.0% | — | 3.800 | 4.70 | 0.0027 | |
| 10 | 28.4% | **18.6%** | 3.781 | 4.51 | 0.0033 | |
| 11 | 28.7% | — | 3.811 | 4.72 | 0.0039 | |
| 12 | 29.0% | — | 3.655 | 4.64 | 0.0048 | |
| 13 | 30.1% | — | 3.653 | 4.67 | 0.0051 | |
| 14 | 28.8% | — | 3.729 | 4.62 | 0.0053 | |
| 15 | 30.6% | — | 3.655 | 3.51 | 0.0073 | |
| 16 | 30.2% | — | 3.793 | 4.89 | 0.0050 | |
| 17 | 29.7% | — | 3.779 | 4.72 | 0.0052 | |
| 18 | 30.2% | — | 3.923 | 4.93 | 0.0054 | |
| 19 | 30.3% | — | 4.092 | 5.14 | 0.0052 | |
| 20 | 32.6% | **27.9%** | 3.804 | 3.93 | 0.0074 | |

**Result:** Peak 41.3% at iter 2, then plateaus ~29-31% (iters 8-20). Does NOT crash to 15% like 3-epoch — **1 epoch prevents catastrophic collapse** but entropy still grows (1.7→4.1) and caps performance. HvH eval: 18.6% (iter 10) → 27.9% (iter 20). Compared to BC baseline 37% hvh, the policy is worse on eval even at plateau — entropy-driven exploration hurts on familiar scenarios.

---

## E2: Lower learning rate

**Hypothesis:** 1e-4 is too aggressive for 113K-param model.

**Flags:**
```
--train-epochs 1 --lr 3e-5 --value-coeff 0.5 --entropy-coeff 0.01
--reward-scale 1.0 --kl-coeff 0.0 --max-train-steps 0
```

**Expected:** Slower climb, higher sustained plateau.

**Code changes:** None (existing flags).

**Status:** COMPLETE

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| 1 | 6.5% | — | 1.146 | 0.76 | 0.0127 | |
| 2 | 17.6% | — | 2.187 | 1.40 | 0.0039 | Slower than E1 (41%) |
| 3 | 27.3% | — | 2.711 | 3.13 | 0.0032 | |
| 4 | **30.3%** | — | 2.861 | 3.60 | 0.0035 | **Peak** |
| 5 | 27.7% | — | 2.811 | 3.67 | 0.0038 | |
| 6 | 27.3% | — | 2.769 | 3.84 | 0.0041 | |
| 7 | 26.7% | — | 2.700 | 3.80 | 0.0044 | |
| 8 | 25.7% | — | 2.718 | 4.01 | 0.0045 | |
| 9 | 24.8% | — | 3.083 | 5.56 | 0.0026 | |
| 10 | 18.6% | **22.5%** | 3.118 | 6.05 | 0.0021 | |
| 12 | 15.4% | — | 3.084 | 6.51 | 0.0024 | |
| 15 | 15.2% | — | 3.046 | 6.71 | 0.0029 | Flat ~15% |
| 20 | 14.5% | **23.5%** | 3.565 | 6.94 | 0.0022 | |

**Result:** Worst experiment. Lower LR (3e-5) slows everything proportionally — peak only 30.3% (vs E1's 41.3%), then steady decline to ~15%. KL still reaches 6-7. No plateau, no recovery. Entropy still grows to 3.6. Lower LR alone reduces signal-to-noise ratio without fixing drift.

---

## E3: LR decay schedule

**Hypothesis:** High LR helps early exploration, low LR helps late convergence.

**Flags:**
```
--train-epochs 1 --lr 1e-4 --lr-decay cosine --lr-min 1e-5
--value-coeff 0.5 --entropy-coeff 0.01 --reward-scale 1.0
```

**Expected:** Best of both worlds — fast early, stable late.

**Code changes:** Done. `--lr-decay cosine --lr-min` → `CosineAnnealingLR(T_max=iters, eta_min=lr_min)`, steps once per iteration.

**Status:** COMPLETE

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | LR | Notes |
|------|---------------|-------------|---------|-----|------------|-----|-------|
| 1 | 6.9% | — | 1.678 | 1.99 | 0.0114 | 9.9e-5 | |
| 2 | **42.9%** | — | 2.057 | 1.13 | 0.0072 | 9.8e-5 | **Peak** |
| 4 | 31.0% | — | 3.234 | 5.91 | 0.0013 | 9.1e-5 | KL spike |
| 6 | 17.4% | — | 3.492 | 7.24 | 0.0012 | 8.2e-5 | Crash |
| 10 | 14.1% | **22.5%** | 3.858 | 7.07 | 0.0021 | 5.5e-5 | |
| 15 | 14.7% | — | 3.897 | 6.95 | 0.0026 | 2.3e-5 | Stuck |
| 20 | 15.7% | **21.1%** | 3.941 | 6.90 | 0.0028 | 1.0e-5 | |

**Result:** Worse than E1. Same peak (42.9%) but crashes to ~15% by iter 6 and never recovers despite LR decaying to 1e-5. KL locked at ~7 from iter 4 onward — reducing LR can't undo the damage already done. Entropy stays at ~3.9 even at LR=1e-5. **Cosine decay doesn't help because drift is irreversible; need to prevent it from happening in the first place.**

---

## E4: Entropy penalty (negative coefficient)

**Hypothesis:** Entropy growth is the direct cause of policy degradation. Negative entropy bonus keeps policy sharp.

**Flags:**
```
--train-epochs 1 --lr 1e-4 --entropy-coeff -0.005
--value-coeff 0.5 --reward-scale 1.0 --kl-coeff 0.0
```

**Expected:** Entropy stays near BC level (~0.9). May undertrain if too strong.

**Code changes:** None (entropy-coeff already accepts negative values).

**Status:** STOPPED EARLY (iter 10 — win rate 0.4%, entropy collapsed)

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| 1 | 6.7% | — | 1.537 | 1.71 | 0.0116 | BC baseline |
| 2 | **43.9%** | — | 1.827 | 0.74 | 0.0072 | **Peak** |
| 3 | 42.1% | — | 1.779 | 1.19 | 0.0044 | Entropy stable! |
| 4 | 38.9% | — | 1.554 | 2.16 | 0.0052 | |
| 5 | 37.3% | — | 2.091 | 2.71 | 0.0028 | |
| 6 | 36.5% | — | 1.841 | 5.48 | 0.0017 | KL spike |
| 7 | 27.7% | — | 1.212 | 5.56 | 0.0020 | Entropy collapsing |
| 8 | 24.1% | — | 0.346 | 7.74 | 0.0029 | Near-deterministic |
| 9 | 24.8% | — | 0.259 | 7.72 | 0.0026 | Collapsed |
| 10 | 0.4% | **4.9%** | 0.487 | 7.16 | 0.0002 | **Killed** — 114/204 timeouts |

**Result:** Entropy penalty controls entropy (stayed ~1.5-2.0 through iter 6, vs 3.0+ in E1) but **-0.005 is far too strong**. Policy collapsed to near-deterministic by iter 8 (ent=0.35), then win rate cratered. Iters 2-5 sustained >37% (best window of any run so far). A weaker penalty (-0.001 or -0.002) is promising — consider E4b variant. The mechanism works; the magnitude needs tuning.

---

## E5: Freeze policy, train value head only (warmup)

**Hypothesis:** V-trace advantages are garbage because value head is untrained (predicts ~0.2 everywhere). Training value head first gives meaningful advantages.

**Flags:**
```
# Phase 1: Freeze all except value head, 10 iterations
--train-epochs 3 --freeze-policy --warmup-iters 10
# Phase 2: Unfreeze everything
--train-epochs 1 --lr 3e-5
```

**Expected:** Better advantages → more stable RL.

**Code changes:** Done. `--freeze-policy` and `--warmup-iters N` flags. When frozen, `loss = value_coeff * value_loss` (policy_loss, entropy_loss, kl_penalty all dropped).

**Status:** STOPPED EARLY (iter 21 — declining to 14.8%, 3-epoch drift dominates)

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| 1 | 6.8% | — | 0.803 | 0.64 | 0.0083 | Frozen (warmup) |
| 5 | 7.2% | — | 0.803 | 0.64 | 0.0079 | vl converging |
| 10 | 6.8% | **3.4%** | 0.800 | 0.64 | 0.0074 | vl plateau; 89 timeouts |
| 11 | 6.5% | — | 1.974 | 2.47 | 0.0005 | **Unfrozen** |
| 12 | 25.1% | — | 2.655 | 7.46 | 0.0013 | Fast learning, KL=7 at ep3 |
| 13 | 24.8% | — | 1.755 | 5.55 | 0.0016 | |
| 14 | 28.7% | — | 1.031 | 3.56 | 0.0068 | |
| 15 | 25.7% | — | 1.256 | 2.83 | 0.0050 | |
| 16 | **34.6%** | — | 2.039 | 1.49 | 0.0024 | **Peak** |
| 17 | 27.5% | — | 2.510 | 4.01 | 0.0011 | |
| 18 | 19.2% | — | 3.201 | 4.91 | 0.0025 | Crashing |
| 19 | 14.4% | — | 3.438 | 6.29 | 0.0017 | |
| 20 | 14.9% | **26.5%** | 3.549 | 6.38 | 0.0021 | |
| 21 | 14.8% | — | — | — | — | **Killed** |

**Result:** Value-head warmup doesn't help. The warmed value head (vl 0.0074) partially forgot after unfreeze (dropped to 0.0005). Peak 34.6% at iter 16 — lower than E1's 41.3%. The 3-epoch training is the dominant problem: KL reaches 7+ at epoch 3, same drift pattern. Warmup can't compensate for per-iteration policy drift. **Conclusion: the value head isn't the bottleneck — epoch count and LR are.**

---

## E6: Advantage clipping

**Hypothesis:** A few extreme advantages drive most policy change. Clipping to [-3, 3] reduces outlier impact.

**Flags:**
```
--train-epochs 1 --lr 1e-4 --advantage-clip 3.0
--value-coeff 0.5 --entropy-coeff 0.01
```

**Expected:** Smoother updates, less variance between iterations.

**Code changes:** Done. `--advantage-clip C` → `np.clip(advantages, -C, C)` after global normalization.

**Status:** COMPLETE

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| 1 | 6.4% | — | 1.709 | 1.84 | 0.0116 | |
| 2 | **43.3%** | — | 2.181 | 1.09 | 0.0068 | **Peak** |
| 4 | 32.8% | — | 3.393 | 6.33 | 0.0011 | |
| 7 | 32.9% | — | 3.537 | 5.32 | 0.0034 | |
| 10 | 30.4% | **22.5%** | 3.483 | 4.48 | 0.0045 | |
| 13 | 30.9% | — | 3.566 | 4.50 | 0.0043 | |
| 16 | 25.5% | — | 4.000 | 5.21 | 0.0056 | Entropy >4 |
| 18 | 34.3% | — | 4.402 | 6.79 | 0.0038 | |
| 20 | 33.5% | **26.5%** | 3.812 | 5.13 | 0.0056 | |

**Result:** Nearly identical to E1. Peak 43.3% (iter 2), plateau ~30%, entropy grows to 3.5-4.4. Advantage clipping at [-3,3] has no effect — drift is not caused by advantage outliers. Entropy reaches 4.4 by iter 18 (higher than E1's 4.1). HvH eval 26.5% (E1: 27.9%). **Advantage outliers are not the problem.**

---

## E7: On-policy A2C (no V-trace)

**Hypothesis:** V-trace importance weighting causes more harm than good. With 1 epoch on fresh data, behavior policy IS current policy — ratios should be ~1.0 anyway.

**Flags:**
```
--train-epochs 1 --lr 1e-4 --no-vtrace
--value-coeff 0.5 --entropy-coeff 0.01
```

**Expected:** Simpler, possibly more stable. Same signal without IS noise.

**Code changes:** Done. `--no-vtrace` → Monte Carlo returns via backward accumulation, `advantages = returns - values`. Skips all importance ratio computation.

**Status:** COMPLETE

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| 1 | 6.2% | — | 1.471 | 1.31 | 0.0397 | High vl (MC variance) |
| 2 | 15.4% | — | 2.456 | 3.47 | 0.0417 | Slow climb |
| 3 | 23.7% | — | 2.646 | 4.43 | 0.0699 | |
| 4 | 25.5% | — | 2.739 | 4.60 | 0.0667 | |
| 5 | 25.9% | — | 3.125 | 6.69 | 0.0394 | |
| 6 | 26.9% | — | 2.566 | 10.19 | 0.0182 | KL=10 |
| 7 | 28.8% | — | 0.961 | 5.30 | 0.0176 | Entropy collapses |
| 8 | 20.6% | — | 1.638 | 5.84 | 0.0160 | Dip |
| 9 | 20.8% | — | 2.020 | 5.82 | 0.0181 | |
| 10 | 20.5% | **20.1%** | 2.181 | 7.09 | 0.0213 | |
| 11 | 36.4% | — | 1.044 | 0.70 | 0.0313 | **Recovery!** KL reset |
| 12 | **41.9%** | — | 2.065 | 0.77 | 0.0503 | **Peak** |
| 13 | 39.5% | — | 3.052 | 1.58 | 0.0580 | Entropy growing again |
| 14 | 35.2% | — | 3.539 | 2.73 | 0.0587 | |
| 15 | 32.1% | — | 3.593 | 3.73 | 0.0558 | |
| 16 | 32.3% | — | 3.458 | 4.24 | 0.0533 | |
| 17 | 31.5% | — | 3.386 | 4.21 | 0.0528 | |
| 18 | 33.4% | — | 3.253 | 3.42 | 0.0480 | |
| 19 | 33.4% | — | 3.302 | 3.46 | 0.0470 | |
| 20 | 33.8% | **27.0%** | 3.252 | 3.67 | 0.0455 | Plateau ~33% |

**Result:** Oscillation with self-recovery. Phase 1 (iters 1-10): slow climb to 29%, KL spike to 10, entropy collapse, dip to 20%. Phase 2 (iters 11-20): spontaneous recovery → peak 41.9% at iter 12, then standard entropy-driven decline to plateau ~33%. Final plateau higher than E1 (~30%). HvH eval 27.0% (E1: 27.9%). V-trace dampens the wild oscillation but A2C's self-correction is interesting. Value loss 10-20× higher than V-trace (MC return variance).

---

## E8: Polyak averaging (EMA weights)

**Hypothesis:** Each iteration's checkpoint is noisy. EMA provides smoother policy for episode generation.

**Flags:**
```
--train-epochs 1 --lr 1e-4 --polyak-tau 0.95
--value-coeff 0.5 --entropy-coeff 0.01
```

**Expected:** Less iteration-to-iteration variance, smoother win rate curve.

**Code changes:** Done. `--polyak-tau T` → `deepcopy(model)` at init, `ema = T*ema + (1-T)*param` after each iter. EMA state_dict used for checkpoint export → episode generation.

**Status:** COMPLETE

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| 1 | 6.8% | — | 1.682 | 1.93 | 0.0114 | |
| 2 | 8.0% | — | 1.893 | 2.46 | 0.0002 | EMA dampens update |
| 5 | 10.7% | — | 2.270 | 2.10 | 0.0023 | Slow climb |
| 8 | 15.5% | — | 2.379 | 3.22 | 0.0006 | |
| 10 | 21.8% | **17.6%** | 2.641 | 7.11 | 0.0002 | 73 timeouts |
| 12 | 31.3% | — | 2.653 | 9.86 | 0.0004 | |
| 14 | **36.1%** | — | 2.718 | 10.81 | 0.0006 | **Peak** |
| 16 | 34.0% | — | 2.923 | 11.67 | 0.0008 | KL peak |
| 18 | 33.7% | — | 2.644 | 9.78 | 0.0010 | KL declining |
| 20 | 33.2% | **14.7%** | 2.334 | 7.38 | 0.0010 | |

**Result:** EMA smoothing produces a unique trajectory: no iter-2 spike, slow monotonic climb to 36.1% (iter 14), stable plateau ~33%. Entropy stays controlled (2.3-2.9, best of all experiments). KL peaked at 11.7 then *declined* (EMA converging to training model). But massive off-policy gap cripples V-trace (near-zero value loss). HvH eval only 14.7% — worst of all. **EMA controls entropy but the generation/training mismatch undermines the whole pipeline.**

---

## Priority Order

All code changes implemented (2026-03-12). All experiments are runnable.

1. **E1** — 1-epoch baseline _(DONE — peak 41.3% iter 2, plateau ~30%, no crash)_
2. **E4** — entropy penalty _(DONE — -0.005 too strong, collapsed by iter 8; try -0.001)_
3. **E7** — on-policy A2C _(DONE — oscillation+recovery, peak 41.9%, plateau ~33%)_
4. **E5** — value head warmup _(DONE — peak 34.6%, worse than E1; 3-epoch drift dominates)_
5. **E2** — LR reduction _(DONE — peak 30.3%, worst result; LR alone doesn't fix drift)_
6. **E6** — advantage clipping _(DONE — identical to E1; clipping has no effect)_
7. **E3** — LR decay _(DONE — crashed to 15% by iter 6; decay can't undo early damage)_
8. **E8** — polyak averaging _(DONE — peak 36.1%, best entropy control, but hvh 14.7%)_

## Summary

| Exp | Peak Win% | Sustained Win% | Best Iter | Notes |
|-----|-----------|---------------|-----------|-------|
| BC baseline | 7% / 37% | — | — | dataset / hvh |
| Prior best | 43.4% | 15% (iter 5) | 2 | 3 epochs, crashes |
| E1 | 41.3% | ~30% | 2 | Plateau not crash; entropy 1.7→4.1; hvh eval 27.9% |
| E2 | 30.3% | ~15% | 4 | Worst result; LR 3e-5 too slow, still drifts |
| E3 | 42.9% | ~15% | 2 | Worse than E1; decay can't undo early drift |
| E4 | 43.9% | 0.4% (collapsed) | 2 | -0.005 too strong; entropy 1.5→0.26; try -0.001 |
| E5 | 34.6% | 14.8% (crashed) | 16 | Warmup didn't help; 3-epoch drift dominates |
| E6 | 43.3% | ~30% | 2 | Identical to E1; clipping doesn't help |
| E7 | 41.9% | ~33% | 12 | A2C oscillation; plateau higher than E1; hvh 27.0% |
| E8 | 36.1% | ~33% | 14 | Best entropy control (2.3-2.9); hvh eval 14.7% |

## Conclusions (2026-03-12)

**The core problem is entropy growth.** Every experiment shows the same pattern: entropy grows from BC level (~0.8-1.7) to 3-4+ within 5-10 iters, driven by the positive entropy coefficient (+0.01). This makes the policy increasingly random, degrading win rate.

**What doesn't matter:**
- Advantage clipping (E6) — identical to baseline
- Value head warmup (E5) — the value head isn't the bottleneck
- Lower LR alone (E2) — just slows everything proportionally
- LR decay (E3) — can't undo damage already done in early iters

**What partially works:**
- 1 epoch instead of 3 (E1) — prevents catastrophic crash, plateau ~30%
- Entropy penalty (E4) — controls entropy but -0.005 is too strong; iters 2-5 were the best sustained window of any run
- Polyak EMA (E8) — best entropy control (2.3-2.9) but cripples V-trace with huge off-policy gap

**Recommended next experiments:** see Round 2 below.

---

## Round 2 — Follow-up Experiments

### E4b: Weaker entropy penalty (-0.001)

**Hypothesis:** E4 showed entropy penalty works (sustained >37% for iters 2-5) but -0.005 collapsed the policy. -0.001 should hold entropy at ~1.5-2.0 without going deterministic.

**Flags:**
```
--train-epochs 1 --lr 1e-4 --entropy-coeff -0.001
--value-coeff 0.5 --reward-scale 1.0 --kl-coeff 0.0 --max-train-steps 0
```

**Status:** RUNNING (started 2026-03-12, log: /tmp/impala_E4b.log)

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| | | | | | | |

---

### E4c: Entropy penalty -0.002

**Hypothesis:** Midpoint between E4 (-0.005, collapsed) and E4b (-0.001). If -0.001 doesn't control entropy enough, -0.002 might be the sweet spot.

**Flags:**
```
--train-epochs 1 --lr 1e-4 --entropy-coeff -0.002
--value-coeff 0.5 --reward-scale 1.0 --kl-coeff 0.0 --max-train-steps 0
```

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| | | | | | | |

---

### E9: EMA + no-vtrace

**Hypothesis:** E8 (EMA) had best entropy control but crippled V-trace with massive off-policy gap. Removing V-trace avoids importance ratios entirely — EMA smoothing + A2C advantages.

**Flags:**
```
--train-epochs 1 --lr 1e-4 --polyak-tau 0.95 --no-vtrace
--value-coeff 0.5 --entropy-coeff 0.01 --max-train-steps 0 --kl-coeff 0.0
```

| Iter | Win% (dataset) | Win% (hvh) | Entropy | KL | Value Loss | Notes |
|------|---------------|-------------|---------|-----|------------|-------|
| | | | | | | |

---

### Round 2 Priority

1. **E4b** — weaker entropy penalty (-0.001), most promising
2. **E4c** — entropy penalty -0.002, backup if -0.001 too weak
3. **E9** — EMA + no-vtrace combo

### E4b: STOPPED — entropy collapsed to 0.62 by iter 6 (same failure as E4, just slower)

### E4c / E9: SKIPPED — entropy penalty approach is fundamentally unstable (feedback loop)

---

### EC1: Curriculum Training (BC init, vs default AI, pre-sim-fixes)

**Hypothesis:** Training on 474 scenarios at 7% win rate gives mostly negative reward signal. Start with easy scenarios (tier1, 37%+ win rate) to build a strong policy, then gradually expand.

**Phases:**
| Phase | Scenarios | Count | Iters | Checkpoint |
|-------|-----------|-------|-------|------------|
| 1 | tier1 | 108 | 10 | BC baseline |
| 2 | tier1+2 | 148 | 10 | phase1/best.pt |
| 3 | tier1-4 | 272 | 10 | phase2/best.pt |
| 4 | all | 474 | 10 | phase3/best.pt |

**Script:** `scripts/impala_curriculum.sh`
**Log:** `/tmp/impala_curriculum.log`
**Settings:** 1 epoch, lr=1e-4, entropy_coeff=0.01 (same as E1)

**Status:** CRASHED mid-Phase 3 iter 3 (GPU SHM /dev/shm/impala_inf vanished)

#### Phase 1 (tier1, 108 scenarios) — vs default AI

| Iter | Win% | Entropy | KL | Value Loss | Notes |
|------|------|---------|-----|------------|-------|
| 1 | 18.3% | 0.988 | 0.31 | 0.0106 | BC start, value head untrained |
| 2 | 17.8% | 1.675 | 0.97 | 0.0006 | |
| 3 | 17.8% | 1.246 | 1.49 | 0.0003 | |
| 4 | 18.5% | 0.771 | 2.37 | 0.0001 | |
| 5 | 19.1% | 0.931 | 3.92 | 0.0001 | |
| 6 | 16.7% | 1.317 | 6.40 | 0.0001 | |
| 7 | 18.9% | 1.558 | 8.92 | 0.0001 | KL diverging |
| 8 | 19.3% | 1.634 | 10.59 | 0.0001 | |
| 9 | 18.1% | 1.698 | 10.95 | 0.0001 | |
| 10 | 18.3% | 1.736 | 11.32 | 0.0001 | No learning, KL=11 |

Phase 1 conclusion: **No learning.** Win rate stuck at 17-19%, KL diverges to 11. Value head collapses to ~0 (vl=0.0001). Entropy grows from 0.99→1.74. The BC checkpoint starts at 18% on tier1 and never improves.

#### Phase 2 (tier1+2, 148 scenarios) — vs default AI

| Iter | Win% | Entropy | KL | Value Loss | Notes |
|------|------|---------|-----|------------|-------|
| 1 | 58.5% | 1.827 | 0.43 | 0.0032 | Fresh checkpoint reset, big jump |
| 2 | 29.5% | 1.928 | 6.86 | 0.0005 | Massive drop |
| 3 | 23.6% | 1.845 | 8.58 | 0.0005 | |
| 4 | 44.3% | 1.792 | 2.10 | 0.0005 | Recovery |
| 5 | 49.1% | 1.875 | 2.55 | 0.0011 | Peak |
| 6 | 42.0% | 1.914 | 5.75 | 0.0009 | |
| 7 | 45.4% | 1.999 | 5.13 | 0.0012 | |
| 8 | 46.4% | 2.048 | 4.61 | 0.0013 | |
| 9 | 47.3% | 2.046 | 4.79 | 0.0014 | |
| 10 | 46.6% | 2.026 | 5.00 | 0.0013 | Plateau 42-49% |

Phase 2 conclusion: **Best phase.** Iter 1 jumps to 58.5% (new scenarios give fresh signal), then oscillates 42-49%. KL unstable (2-9), entropy grows. Episodes get shorter over time (173K→67K steps), suggesting faster kills. But win rate relative to random baseline (~45%) means this is barely above random.

#### Phase 3 (tier1-4, 272 scenarios) — vs default AI, CRASHED

| Iter | Win% | Entropy | KL | Value Loss | Notes |
|------|------|---------|-----|------------|-------|
| 1 | 31.8% | 2.094 | 0.47 | 0.0039 | Drop from harder scenarios |
| 2 | 30.8% | 2.285 | 0.76 | 0.0025 | |
| 3 | 0.0% | 2.706 | 1.43 | 0.0013 | GPU SHM crash, trained on stale data |

Phase 3 conclusion: **Crashed.** GPU inference server died mid-iter 3. Before crash, win rate dropped to ~31% on the harder scenario set.

#### Phase 4 (all, 474 scenarios)

Not reached.

#### EC1 Key Findings

1. **Random baseline is ~45% on all tiers vs default AI** — the default AI is barely better than random
2. **Phase 2 peak of 49% is within noise of random** — no meaningful learning occurred
3. **KL diverges in every phase** — V-trace importance ratios become meaningless
4. **Value head collapses** — vl drops to 0.0001 in Phase 1, never recovers
5. **The entire curriculum experiment was training against default AI, which is nearly equivalent to random play for these scenarios** — the RL signal is fundamentally too weak

#### Sim Findings (during EC1 investigation)

- **Hold-only policy wins 41% vs default AI** on tier1 — squad AI leaks through casting, and default AI is weak
- **Random vs random (true self-play, no squad AI):** 49.3% — correct coin flip
- **Sim has first-mover advantage:** heroes always iterate first in unit array. Fixed with per-tick Fisher-Yates shuffle on unit processing order.
- **Squad AI `generate_intents` runs for BOTH teams** even in "transformer policy" mode — hero overrides only apply to non-casting units, letting squad AI intents leak through

---

### EC1b: Curriculum (BC init, vs default AI, post-sim-fixes, lr=1e-4)

Restarted EC1 after sim fixes (Fisher-Yates shuffle, team-specific policy, squad AI only for enemies). Still uses the BC-initialized checkpoint and lr=1e-4. Eval on scenarios/hvh/ (wrong — should eval on same tier).

**Status:** Phase 3 running (Phases 1-2 complete)

#### Phase 1 (tier1, 108 scenarios, 10 iters)

| Iter | Win% | Entropy | KL | Value Loss | Notes |
|------|------|---------|-----|------------|-------|
| 1 | 1.9% | 1.501 | 1.49 | 0.0017 | Near-zero — sim fixes work |
| 2 | 1.3% | 1.507 | 1.66 | 0.0000 | |
| 3 | 1.3% | 1.589 | 1.35 | 0.0007 | |
| 4 | 1.1% | 1.733 | 1.97 | 0.0000 | |
| 5 | 2.2% | 1.833 | 0.93 | 0.0002 | HvH eval: 36.8% (204 scenarios) |
| 6 | 1.1% | 2.034 | 1.05 | 0.0000 | |
| 7 | 1.9% | 1.818 | 1.66 | 0.0002 | |
| 8 | 32.0% | 2.543 | 1.61 | 0.0000 | Sudden jump — stale checkpoint? |
| 9 | 29.4% | 2.934 | 0.91 | 0.0001 | |
| 10 | 33.3% | 2.967 | 0.94 | 0.0001 | HvH eval: 6.4% |

Phase 1 notes: **Sim fixes validated** — BC checkpoint drops from 18% to 1-2% on tier1 (can no longer free-ride on squad AI leakage). Iters 8-10 jump to ~30% with entropy spike — likely the model discovered a degenerate action that happens to win some matchups. HvH eval: 36.8% at iter 5 (BC still intact for HvH), drops to 6.4% at iter 10 (catastrophic forgetting from tier1 training).

#### Phase 2 (tier1+2, 148 scenarios, 10 iters)

| Iter | Win% | Entropy | KL | Value Loss | Notes |
|------|------|---------|-----|------------|-------|
| 1 | 1.2% | 2.658 | 1.58 | 0.0003 | Back to near-zero |
| 2 | 1.4% | 2.660 | 2.67 | 0.0000 | |
| 3 | 27.4% | 3.024 | 1.28 | 0.0001 | Jump again |
| 4 | 23.2% | 3.103 | 0.67 | 0.0001 | |
| 5 | 23.4% | 3.078 | 0.77 | 0.0001 | HvH eval: 24.0% |
| 6 | 24.7% | 2.989 | 0.97 | 0.0001 | |
| 7 | 25.5% | 3.020 | 0.91 | 0.0002 | |
| 8 | 26.6% | 3.071 | 0.88 | 0.0001 | |
| 9 | 25.5% | 3.073 | 0.77 | 0.0001 | |
| 10 | 24.2% | 3.106 | 0.65 | 0.0001 | HvH eval: 2.9% |

Phase 2 notes: Starts at 1.2%, jumps to ~25% at iter 3 (same sudden-jump pattern as P1i8). Plateau 23-27% with KL stabilizing at 0.6-1.0. Entropy high at 3.0-3.1. Value loss near zero — value head not learning. HvH eval collapsed to 2.9% (catastrophic forgetting). This plateau is better than random (random gets 0% with sim fixes + 10x HP), so there IS learning happening.

#### Phase 3 (tier1-4, 272 scenarios, 10 iters)

| Iter | Win% | Entropy | KL | Value Loss | Notes |
|------|------|---------|-----|------------|-------|
| 1 | 34.6% | 3.351 | 0.65 | 0.0016 | Jump from new scenarios |
| 2 | 0.0% | 3.240 | 0.72 | 0.0003 | GPU SHM stale — 0 eps generated |
| 3 | 0.0% | 3.253 | 0.71 | 0.0009 | Stale data (retrained on same eps) |
| 4 | 0.0% | 3.273 | 0.76 | 0.0003 | Stale |
| 5 | 0.0% | 3.235 | 0.78 | 0.0007 | HvH eval: 27.5% |
| 6 | 21.6% | 2.654 | 1.93 | 0.0014 | GPU recovered, fresh episodes |
| 7 | 22.1% | 2.051 | 5.32 | 0.0004 | KL spike, entropy collapsing |
| 8 | 21.0% | 1.494 | 7.78 | 0.0002 | Entropy collapsed to 1.5 |
| 9 | 21.0% | 1.315 | 9.37 | 0.0001 | Entropy=1.3, KL=9.4 — degenerate |
| 10 | 20.7% | 1.424 | 10.37 | 0.0001 | KL=10.4 — fully collapsed |

Phase 3 notes: GPU inference server died on iters 2-5 (0 episodes generated, training on stale data from iter 1). Recovered at iter 6 with fresh episodes. But entropy collapsed from 3.4→1.3 in 5 iters — the lr=1e-4 drift problem. KL reached 10.4 by iter 10. Win rate plateaued at ~21%, policy fully degenerate. Uses OLD reward normalization (total_hp_start), not the new avg_unit_hp. HvH eval: 23.5%.

#### Phase 4 (all, 474 scenarios, 10 iters) — RUNNING

| Iter | Win% | Entropy | KL | Value Loss | Notes |
|------|------|---------|-----|------------|-------|
| 1 | 27.8% | 2.980 | 5.71 | 0.0025 | Inherited KL=5.7 from P3 |
| 2 | 25.3% | 1.976 | 5.13 | 0.0001 | Entropy crashed 3.0→2.0 |
| 3 | 25.7% | 1.449 | 6.21 | 0.0003 | ent=1.45 |
| 4 | 29.2% | 1.216 | 6.94 | 0.0003 | ent=1.2, near-deterministic |
| 5 | 30.3% | 1.265 | 7.88 | 0.0002 | HvH eval: 25.0% |
| 6 | 29.7% | 1.547 | 7.97 | 0.0003 | Slight entropy recovery |
| 7 | 27.3% | 1.548 | 8.50 | 0.0002 | |
| 8 | 28.6% | 1.353 | 10.91 | 0.0001 | KL>10 again |
| 9 | 28.6% | 1.087 | 12.50 | 0.0000 | ent=1.1, KL=12.5 |
| 10 | 27.8% | — | — | — | Best: 30.3% |

Phase 4 notes: Inherited degenerate policy from P3 (KL=10→5.7). Entropy collapsed further (3.0→1.1). KL exploded to 12.5 by iter 9. pg_loss at -3.4. Win rate stuck 25-30%, HvH eval 25%. Best checkpoint: 30.3% (iter 5).

**EC1b final: Dead.** lr=1e-4 + old reward normalization = catastrophic drift every phase. Best eval: 36.8% HvH (P1i5, still riding BC init). Curriculum completed Thu Mar 12 18:57.

**EC1b conclusion:** lr=1e-4 causes catastrophic KL drift in every phase (P1: oscillating, P2: plateau at ~25%, P3: KL=10 collapse, P4: inherited degenerate). The old reward normalization (total_hp_start) also gave vanishingly small signal (0.0001-0.0004).

---

### EC2: From-Scratch Training (random init, vs default AI, post-sim-fixes, lr=3e-5)

**New approach:** No BC/oracle data at all. Random-init all V4 weights (113K params). Keep only the behavioral embedding registry (external 128d CLS, pretrained separately). Sim fixes applied.

**Phases:** Same 4-tier curriculum but 20 iters each (more time for random init to learn). Eval on same training tier (not HvH).

**Script:** `scripts/impala_from_scratch.sh`
**Log:** `/tmp/impala_scratch.log`
**Settings:** 1 epoch, lr=3e-5, entropy_coeff=0.01
**Checkpoint:** `generated/actor_critic_v4_random_init.pt`

**Status:** Phase 2 running (iter 1/20)

**Key change from EC1b:** Added dense action-specific rewards (approach, engagement, hold penalty, attack bonus) + fixed reward normalization (avg_unit_hp instead of total_hp_start). This gives 15x stronger reward signal (0.006 vs 0.0004).

#### Phase 1 (tier1 autoattack, 108 scenarios, 20 iters) — COMPLETE

| Iter | Win% | Entropy | KL | Value Loss | Reward | Notes |
|------|------|---------|-----|------------|--------|-------|
| 1 | 26.7% | 3.229 | 0.10 | 0.0065 | 0.0062 | Random init baseline |
| 2 | 30.4% | 3.167 | 0.33 | 0.0017 | 0.0062 | |
| 3 | 29.4% | 3.060 | 0.58 | 0.0008 | 0.0061 | |
| 4 | 31.1% | 2.927 | 0.86 | 0.0006 | 0.0062 | |
| 5 | 29.1% | 2.786 | 1.18 | 0.0005 | 0.0061 | **Eval: 38.9%** (42W/51L/15T) |
| 6 | 31.9% | 2.667 | 1.53 | 0.0004 | 0.0063 | |
| 7 | 31.5% | 2.595 | 1.93 | 0.0004 | 0.0063 | |
| 8 | 29.6% | 2.558 | 2.38 | 0.0004 | 0.0062 | |
| 9 | 30.6% | 2.519 | 2.98 | 0.0004 | 0.0063 | |
| 10 | 30.6% | 2.520 | 3.39 | 0.0004 | 0.0063 | **Eval: 40.7%** (44W/48L/16T) |
| 11 | 31.9% | 2.476 | 4.23 | 0.0003 | 0.0063 | |
| 12 | 29.8% | 2.497 | 4.53 | 0.0004 | 0.0063 | |
| 13 | 31.1% | 2.489 | 4.87 | 0.0004 | 0.0062 | |
| 14 | 30.2% | 2.474 | 4.79 | 0.0004 | 0.0063 | KL plateau |
| 15 | 31.3% | 2.460 | 4.81 | 0.0004 | 0.0063 | **Eval: 38.9%** (42W/48L/18T) |
| 16 | 29.3% | 2.439 | 4.83 | 0.0004 | 0.0062 | |
| 17 | 26.7% | 2.388 | 4.89 | 0.0004 | 0.0062 | |
| 18 | 30.7% | 2.280 | 5.01 | 0.0003 | 0.0063 | |
| 19 | 30.6% | 2.159 | 5.24 | 0.0002 | 0.0062 | Entropy declining faster |
| 20 | 51.3% | 1.844 | 6.13 | 0.0001 | 0.0106 | **Eval: 51.9%** (56W/35L/17T) |

Phase 1 notes:
- **Final eval: 51.9%** — massive jump at iter 20! Training win rate spiked to 51.3% (from ~30%).
- Entropy dropped sharply at end: 2.16→1.84. The policy became more peaked and it paid off.
- Reward jumped to 0.0106 (vs 0.006 plateau) — the policy found something new.
- KL reached 6.1 by iter 20 — growing but manageable compared to EC1b's 10+.
- The late spike is encouraging: 20 iters of gradual refinement finally crossed a performance threshold.
- **Dense rewards confirmed essential.** Without them (original EC2): 0% greedy eval. With them: 51.9%.

#### Phase 2 (tier1+2, 148 scenarios, 20 iters) — COMPLETE

| Iter | Win% | Entropy | KL | Value Loss | Reward | Notes |
|------|------|---------|-----|------------|--------|-------|
| 1 | 34.2% | 2.001 | 0.37 | 0.0033 | 0.0218 | KL reset from new behavior policy |
| 2 | 36.5% | 1.783 | 0.72 | 0.0015 | 0.0219 | |
| 3 | 34.5% | 1.555 | 1.20 | 0.0009 | 0.0217 | |
| 4 | 39.5% | 1.396 | 1.80 | 0.0008 | 0.0218 | |
| 5 | 35.7% | 1.298 | 2.37 | 0.0007 | 0.0219 | **Eval: 45.3%** (67W/46L/35T) |
| 6 | 44.5% | 1.320 | 3.21 | 0.0008 | 0.0217 | Training win rate spike |
| 7 | 36.4% | 1.262 | 3.32 | 0.0007 | 0.0219 | |
| 8 | 42.7% | 1.279 | 3.85 | 0.0008 | 0.0217 | |
| 9 | 35.1% | 1.246 | 3.58 | 0.0008 | 0.0217 | |
| 10 | 35.3% | 1.249 | 3.60 | 0.0008 | 0.0217 | **Eval: 49.3%** (73W/47L/28T) |
| 11 | 36.9% | 1.262 | 3.61 | 0.0008 | 0.0219 | |
| 12 | 43.9% | 1.242 | 4.19 | 0.0007 | 0.0217 | |
| 13 | 36.6% | 1.259 | 3.59 | 0.0008 | 0.0218 | |
| 14 | 32.6% | 1.261 | 3.89 | 0.0008 | 0.0214 | |
| 15 | 43.0% | 1.413 | 3.99 | 0.0008 | 0.0217 | **Eval: 47.3%** (70W/49L/29T) |
| 16 | 36.5% | 1.596 | 3.58 | 0.0008 | 0.0218 | Entropy recovering! |
| 17 | 41.4% | 1.820 | 4.22 | 0.0007 | 0.0217 | |
| 18 | 35.9% | 1.931 | 3.64 | 0.0006 | 0.0219 | |
| 19 | 35.0% | 1.938 | 3.77 | 0.0004 | 0.0218 | |
| 20 | 35.5% | 1.906 | 3.94 | 0.0003 | 0.0218 | **Eval: 0.0%** (0W/59L/89T) |

Phase 2 notes:
- **Eval: 45.3% → 49.3% → 47.3% → 0.0%.** Catastrophic collapse at iter 20 eval.
- **Entropy did something unusual:** declined to 1.25 (iters 5-13), then *recovered* to 1.9 (iters 15-20). This is the opposite of EC1b's monotonic collapse. The policy became more exploratory again — but this may have randomized the greedy argmax.
- **0% eval with 89 timeouts** = the model's greedy action became hold/stay again. The entropy recovery randomized the policy enough that the argmax action is no longer attacking.
- **Training win rate stayed 35-43%** throughout — the stochastic policy still wins via exploration, but the deterministic eval collapsed.
- **KL stable at 3.5-4.0** throughout — no explosion. The policy drift was gradual entropy recovery, not KL blowup.
- **Best checkpoint: iter 10 (49.3% eval).** The "best.pt" saved by the script should be iter 6's 44.5% training win rate, not the eval peak.

#### Phase 3 (tier1-4, 272 scenarios, 20 iters) — COMPLETE

| Iter | Win% | Entropy | KL | Value Loss | Reward | Notes |
|------|------|---------|-----|------------|--------|-------|
| 1 | 28.7% | 1.170 | 0.10 | 0.0268 | 0.0207 | KL reset; value head reactivated |
| 2 | 28.2% | 1.102 | 0.44 | 0.0061 | 0.0208 | |
| 3 | 29.6% | 1.029 | 0.70 | 0.0020 | 0.0209 | |
| 4 | 29.7% | 1.076 | 0.97 | 0.0017 | 0.0209 | |
| 5 | 28.0% | 1.168 | 1.69 | 0.0011 | 0.0207 | **Eval: 39.0%** (106W/127L/39T) |
| 6 | 28.5% | 1.131 | 3.17 | 0.0006 | 0.0209 | KL jumped |
| 7 | 28.2% | 0.990 | 5.19 | 0.0003 | 0.0208 | Entropy < 1.0 |
| 8 | 28.5% | 0.913 | 7.11 | 0.0003 | 0.0208 | |
| 9 | 29.3% | 0.907 | 8.19 | 0.0003 | 0.0208 | |
| 10 | 28.2% | 0.906 | 9.24 | 0.0003 | 0.0209 | **Eval: 42.3%** (115W/114L/43T) |
| 11 | 29.0% | 0.848 | 9.73 | 0.0003 | 0.0209 | |
| 12 | 28.2% | 0.759 | 9.78 | 0.0003 | 0.0208 | Entropy floor = 0.76 |
| 13 | 27.9% | 0.756 | 9.78 | 0.0003 | 0.0208 | KL plateau at ~9.8 |
| 14 | 30.2% | 1.093 | 9.69 | 0.0003 | 0.0210 | Entropy recovering again! |
| 15 | 29.6% | 1.575 | 9.68 | 0.0003 | 0.0209 | **Eval: 40.1%** (109W/118L/45T) |
| 16 | 27.9% | 1.721 | 9.79 | 0.0003 | 0.0210 | |
| 17 | 28.7% | 1.799 | 9.60 | 0.0003 | 0.0209 | |
| 18 | 29.6% | 1.889 | 9.83 | 0.0003 | 0.0211 | |
| 19 | 29.8% | 1.767 | 9.73 | 0.0003 | 0.0208 | |
| 20 | 28.4% | 1.875 | 9.77 | 0.0003 | 0.0209 | **Eval: 35.3%** (96W/121L/55T) |

Phase 3 notes:
- **Eval: 39.0% → 42.3% → 40.1% → 35.3%.** Peaked at iter 10, then declined.
- **Same entropy oscillation as P2:** collapsed to 0.76 (iters 12-13), then spontaneously recovered to 1.9 (iters 14-20). This happened in P2 too (1.25→1.9) and preceded the eval collapse there.
- **KL saturated at ~9.7-9.8** from iter 11 onwards. Completely off-policy — V-trace corrections are meaningless at this KL.
- **Eval decline correlates with entropy recovery:** 42.3% at ent=0.91, down to 35.3% at ent=1.88. The deterministic argmax policy was better when the policy was peaked.
- **Training win rate never moved:** flat 28-30% across all 20 iters. Zero learning on the actual exploration policy.
- **Recurring pattern confirmed:** entropy collapses → eval improves (peaked argmax), entropy recovers → eval declines (argmax randomized). The model oscillates between these states rather than converging.

#### Phase 4 (all, 474 scenarios, 20 iters) — RUNNING (iter 11, generating)

| Iter | Win% | Entropy | KL | Value Loss | Reward | Notes |
|------|------|---------|-----|------------|--------|-------|
| 1 | 16.6% | 3.013 | 7.29 | 0.0012 | 0.0178 | 656 timeouts; entropy jumped 1.9→3.0 |
| 2 | 16.4% | 3.371 | 9.01 | 0.0001 | 0.0178 | |
| 3 | 15.5% | 3.230 | 11.44 | 0.0000 | 0.0178 | |
| 4 | 16.2% | 3.215 | 14.10 | 0.0000 | 0.0179 | |
| 5 | 16.4% | 3.525 | 14.52 | 0.0000 | 0.0178 | **Eval: 31.6%** (150W/268L/56T) |
| 6 | 16.2% | 3.744 | 14.50 | 0.0000 | 0.0178 | |
| 7 | 16.8% | 3.843 | 14.53 | 0.0000 | 0.0178 | |
| 8 | 16.6% | 4.112 | 14.69 | 0.0000 | 0.0178 | Entropy > 4.0 |
| 9 | 16.5% | 3.989 | 18.37 | 0.0000 | 0.0178 | KL broke plateau → 18 |
| 10 | 16.3% | 3.871 | 22.23 | 0.0000 | 0.0178 | **Eval: 33.5%** (159W/251L/64T); KL=22 |
| 11 | 16.6% | 3.886 | 22.27 | 0.0000 | 0.0179 | KL saturated at ~22.2 |
| 12 | 16.1% | 3.918 | 22.25 | 0.0000 | 0.0178 | |
| 13 | 16.4% | 3.961 | 22.23 | 0.0000 | 0.0177 | |

Phase 4 notes:
- **Eval: 31.6% (i5) → 33.5% (i10).** Greedy policy stable despite KL=22. Gradient updates are effectively no-ops — the checkpoint is frozen by V-trace clipping.
- **KL saturated at ~22.2** after iter 10. V-trace importance ratios are clipped so heavily that updates have zero effective learning rate. The training loop continues but changes nothing.
- **Entropy stable at ~3.9** — near-uniform over action space.
- **Training win rate locked at 16.1-16.8%** across all 13 iters. Zero learning.
- **Value loss = 0.0000** for 11 straight iters.
- 7 more iters of wasted compute. The checkpoint is effectively immutable.

### Round 2 Summary

| Exp | Peak Win% | Sustained Win% | Best Iter | Notes |
|-----|-----------|---------------|-----------|-------|
| E4b | 42.9% | collapsed | 2 | -0.001 still too strong |
| EC1 | 49.1% (P2i5) | ~46% | P2 i5-10 | Barely above random baseline (45%), pre-sim-fixes |
| EC1b | 34.6% (P3i1) | ~21% (P3) | P2 plateau | Post-sim-fixes; lr=1e-4 causes KL=10 collapse every phase |
| EC2 | **51.9% P1, 49.3% P2, 42.3% P3** | dead P4 (KL=14) | P3 i10 | Dense rewards help early phases; KL saturates ~10 then explodes in P4 |

**⚠ POST-MORTEM: ALL EC1/EC1b/EC2 results were invalidated by a critical SHM bug.**
GPU inference server weight reload used a 64-byte path buffer, but our paths are 68-73 chars.
The server NEVER successfully reloaded weights — it used the initial checkpoint for the entire phase.
All "KL drift" was actually cumulative divergence from a frozen behavior policy, not inherent V-trace instability.
Fix: `RELOAD_PATH_LEN = 64 → 256` in gpu_inference_server.py and impala_learner.py.

---

### EC3: From-Scratch Training (with GPU reload fix)

**Same setup as EC2** but with the SHM reload path fix. The GPU server now successfully reloads weights each iteration, making the behavior policy match the current policy (true on-policy IMPALA).

**Script:** `scripts/impala_from_scratch.sh`
**Log:** `/tmp/impala_scratch.log`
**Settings:** 1 epoch, lr=3e-5, entropy_coeff=0.01, dense rewards, avg_unit_hp normalization
**Checkpoint:** `generated/actor_critic_v4_random_init.pt`

**Status:** Phase 1 running (iter 5/20)

#### Phase 1 (tier1 autoattack, 108 scenarios, 20 iters) — RUNNING

| Iter | Win% | Entropy | KL | Value Loss | Reward | Notes |
|------|------|---------|-----|------------|--------|-------|
| 1 | 28.9% | 3.227 | 0.095 | 0.0065 | 0.0062 | Same random init baseline |
| 2 | 35.2% | 3.229 | 0.116 | 0.0070 | 0.0096 | GPU reload confirmed! |
| 3 | 38.7% | 3.171 | 0.138 | 0.0060 | 0.0127 | Win rate climbing |
| 4 | 38.3% | 3.085 | 0.156 | 0.0056 | 0.0154 | |
| 5 | 40.0% | 2.969 | 0.169 | 0.0052 | 0.0175 | **Eval: 41.7%** (45W/46L/17T) |
| 6 | 44.3% | 2.852 | 0.183 | 0.0056 | 0.0193 | |
| 7 | 45.2% | 2.729 | 0.181 | 0.0057 | 0.0204 | |
| 8 | 41.5% | — | — | — | — | Training in progress |

Phase 1 early notes:
- **GPU reload works.** Every iteration shows "Reloaded weights from ..." — first successful reload ever.
- **KL = 0.17 at iter 5** (was 1.18 in EC2). Proper on-policy: behavior policy matches current policy.
- **Value loss sustained at 0.005-0.007** (was collapsing to 0.0004 by iter 4 in EC2). Value head is actually training.
- **Training win rate CLIMBING: 29→35→39→38→40%** (was flat at ~30% in EC2). The model is improving its exploration policy, not just its argmax.
- **Reward climbing: 0.006→0.018** — policy getting better returns each iteration.
- **Entropy barely moving: 3.23→2.97** — no collapse (EC2 was at 2.79 by iter 5).
- **Eval 41.7% at iter 5** — already matches EC2's iter 5 (38.9%) and approaching EC2's peak (51.9% at iter 20). And we're only 5 iters in with stable KL.
