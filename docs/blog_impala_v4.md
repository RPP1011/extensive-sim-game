# Teaching a Neural Network to Fight: IMPALA on a Tactical Combat Sim

*How a 113K-parameter actor-critic learns to command squads of heroes with 943 unique abilities — and how a 4-byte bug hid for weeks inside a shared-memory inference server.*

---

## The Problem

We have a deterministic tactical combat simulator. Two teams of heroes fight on a grid. Each hero has up to 8 abilities drawn from a pool of **943 unique abilities** — heals, stuns, dashes, AoE damage, shields, channeled beams, traps, you name it. Every ability is defined in a custom DSL:

```
ability Fireball {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage
    damage 55 [FIRE: 60]
    in circle(3.0)
}

ability Divine_Shield {
    target: self
    cooldown: 30s, cast: instant
    hint: defensive
    shield 500 duration 4s
    immune [STUN, SLOW, ROOT] duration 4s
}
```

The action space is heterogeneous: movement is a 9-way directional choice, but combat involves selecting both an **action type** (attack, hold, or use ability 0-7) and a **target entity** via pointer attention. The number of valid abilities changes per hero, per tick.

The goal: train a single policy network that commands any hero composition against any enemy composition, from random initialization, using only RL.

---

## Architecture

The core challenge is representing 943 abilities in a way that generalizes. We can't use a flat one-hot — the action space would be enormous and sparse. Instead, we decompose the problem into pretrained components and a lightweight policy head:

```
                    ┌─────────────────────────────────────────────┐
                    │         PRETRAINED (frozen in RL)           │
                    │                                             │
  Ability DSL ───► │  Ability Transformer  ───►  [CLS] tokens    │
  (text tokens)    │  d=128, 4 layers, 4 heads                   │
                    │  252-token vocab                             │
                    └──────────────────┬──────────────────────────┘
                                       │
                              943 × 128d embeddings
                            (cached in registry JSON)
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │          LEARNED DURING RL (113K params)     │
                    │                                              │
                    │  ┌──────────┐     ┌─────────────────────┐   │
                    │  │ CLS Proj │     │   Entity Encoder    │   │
                    │  │ 128→32d  │     │   d=32, 4 layers    │   │
                    │  └────┬─────┘     │   7 slots × 30 feat │   │
                    │       │           │   + self-attention   │   │
                    │       │           └──────────┬──────────┘   │
                    │       │                      │              │
                    │       ▼                      ▼              │
                    │  ┌────────────────────────────────┐         │
                    │  │    Cross-Attention Block        │         │
                    │  │    ability queries × entity keys│         │
                    │  │    4 heads, pre-norm, FF resid  │         │
                    │  └───────────────┬────────────────┘         │
                    │                  │                           │
                    │          ┌───────┴───────┐                  │
                    │          ▼               ▼                  │
                    │   ┌────────────┐  ┌──────────────┐          │
                    │   │  Move Head  │  │ Combat Head  │          │
                    │   │  9-way dir  │  │ type + ptr   │          │
                    │   │  softmax    │  │ dot-product  │          │
                    │   └────────────┘  │ over entities │          │
                    │                   └──────────────┘          │
                    │          ▲                                   │
                    │   ┌────────────┐                             │
                    │   │ Value Head │  (not exported to Rust)     │
                    │   │ state → R  │                             │
                    │   └────────────┘                             │
                    └─────────────────────────────────────────────┘
```

**Why this decomposition?** The ability transformer is expensive (~2.8M params, 4-layer encoder over token sequences). But abilities don't change mid-fight — a hero's kit is fixed at spawn. So we **pre-compute** the [CLS] embedding for every ability once at fight start, cache them in a registry, and only the lightweight 113K-param policy runs per tick.

### Entity Representation

Each game tick, we extract a fixed-size state vector from the simulator:

```
Entity slot layout (7 slots × 30 features = 210 dims):

Slot 0: Self       ─┐
Slot 1: Enemy 0     │  Each slot: ┌─────────────────────────────────┐
Slot 2: Enemy 1     │             │ vitals(5)    │ hp, hp%, shield, │
Slot 3: Enemy 2     ├────────────►│              │ mana, threat     │
Slot 4: Ally 0      │             │ position(7)  │ xy, dist, angle, │
Slot 5: Ally 1      │             │              │ terrain, LOS     │
Slot 6: Ally 2     ─┘             │ combat(3)    │ dps, burst, atk% │
                                  │ ability(3)   │ cd_ready, cd_avg │
                                  │ healing(3)   │ hps, can_heal    │
                                  │ CC(3)        │ stunned, slowed  │
                                  │ state(4)     │ casting, moving  │
                                  │ cumul(2)     │ dmg_dealt, taken │
                                  └─────────────────────────────────┘
```

The entity encoder applies self-attention across all 7 slots, letting the model reason about team compositions — e.g., "I'm a healer, my ally is low HP, the enemy has a stun ready."

### Pointer-Based Target Selection

The combat head uses **scaled dot-product attention** as a pointer mechanism:

```
Query: pooled entity state (d=32)
Keys:  per-entity encodings (7 × d=32)

attack_logits = (Q · Kᵀ) / √d    →  softmax over enemy entities
ability_logits[i] = (Q_ability_i · Kᵀ) / √d   →  per-ability target
```

This lets the model generalize across different numbers of enemies and different ability target types (single-target heal vs AoE damage vs self-buff).

---

## Behavioral Embeddings: Understanding Abilities from Source Code

The 128-dimensional ability embeddings aren't just MLM features — they encode **behavioral semantics**. We train them in a two-phase curriculum:

```
Phase 1: Masked Language Modeling          Phase 2: Behavioral Prediction
┌──────────────────────────┐              ┌──────────────────────────┐
│                          │              │                          │
│  "damage 55 [FIRE: 60]" │              │  Run 135,792 controlled  │
│  "in circle(3.0)"       │              │  sim experiments:        │
│  "cooldown 5s"           │              │  943 abilities ×         │
│       ↓                  │              │  144 conditions          │
│  Mask 15% of tokens      │              │  (HP × dist × targets)   │
│  Predict masked tokens   │              │       ↓                  │
│                          │              │  119-dim outcome vector  │
│  97.5% accuracy          │              │  (dmg, heal, CC, etc.)   │
│                          │              │                          │
│  Result: syntactic        │              │  Result: semantic        │
│  understanding           │              │  understanding           │
└──────────────────────────┘              └──────────────────────────┘
                    │                                    │
                    └──────────┬─────────────────────────┘
                               ▼
                    943 abilities × 128d
                    kNN@5: 99.4% (similar abilities cluster)
                    Behavioral MSE: 0.01 (z-normed)
```

**Key insight:** Joint training (MLM + behavioral) destroys MLM performance. The solution is curriculum: MLM first to learn syntax, then freeze the encoder and train only a behavioral prediction head. The [CLS] embeddings from this frozen encoder capture both syntactic structure (what tokens appear) and behavioral semantics (what the ability actually does in the sim).

The registry is a JSON file mapping ability names to 128d vectors. At episode generation time, Rust looks up each hero's abilities and passes the embeddings through shared memory alongside the game state.

---

## Training Infrastructure: 4,096 Parallel Sims via Shared Memory

IMPALA needs massive throughput: each iteration generates thousands of episodes, trains one gradient step, then generates again. The bottleneck is inference — every sim tick, every hero needs a forward pass to choose actions.

We solve this with a **shared-memory GPU inference server**:

```
                     ┌─────────────────────────────────┐
                     │        GPU Server (Python)       │
                     │   PyTorch model on CUDA          │
                     │   Polls SHM for batch requests   │
                     │   ~25,000 inferences/sec         │
                     └──────────┬──────────────────────┘
                                │
                          /dev/shm/impala_inf
                          (memory-mapped file)
                                │
     ┌──────────────────────────┼──────────────────────────┐
     │           512-byte Header                           │
     │  ┌────────┬─────────┬──────────┬───────────────┐    │
     │  │ magic  │ cls_dim │ flag     │ reload_path   │    │
     │  │ 4 bytes│ 4 bytes │ 0/1/2    │ 256 bytes     │    │
     │  └────────┴─────────┴──────────┴───────────────┘    │
     │                                                      │
     │  Request region: 1024 samples × ~7KB each = ~7MB    │
     │  Response region: 1024 × 16 bytes = 16KB             │
     └──────────────────────────────────────────────────────┘
                                │
     ┌──────────────────────────┼──────────────────────────┐
     │               Rust Sim (64 threads)                  │
     │                                                      │
     │  Thread 0: ┌─sim─┐┌─sim─┐┌─sim─┐ ... ┌─sim─┐      │
     │            │  0  ││  1  ││  2  │     │ 63  │      │
     │  Thread 1: ├─sim─┤├─sim─┤├─sim─┤     ├─sim─┤      │
     │            │ 64  ││ 65  ││ 66  │     │127  │      │
     │  ...       └─────┘└─────┘└─────┘     └─────┘      │
     │  Thread 63:           (64 × 64 = 4,096 parallel)    │
     │                                                      │
     │  Batcher thread collects requests via crossbeam      │
     │  channels, writes to SHM, polls for response         │
     └─────────────────────────────────────────────────────┘
```

**Protocol:**
1. Rust sim threads submit `InferenceRequest` structs to a batcher via lock-free channels
2. Batcher serializes up to 1024 requests into the SHM request region
3. Sets `flag = 1` (request ready)
4. GPU server reads batch, runs forward pass, writes responses, sets `flag = 2`
5. Batcher reads responses, dispatches to waiting threads via condvar

**Response format** (16 bytes, zero-copy):
```
┌──────────┬──────────────┬───────────┬──────────┬───────────┬────────────┐
│ move_dir │ combat_type  │ target_id │ lp_move  │ lp_combat │ lp_pointer │
│  u8      │  u8          │  u16 LE   │  f32 LE  │  f32 LE   │  f32 LE    │
└──────────┴──────────────┴───────────┴──────────┴───────────┴────────────┘
```

The log probabilities travel back with the action — they're stored in the episode data and used later for V-trace importance ratio computation.

**Throughput:** ~25,000 inferences/sec sustained, enough to generate 540 episodes across 108 scenarios in ~14 seconds.

---

## The Training Loop: IMPALA with V-Trace

We use [IMPALA](https://arxiv.org/abs/1802.01561) (Importance Weighted Actor-Learner Architecture) adapted for our dual-head action space:

```
┌─────────────────────────────────────────────────────────┐
│                   IMPALA Iteration                        │
│                                                           │
│  1. Export current weights → current.pt                   │
│  2. Reload GPU server with new weights  ◄── THE FIX      │
│  3. Generate 540 episodes (4,096 parallel sims)           │
│  4. Compute V-trace targets:                              │
│     ρ = clip(π_current(a|s) / π_behavior(a|s), 0, 1)    │
│     Advantage = R + γV(s') - V(s), weighted by ρ         │
│  5. One epoch of gradient descent                         │
│     Loss = -ρ·A·log π(a|s) + 0.5·(V-target)² + 0.01·H  │
│  6. Repeat                                                │
│                                                           │
│  Dual-head V-trace:                                       │
│    log_rho = lp_move_curr - lp_move_behav                │
│            + lp_combat_curr - lp_combat_behav             │
│    (pointer log-probs excluded from ratio — too noisy)    │
└─────────────────────────────────────────────────────────┘
```

### Dense Reward Shaping

Win/loss alone is too sparse for 113K params to learn from. We add per-tick dense rewards:

| Reward Component | Signal | Scale |
|-----------------|--------|-------|
| HP differential | `(enemy_dmg - hero_dmg) / avg_unit_hp` | 1.0 |
| Kill bonus | hero kills an enemy | +0.5 |
| Death penalty | hero dies | -0.5 |
| Approach reward | closing distance to nearest enemy | 0.002/unit |
| Engagement bonus | being within attack range | +0.01/tick |
| Hold penalty | holding position near enemies | -0.02/tick |
| Attack bonus | attacking or using abilities | +0.01/tick |

### Curriculum

Training proceeds in 4 phases of increasing difficulty:

```
Phase 1: Autoattack Only (108 scenarios)
  └─ 10× HP, no abilities, pure movement + targeting
Phase 2: One Ability (148 scenarios)
  └─ Tier 1-2, simple kits (1-2 abilities per hero)
Phase 3: Full Kits (272 scenarios)
  └─ Tier 1-4, complex compositions
Phase 4: Everything (474 scenarios)
  └─ All scenarios including stress tests
```

Each phase runs 20 iterations, starting from the previous phase's best checkpoint.

---

## The Bug: 64 Bytes of Chaos

After weeks of experiments, every run showed the same pattern: KL divergence between the behavior policy and current policy would grow monotonically until the importance ratios became meaningless, typically reaching 10-22 within 10-15 iterations. We attributed this to fundamental V-trace instability and tried everything — learning rate reduction, entropy penalties, PPO clipping, advantage normalization, value head warmup. Nothing worked.

```
EC2 Phase 3 KL trajectory (BROKEN):
  Iter 1:  0.10  ←  fresh phase, KL resets
  Iter 5:  1.69
  Iter 7:  5.19  ←  V-trace corrections becoming meaningless
  Iter 10: 9.24
  Iter 13: 9.78  ←  saturated, training is pointless
```

Then, while implementing a "policy resync" feature, we noticed error messages in the logs that had been there all along:

```
[gpu] Failed to reload: [Errno 2] No such file or directory:
  '/home/ricky/Projects/game/generated/impala_scratch/phase1/curre'
```

`curre`. Not `current.pt`. **The path was truncated.**

The shared memory protocol allocated **64 bytes** for the reload path:

```c
OFF_RELOAD_PATH = 0x80   // offset in SHM header
RELOAD_PATH_LEN = 64     // bytes allocated
OFF_RELOAD_REQ  = 0xC0   // next field starts right after
```

Our checkpoint paths were 68-73 characters:

```
/home/ricky/Projects/game/generated/impala_scratch/phase1/current.pt
└──────────────────────── 68 characters ──────────────────────────┘
```

The Python code dutifully truncated to fit:
```python
path_bytes = abs_path.encode('utf-8')[:RELOAD_PATH_LEN - 1] + b'\x00'
```

The GPU server tried to open the truncated path, failed, printed a warning, and **continued serving with the initial checkpoint weights**. The training loop received an ACK timeout but treated it as non-fatal.

**The behavior policy never updated.** Every experiment — every phase, every iteration — generated episodes using the **initial checkpoint** while the training policy diverged further and further. The "KL drift" wasn't V-trace instability. It was the cumulative divergence between a frozen behavior policy and an ever-changing trained policy.

### The Fix

Three lines:

```python
# gpu_inference_server.py and impala_learner.py
RELOAD_PATH_LEN = 256      # was 64
OFF_RELOAD_REQ  = 0x180    # was 0xC0
OFF_RELOAD_ACK  = 0x184    # was 0xC4
```

### The Result

```
EC3 Phase 1 KL trajectory (FIXED):
  Iter 1:  0.095
  Iter 2:  0.116
  Iter 3:  0.138
  Iter 4:  0.156
  Iter 5:  0.169  ←  KL stays near zero (proper on-policy)
  Iter 6:  0.183
  Iter 7:  0.181
```

Every metric changed overnight:

| Metric (iter 7) | Broken (EC2) | Fixed (EC3) |
|-----------------|-------------|-------------|
| KL divergence | 1.93 | **0.18** |
| Value loss | 0.0004 (dead) | **0.0057** (learning) |
| Training win% | 31% (flat) | **45%** (climbing) |
| Mean reward | 0.006 (flat) | **0.020** (climbing) |
| Entropy | 2.60 (collapsing) | **2.73** (stable) |

The value head — which had collapsed to zero in every previous experiment within 3-4 iterations — is now **sustained at 0.005-0.007** after 7 iterations. It's actually learning state values because the V-trace advantages are computed against the correct behavior policy.

Training win rate is **climbing monotonically** (29→35→39→38→40→44→45%) instead of flatting at ~30%. The model is improving its exploration policy, not just getting lucky with its argmax.

---

## What We Learned

**1. Off-policy RL is fragile in ways that look like algorithmic failures.** We spent weeks trying regularization techniques, learning rate schedules, and architectural changes to fix "V-trace instability" that was actually a plumbing bug. The symptoms (KL drift, value collapse, entropy oscillation) perfectly mimicked known failure modes of off-policy methods.

**2. Always verify your data pipeline end-to-end.** The weight reload was tested when the paths were short. It broke when we moved to a deeper directory structure. The truncation was silent (the server continued serving), the error message was buried in GPU server logs mixed with batch throughput stats, and the training loop treated the ACK timeout as non-fatal.

**3. Dense rewards are necessary but not sufficient.** Without dense per-step rewards, the value head has nothing to learn from and collapses immediately. With them, it sustains learning — but only if the behavior policy actually matches the current policy so V-trace corrections are meaningful.

**4. Behavioral embeddings bridge the sim-to-policy gap.** The pretrained 128d ability embeddings let a 113K-param policy network handle 943 unique abilities without ever seeing most of them during training. The curriculum (MLM → behavioral prediction) is critical — joint training destroys both objectives.

**5. Shared-memory GPU inference makes IMPALA practical for complex environments.** 4,096 parallel sims generating ~25K inferences/sec lets us run 20 IMPALA iterations in ~15 minutes on a single GPU. The serialization overhead is minimal compared to the forward pass.

---

*Training is still running. With proper weight reload, we're seeing the first sustained learning curve in the project's history. Phase 1 is approaching 50% win rate at iteration 7 with no sign of the KL drift that killed every previous run. The curriculum has 3 more phases to go.*
