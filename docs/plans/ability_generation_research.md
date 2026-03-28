# Ability Generation via Grammar-Constrained Flow Matching

## Problem Statement

We need a system that generates syntactically valid, semantically coherent game abilities from natural language descriptions. The abilities are expressed in a custom DSL with ~76 effect types, 15 targeting modes, 7 delivery methods, conditions, scaling, and targeting filters — covering both tactical combat and campaign-level interactions.

**Key constraint**: every generated ability must parse as a valid DSL program. No post-hoc filtering or rejection sampling.

## Architecture Overview

```
"devastating fire AoE with stun"
    → Text Encoder (6-layer transformer, d=256, 14K n-gram vocab)
    → 256-dim MRL embedding → 128-dim projection
    → Flow Model (conditioned denoiser, 50 Euler steps)
    → [0,1]^48 grammar space point
    → decode() → valid DSL program (100% parse rate, by construction)
```

### Key Insight: Grammar-Constrained Generation Space

Instead of generating DSL tokens directly (which requires learning syntax), we define an **invertible, deterministic mapping** between `[0,1]^48` and the space of valid DSL programs. Every point in the unit hypercube maps to exactly one valid ability. The generation model only needs to learn where *good* abilities live within this space — it cannot produce invalid output.

This sidesteps the fundamental challenge of neural program generation: guaranteeing syntactic validity.

#### Grammar Space Layout (48 dimensions)

| Dims | Controls | Type |
|------|----------|------|
| 0 | Active vs passive | Categorical (2 bins) |
| 1 | Combat vs campaign domain | Categorical (2 bins) |
| 2 | Targeting mode | Categorical (8-15 bins by domain) |
| 3 | Range | Continuous (0.5–10.0) |
| 4 | Cooldown | Log-continuous (1s–60s / 150s–30000s) |
| 5 | Cast time | Continuous (0–2000ms) |
| 6 | Hint category | Categorical (5-7 bins) |
| 7 | Resource cost | Continuous (0–30) |
| 8 | Delivery type | Categorical (7 bins, weighted toward none) |
| 9-10 | Delivery params | Continuous |
| 11 | Number of effects | Categorical (1-4) |
| 12-19 | Effect 0 (type, param, duration, area, tag, condition) | Mixed |
| 20-27 | Effect 1 | Mixed |
| 28-35 | Effect 2 | Mixed |
| 36-43 | Effect 3 | Mixed |
| 44-45 | Passive trigger + params | Mixed |
| 46-47 | Scaling stat + percentage | Mixed |

**Validation**: 10,000/10,000 random points decode to parseable DSL. The mapping is bijective — `encode(decode(v)) ≈ v`.

### Flow Matching (Conditional OT)

We use **conditional flow matching** (Lipman et al., 2023) rather than DDPM diffusion:

- **Forward**: interpolate between data `x_0` and noise `x_1`: `x_t = (1-t)x_0 + t·x_1`
- **Target**: velocity `u = x_1 - x_0` (straight-line optimal transport path)
- **Loss**: `||v_θ(x_t, t, cond) - u||²`
- **Sampling**: Euler integration from `t=1` (noise) to `t=0` (data), 50 steps

Advantages over DDPM:
- Simpler training (no noise schedule)
- Straight OT paths → fewer sampling steps needed
- More stable gradients

### Text Encoder

**Architecture**: 6-layer transformer encoder, d_model=256, 8 heads, d_ff=512

**Tokenizer**: Character n-gram (3-gram + 4-gram) + full words, 14.5K vocabulary. N-grams provide subword sharing: "flame", "blazing", "burning" share n-grams, giving similarity signal without co-occurrence training data.

**Training**:
- Phase 1: Pretrain on STS-B (8.6K sentence pairs) + TWI skill descriptions (1.1K pairs)
- Phase 2: Fine-tune on (ability_description, grammar_vector) pairs with InfoNCE + MRL loss

**Matryoshka Representation Learning (MRL)**: Loss computed at truncation dims [64, 128, 256], so any prefix of the embedding is useful. The flow model uses a 128-dim prefix via linear projection.

### Conditioning: Concatenation vs FiLM

**Current (concatenation)**: `input = [x_t, time_emb, text_emb]` → MLP → velocity

**Proposed (FiLM)**: Text embedding produces per-layer scale/shift parameters:
```
h = LayerNorm(Linear(x_t ++ time_emb))
gamma, beta = FiLM_proj(text_emb)
h = GELU(gamma * h + beta)
```

FiLM is the standard conditioning mechanism for diffusion models (Stable Diffusion uses a variant called AdaLN). It's more expressive than concatenation because the conditioning modulates the internal representations rather than just being mixed into the input.

## Dataset

| Source | Abilities | Description Pairs |
|--------|-----------|-------------------|
| Hand-crafted .ability files | 2,291 | 11,105 (LFM-generated) |
| Grammar-space quality-filtered | 10,400 | 52,345 (LFM-generated) |
| Template descriptions | — | ~7,000 |
| **Total** | **12,691** | **63,651** |

**Description generation**: LiquidAI LFM2.5-1.2B-Instruct via vLLM, 5 prompt styles per ability:
1. Technical/mechanical
2. RPG flavor text
3. Search keywords
4. Player guide (simple)
5. Comparison to archetypes

**Quality scoring heuristics** for grammar-space sampling:
- Coherence: heal shouldn't target enemy, campaign effects shouldn't use projectile delivery
- Balance: high damage → long cooldown, multi-effect → complex abilities
- Purpose: hint should match primary effect type
- Tag consistency: multiple effects should share element theme
- Variety bonus: reward underrepresented categories

## Experiments

### Completed

| Experiment | Config | Loss | Key Finding |
|------------|--------|------|-------------|
| Flow-only (no text) | 48d, one-hot labels, 200 steps | 0.053 | 100% parse, learns data distribution |
| Flow-only (50 Euler) | 48d, one-hot labels | 0.053 | Same quality, 4x fewer steps |
| E2E concat, small | 4-layer d=128 encoder, 64d cond | 0.051 | "healing"→ally correct, "trade"→region correct |
| E2E concat, big | 6-layer d=256 encoder, 128d cond | 0.047 | Better loss but similar eval quality |
| E2E + Grokfast EMA | α=0.98, λ=2.0 | 0.059 | Worse — λ too aggressive, gradients overshoot |
| FiLM conditioning | 6-layer d=256, FiLM layers | — | 1 epoch completed before interruption |

### Eval Results (Best: E2E concat big, epoch 300, loss 0.051)

| Prompt | Generated | Correct? |
|--------|-----------|----------|
| "fire damage AoE with stun" | target: ground, range: 7.4 | ✓ ground for AoE |
| "healing ally support" | target: ally, range: 7.1 | ✓ ally correct |
| "dark melee assassin strike" | passive on_damage_dealt | ✗ should be active melee |
| "army-wide leadership buff" | target: self | ~ partial |
| "passive on kill shield" | passive on_damage_dealt | ~ passive correct, wrong trigger |
| "devastating ice ultimate" | target: enemy, range: 3.9 | ✓ enemy, reasonable |
| "trade embargo" | target: region | ✓ campaign targeting |
| "quick ranged projectile" | target: self_aoe | ✗ wrong targeting |

**Parse rate**: 30/30 (100%) — grammar space guarantee holds.

### Proposed Experiments

1. **FiLM conditioning** (highest priority)
   - Replace concatenation with feature-wise linear modulation
   - Conditioning modulates each MLP layer via learned scale+shift
   - Expected: better fine-grained control (melee vs ranged, active vs passive)

2. **Auxiliary classification loss**
   - Add classification heads on text encoder: predict hint, element, targeting as categories
   - Multi-task loss: `L = L_flow + α·L_classify`
   - Forces encoder to extract semantically meaningful features

3. **Curriculum training**
   - Phase 1: Freeze text encoder, train flow model on one-hot hint labels (easy conditioning)
   - Phase 2: Unfreeze encoder, replace one-hot with text embeddings
   - Prevents early encoder instability from destabilizing flow model

4. **Separate learning rates**
   - Text encoder: 5e-5 (lower, harder task — language understanding)
   - Flow model: 3e-4 (higher, easier task — density estimation)
   - Currently both at 1e-4

5. **Grammar space expansion to 64 dims**
   - More room for flow model to disentangle features
   - Add explicit dims for: DoT vs instant, buff stat type, delivery speed

6. **Grokfast EMA with lower λ**
   - λ=2.0 was too aggressive
   - Try λ=0.5, λ=1.0 — gentler gradient amplification

## Technical Stack

- **Language**: Rust
- **ML Framework**: burn 0.20 with LibTorch backend (CUDA)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Training speed**: ~7K pairs/s, ~15s/epoch for 102K pairs
- **Description generation**: LFM2.5-1.2B via vLLM, ~60 abilities/s
- **Tokenizer**: 397-token DSL vocabulary + 14.5K n-gram text vocabulary
- **DSL Parser**: winnow-based, 76 effect types, roundtrip-tested

## Related Work

- **Grammar VAE** (Kusner et al., 2017): VAE over CFG production rules. Our grammar space is simpler — deterministic mapping, no learned decoder.
- **Flow Matching** (Lipman et al., 2023): Conditional OT for generative models. We apply it to a structured program space rather than images.
- **FiLM** (Perez et al., 2018): Feature-wise linear modulation for visual reasoning. We use it for text→ability conditioning.
- **Grokfast** (Lee et al., 2024): EMA gradient filter for accelerating grokking. Mixed results in our setting.
- **Static Embeddings** (Sentence Transformers, 2024): Bag-of-words with learned embeddings. We evolved past this to full transformer encoding.
- **MRL** (Kusupati et al., 2022): Matryoshka representation learning for flexible-dimension embeddings.

## Open Questions

1. Is 48 dims enough for the grammar space, or do we need 64-96 for fine-grained control?
2. Can FiLM conditioning close the gap on hard queries ("dark melee assassin strike")?
3. Would cross-attention (flow model attends to text encoder token outputs) outperform FiLM?
4. How much does data quality matter vs model architecture? (We have 63K LFM descriptions — is 200K better?)
5. Should the grammar space have tiered effect pools (low-level effects at low power_level, high-level at high)?
