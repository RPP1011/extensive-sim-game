# Why Training Is Slow: GPU Underutilization Analysis

## The Numbers
- Model: **49K params**, d_model=32, 4 heads (d_k=8), 4 layers
- Sequence: **176 tokens** (16 entity + 8 threat + 8 position + 144 ability)
- Throughput: **20 steps/sec** (baseline) → **72 steps/sec** (compile+AMP)
- 4090 peak: 83 TFLOPS fp16, 1.3M CUDA kernel launches/sec

## Root Cause: Kernel Launch Overhead Dominates Compute

Each training step launches **hundreds of CUDA kernels** (linear, layernorm, softmax, attention, per each of 4 layers, plus 5 separate loss heads). Each kernel does almost no work:

- **Attention dot products**: d_k=8. Each Q·K dot product is 8 multiplies. The attention matrix is 176×176 = 31K entries, but 31K × 8 = 248K FLOPs — trivial for a GPU that does 83 trillion per second.
- **Linear projections**: 32×32 matrices (1,024 FLOPs per token). For 1024×176 tokens that's 184M FLOPs — the GPU can do this in **2 microseconds** but the kernel launch alone takes **5-10 microseconds**.
- **Per-group loss loop**: 5 Python iterations, each doing separate tensor indexing, BCE/beta-NLL, masking. That's 15+ kernel launches with Python interpreter between them.

**The GPU spends more time starting and stopping kernels than actually computing.** It's like using a freight train to deliver individual letters.

## Why d_model=32 Is Uniquely Bad

GPU tensor cores operate on **16×16 tiles minimum** (fp16). With d_k=8 per attention head, every matmul wastes half the tile. With d_model=32, the QKV projection is a 32×96 matrix — the GPU pads this internally and wastes cycles.

For context: GPT-2 Small (124M params, d=768) gets ~150M tokens/sec on a 4090. It's 2500× more params but only 10× our token throughput, because its matrix multiplies are large enough to actually saturate the hardware.

## What Helps

| Fix | Speedup | Why |
|-----|---------|-----|
| `torch.compile` | ~1.3× | Fuses adjacent kernels, reduces launch count |
| AMP (fp16) | ~2.5× | Tensor cores + halved memory bandwidth |
| d_model=64 | ~2-4× (expected) | Matmuls fill tensor core tiles, better compute/overhead ratio |
| d_model=128 | ~4-8× (expected) | Near-optimal tile utilization |

## Recommendation: d_model=256

Go straight to **d_model=256, n_heads=8** (d_k=32). Rationale:

- d_k=16 (d=128) fills exactly one tensor core tile, but with **zero overlap between compute and memory access** — no pipelining possible.
- d_k=32 (d=256) provides enough arithmetic intensity for the GPU to actually pipeline memory loads with compute.
- Param count goes from ~49K to ~2-3M — still tiny, trains faster in wall-clock than d=32.
- Overfitting risk is mitigated by effectively unbounded RL data.

## Additional Optimizations

### Vectorize the Loss Loop (High Priority)

The 5-group Python loop with per-group indexing, masking, and different loss functions is the second-biggest bottleneck. Fix:

1. Precompute a single `[batch, max_groups]` target tensor with group assignments as an index dimension
2. Compute all BCE/beta-NLL in one fused call over the full batch
3. Scatter-reduce by group

Collapses ~15 kernel launches into 2-3. `torch.compile` might fuse some of this, but explicit vectorization is more reliable and makes the compiled graph simpler.

### Gradient Accumulation

Larger microbatches amortize fixed per-step overhead. Double the microbatch and halve accumulation steps — cuts kernel launch count roughly in half for the same effective batch.

### Structured Attention Mask (Investigate)

With 176 tokens where 144 are abilities, **profile whether full self-attention is learning useful cross-ability interactions** or whether abilities mostly attend to their parent entity. If the latter, a structured block-sparse attention mask would:
- Meaningfully reduce FLOPs at d_model=256
- Potentially improve learning signal by removing noise from irrelevant cross-ability attention

### Int8 Quantization for Rust Inference

At d_model=256 with 4 layers, weights are ~2-3MB fp32. Int8 with per-layer scale factors cuts to <1MB, solidly in L2 cache. Important for the 5000 sims/min self-play workload.

## Impact on Downstream

Changing d_model affects:
- **Actor-critic integration**: EntityEncoderV3 in `model.py` must match (currently d=32)
- **Ability transformer [CLS] embeddings**: Still d=32 from frozen transformer, projected via `ability_proj` — no retraining needed
- **Rust inference**: `ActorCriticWeightsV3` in `weights.rs` must match new dimensions
- **Export scripts**: JSON weight format adapts automatically to dimensions

The ability transformer's frozen d=32 embeddings are projected into the encoder's d_model via `ability_proj`, so changing encoder d_model doesn't require retraining the ability transformer.

## Action Plan

1. **Increase d_model to 256** in `EntityEncoderDecomposed` (and downstream EntityEncoderV3, Rust inference)
2. **Add `torch.compile` + AMP** to training loop
3. **Vectorize loss computation** — eliminate Python loop over groups
4. **Benchmark** the new config — expect 200-500+ steps/sec
5. **Retrain curriculum** from scratch at d=256 — should complete in minutes, not hours
6. **Profile attention maps** to evaluate structured masking opportunity
