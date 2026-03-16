# ML Terrain Generation System — Implementation Plan (v2)

## Project Goal

Replace the hand-coded procedural room generation system (`src/mission/room_gen/`) with a learned generative model that produces tactical room layouts from text prompts. The system generates multi-channel grids (walkable/blocked, obstacle type, height, elevation) conditioned on natural language descriptions, with room dimensions themselves predicted by the model.

**Example prompt → output:**
```
"Tight pressure room with a central corridor, flanking barricades,
 and an elevated sniper platform on the east side"
    ↓
  Dimension predictor: (12, 18)
    ↓
  ELIT-DiT generates 12×18 multi-channel grid
    ↓
  NavGrid → GridNav → Bevy 3D scene
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE (offline)                   │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │ Existing  │───▸│ Top-down     │───▸│ Qwen3-VL-4B         │   │
│  │ Proc-Gen  │    │ Render +     │    │ Captioning           │   │
│  │ (Rust)    │    │ Metrics      │    │ (image + metrics     │   │
│  │ *varied   │    │ Computation  │    │  → text description) │   │
│  │  sizes*   │    │              │    │                      │   │
│  └──────────┘    └──────────────┘    └─────────┬───────────┘   │
│       │                                         │               │
│       │ grids + dimensions                text captions         │
│       ▼                                         ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Phase 1: Dimension Predictor Training                   │   │
│  │   Input: text_embedding + room_type                       │   │
│  │   Output: (width, depth) continuous, clamped [8, 64]      │   │
│  │   Architecture: MLP head on text encoder                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Phase 2: ELIT-DiT Training                              │   │
│  │   Spatial domain: W×D multi-channel grid tokens           │   │
│  │   Read → K latent tokens (fixed J per group) → Core → Write│  │
│  │   Conditioned on: text_embedding + (width, depth)         │   │
│  │   Multi-budget tail dropping for elastic inference         │   │
│  │   CFG dropout: 15% unconditional during training          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Phase 3: PCGRL Critic Training                          │   │
│  │   Input: latent tokens (fixed size, resolution-agnostic)  │   │
│  │   Output: predicted tactical quality score                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE (runtime)                   │
│                                                                 │
│  text prompt + room_type                                        │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Phase 1: Dimension Prediction                            │   │
│  │  text_embedding + room_type → MLP → (width, depth)        │   │
│  │  Clamp to [8, 64], round to integers                      │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │ (W, D)                              │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Phase 2: ELIT-DiT Reverse Sampling                       │   │
│  │                                                           │   │
│  │  Spatial domain: W×D grid tokens (N = W*D)                │   │
│  │  Groups: G = N / group_area  (e.g. 4×4 groups)           │   │
│  │  Latent interface: K = G * J  (J fixed per group)         │   │
│  │                                                           │   │
│  │  ┌─ Head blocks (B_in): spatial domain, full resolution   │   │
│  │  │                                                        │   │
│  │  ├─ Read: cross-attn, spatial → latent interface          │   │
│  │  │  (pulls info from important grid regions)              │   │
│  │  │                                                        │   │
│  │  ├─ Core blocks (B_core): latent domain, fixed cost       │   │
│  │  │  + PhyScene guidance on latent gradients                │   │
│  │  │  + PCGRL critic guidance on latent tokens              │   │
│  │  │                                                        │   │
│  │  ├─ Write: cross-attn, latent → spatial                   │   │
│  │  │  (broadcasts updates back to grid cells)               │   │
│  │  │                                                        │   │
│  │  └─ Tail blocks (B_out): spatial domain, output velocity  │   │
│  │                                                           │   │
│  │  CCFG: conditioned path at full budget J                  │   │
│  │        unconditioned path at J_w ≈ 0.35*J (free AG)      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Post-processing                                          │   │
│  │  1. Argmax on obstacle-type channel → discrete tile map   │   │
│  │  2. Quantize height/elevation channels                    │   │
│  │  3. Enforce perimeter walls (always blocked)              │   │
│  │  4. Run validate_layout() — reject if failed, retry       │   │
│  │  5. Extract ObstacleRegion list for Bevy visual spawning  │   │
│  │  6. NavGrid::to_gridnav() for AI pathfinding              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Architecture Decision: ELIT over MiDiffusion + VAE

The original plan (v1) proposed MiDiffusion (mixed discrete-continuous diffusion on obstacle sets) with a separate VAE for resolution independence. ELIT (Elastic Latent Interface Transformer, Haji-Ali et al., arXiv:2603.12245, March 2026) replaces both components with a single, cleaner architecture.

**Why ELIT is the better fit:**

1. **Resolution independence is built in.** ELIT's grouped cross-attention partitions spatial tokens into G groups, each with J learnable latent tokens. J is fixed; G scales with room area. A 10×10 room and a 50×50 room use the same J per group — only G changes. No separate VAE needed.

2. **Adaptive compute allocation.** The Read layer learns to pull more information from spatially complex grid regions (dense obstacle clusters, chokepoints) and less from empty floor. This is exactly the compute pattern tactical rooms need — most cells are empty floor, a few regions contain all the interesting geometry.

3. **Multi-budget inference for free.** Tail-token dropping during training creates importance-ordered latents: early tokens encode global structure (room connectivity, major walls), later tokens encode fine detail (small cover positions, precise edges). At inference, the latent count per group is a knob:
   - 25% tokens: fast validation pass for PCGRL critic evaluation
   - 100% tokens: full quality for final generation

4. **CCFG saves ~33% inference cost.** Cheap classifier-free guidance runs the conditioned path at full budget and the unconditioned path at ~35% budget. No separate weak model needed — ELIT provides one natively.

5. **Drop-in compatibility.** ELIT adds only Read and Write cross-attention layers to a standard DiT. The rectified flow objective is unchanged. This means standard diffusion training infrastructure works.

6. **Proven at scale.** Demonstrated on DiT, U-ViT, HDiT, and MM-DiT (20B Qwen-Image). Achieves 35-58% FDD improvement on ImageNet 512px. Scales favorably with larger models and higher resolutions.

**Reference:** `https://snap-research.github.io/elit`

---

## Milestone Breakdown

### Milestone 0 — Data Generation & Captioning Pipeline

**Goal:** Generate paired (grid, text_caption, dimensions) training data from the existing proc-gen system, with varied room sizes.

#### M0.1 — Augmented Batch Room Generation (Rust)

Extend the existing `generate_room` to support varied dimensions per room type, breaking the rigid `room_dimensions()` lookup table.

**Modification to proc-gen:**
```rust
/// Generate a room with optionally overridden dimensions.
/// If no override, use default dimensions with ±30% random perturbation.
pub fn generate_room_varied(
    seed: u64,
    room_type: RoomType,
    dim_override: Option<(f32, f32)>,
) -> RoomLayout {
    let (base_w, base_d) = room_dimensions(room_type);
    let (width, depth) = match dim_override {
        Some((w, d)) => (w.clamp(8.0, 64.0), d.clamp(8.0, 64.0)),
        None => {
            let mut rng = Lcg::new(seed.wrapping_add(0xDIM5));
            let scale_w = rng.next_f32_range(0.7, 1.3);
            let scale_d = rng.next_f32_range(0.7, 1.3);
            (
                (base_w * scale_w).clamp(8.0, 64.0),
                (base_d * scale_d).clamp(8.0, 64.0),
            )
        }
    };
    // ... rest of generation with (width, depth)
}
```

This produces training data where Entry rooms range from ~14×14 to ~26×26 instead of always 20×20, etc. Non-square rooms emerge naturally from independent axis scaling.

**CLI binary output per room:**
```json
{
  "seed": 42,
  "room_type": "Entry",
  "width": 17,
  "depth": 23,
  "grid": {
    "channels": ["obstacle_type", "height", "elevation"],
    "obstacle_type": [[0, 0, 1, ...], ...],
    "height": [[0.0, 0.0, 1.5, ...], ...],
    "elevation": [[0.0, 0.0, 0.0, ...], ...]
  },
  "metrics": {
    "blocked_pct": 0.12,
    "chokepoint_count": 2,
    "cover_density": 0.34,
    "elevation_zones": 1,
    "flanking_routes": 3,
    "spawn_quality_diff": 1.2,
    "mean_wall_proximity": 3.1,
    "aspect_ratio": 0.74
  },
  "player_spawn": [[2.5, 5.5], ...],
  "enemy_spawn": [[14.5, 18.5], ...]
}
```

**Multi-channel grid encoding** (per cell):
- `obstacle_type`: integer, 0=floor, 1=wall, 2=pillar, 3=barricade, 4=l_shape, 5=cover_cluster, 6=sandbag, 7=platform_edge, 8=ramp
- `height`: float, visual height in meters (0.0 for floor, 0.5–2.0 for obstacles)
- `elevation`: float, walkable surface elevation (0.0–1.5)

**Tactical metrics to compute** (extend existing validation):
- `blocked_pct` — interior blocked cells / total interior cells
- `chokepoint_count` — cells where `chokepoint_score_by_cell >= 2`
- `cover_density` — fraction of walkable cells within 1 cell of a blocked cell
- `elevation_zones` — number of distinct elevation values > 0
- `flanking_routes` — distinct shortest paths between spawn centroids
- `spawn_quality_diff` — existing `score_spawn_quality` delta
- `mean_wall_proximity` — average of `wall_proximity_by_cell`
- `aspect_ratio` — width / depth

**Target:** 30,000–50,000 rooms across all 6 room types × varied seeds × varied dimensions.

#### M0.2 — Headless Top-Down Rendering (Rust)

Produce a color-coded top-down image of each room for VLM captioning. Pure Rust with the `image` crate — no Bevy dependency.

**Rendering spec:**
- Resolution: 4 pixels per cell (e.g., 17×23 room → 68×92 image)
- Color mapping:
  - `#000000` black — perimeter wall
  - `#404040` dark grey → `#606060` — interior obstacles, darker = taller
  - `#FFFFFF` white — walkable floor at elevation 0
  - `#6688CC` blue tint — walkable floor with elevation > 0, brighter = higher
  - `#22AA22` green — player spawn cells
  - `#CC2222` red — enemy spawn cells
- Output: PNG per room

#### M0.3 — Qwen3-VL Captioning (Python)

Batch inference: feed each room image + metrics JSON to Qwen3-VL-4B-Instruct.

**Model:** `Qwen/Qwen3-VL-4B-Instruct` (~8GB VRAM at bf16)

**Prompt template:**
```
You are describing tactical room layouts for a squad-based combat game.
Given the top-down image and computed metrics, write a 1-3 sentence
description of the room's tactical properties. Focus on:
- Room shape and proportions
- Cover layout and density
- Chokepoints and corridors
- Elevation advantages
- Flanking routes
- Asymmetries between the two spawn sides (green=player, red=enemy)

Room type: {room_type}
Dimensions: {width}×{depth}
Metrics:
  Blocked: {blocked_pct:.0%}
  Chokepoints: {chokepoint_count}
  Cover density: {cover_density:.2f}
  Elevation zones: {elevation_zones}
  Flanking routes: {flanking_routes}
  Aspect ratio: {aspect_ratio:.2f}

Describe this room's tactical layout:
```

**Output:** Append `caption` field to each room's JSON record.

**Quality control:** Spot-check 100 captions. If 4B is too generic, upgrade to Qwen3-VL-8B-Thinking (~16GB VRAM).

**Target:** Caption all rooms. At ~0.5s/room on 4090, 30K rooms takes ~4 hours.

---

### Milestone 1 — ELIT-DiT Model Training

**Goal:** Train an ELIT-enhanced diffusion transformer that generates multi-channel room grids at arbitrary dimensions, conditioned on text + room type.

#### M1.1 — Spatial Domain Representation

Each grid cell is encoded as a multi-channel vector for the spatial domain:

```
cell_token ∈ R^d  where d = d_embed

Channels (before embedding):
  obstacle_type: int in [0, 8]  → learned embedding (d_type = 32)
  height: float [0, 2.0]       → linear projection (d_height = 16)
  elevation: float [0, 1.5]    → linear projection (d_elev = 16)
  
  Concatenated + linear → d_model
```

The spatial grid has N = W × D tokens. Each token knows its (col, row) position via 2D RoPE (following ELIT's approach), so the model handles arbitrary grid dimensions without retraining positional encodings.

#### M1.2 — Dimension Predictor

A lightweight module that predicts room dimensions from the text prompt and room type.

**Architecture:**
```
text_embedding (from text encoder) ∈ R^d_text
room_type_embedding ∈ R^d_type  (learned, 6 room types)

concat → MLP(d_text + d_type, 256, 256, 2) → (width_raw, depth_raw)
width = clamp(sigmoid(width_raw) * 56 + 8, 8, 64)
depth = clamp(sigmoid(depth_raw) * 56 + 8, 8, 64)
```

**Training:** Simple L2 regression loss on ground-truth dimensions from the training data.

**Joint training option:** The dimension predictor shares the text encoder with the ELIT-DiT. Train it in Phase 1 of each forward pass, then use predicted (or ground-truth during training) dimensions to instantiate the spatial grid for Phase 2.

**Post-training refinement via PCGRL:** After initial training, fine-tune the dimension predictor with a reward signal from the tactical quality critic. This teaches it to deviate from proc-gen priors when different dimensions would yield better tactical layouts for a given prompt.

#### M1.3 — Text Encoder

Options in order of recommendation:

1. **Frozen sentence encoder:** `sentence-transformers/all-MiniLM-L6-v2` (80MB, 384-dim). No training needed. Start here.
2. **Lightweight custom encoder:** 2-layer transformer MLM trained on the caption corpus. MarioGPT found this outperformed larger pretrained encoders for domain-specific vocabularies.
3. **Qwen3.5-4B hidden states:** Most expensive but highest capacity. Only if controllability from options 1-2 is insufficient.

#### M1.4 — ELIT-DiT Architecture

Adapt ELIT for multi-channel grid generation using rectified flow.

**Block allocation** (following ELIT ablations, ~70% in latent core):
- For a DiT-B scale model (~12 blocks): B_in=2, B_core=8, B_out=2
- For a DiT-L scale model (~24 blocks): B_in=4, B_core=16, B_out=4

**Spatial domain:**
- Input: N = W × D patchified grid tokens (patch size 1×1 for small rooms, 2×2 for rooms > 32 on either axis)
- 2D RoPE positional encoding (resolution-independent)
- Head blocks process raw noisy grid tokens

**Latent interface:**
- Group size: 4×4 spatial tokens per group (following ELIT ablation, Table 4a — 16 groups was optimal at 256px, 4×4 groups is the analogous choice for game grids)
- J = 8 latent tokens per group (tunable)
- Groups G = ceil(W/4) × ceil(D/4)
- Total latent tokens K = G × J
- Learnable positional embeddings per group (shared across groups, resolution-independent)

**Read layer:**
- Grouped cross-attention: latent queries attend to spatial keys/values within corresponding group
- adaLN-Zero modulation (timestep-aware)
- QK normalization for stability
- No FFN expansion (minimal overhead)

**Core blocks:**
- Standard DiT transformer blocks operating on K latent tokens
- Conditioning via cross-attention on text embedding + dimension embedding
- This is where ~70% of compute lives, at fixed cost regardless of room size

**Write layer:**
- Symmetric to Read: spatial queries attend to latent keys/values within group
- Broadcasts updated latent information back to spatial grid

**Tail blocks:**
- Spatial domain, produces output velocity prediction
- Restores fine spatial detail and aligns to prediction space

**Multi-budget training (tail dropping):**
- J_min = 2, J_max = 8 (per group)
- Each training iteration: sample J̃ ~ Uniform{2, ..., 8}, drop tail latents beyond J̃
- Creates importance-ordered latents: early = global structure, late = fine detail

**Output channels:**
- obstacle_type logits: R^9 per cell (softmax → discrete type)
- height: R^1 per cell (continuous)
- elevation: R^1 per cell (continuous)

**CFG implementation:**
- During training: drop text conditioning with probability 0.15
- At inference: CCFG with conditioned path at full J, unconditioned path at J_w = max(2, J//3)
- Saves ~33% inference FLOPs with improved quality over standard CFG

#### M1.5 — Training Details

```
Dataset: 30K-50K rooms from M0 (varied dimensions, all room types)
Objective: Rectified flow (linear interpolation, velocity prediction)
Timestep sampling: logit-normal distribution
Optimizer: AdamW, lr=1e-4, 10k warmup, cosine decay
Batch size: 256 (variable effective batch via multi-budget tail dropping,
            increase to 384 to match compute)
Gradient clipping: 1.0
EMA: β = 0.9999
Training steps: 200K-500K (monitor FDD on held-out set)
Hardware: Single 4090
```

**Estimated VRAM:** ~14-18GB depending on max room size in batch. Rooms > 40×40 may require gradient checkpointing or smaller batch size. Consider capping training room size at 48×48 and relying on ELIT's resolution generalization for larger rooms at inference.

**Estimated training time:** 24-72 hours on 4090.

#### M1.6 — Training Infrastructure

```
project/
├── data/
│   ├── rooms/           # JSON records from M0.1
│   ├── images/          # Top-down PNGs from M0.2
│   └── captions/        # Augmented JSONs from M0.3
├── model/
│   ├── elit_dit.py      # ELIT-DiT architecture
│   ├── read_write.py    # Read/Write cross-attention layers
│   ├── flow_matching.py # Rectified flow forward/reverse
│   ├── dim_predictor.py # Dimension prediction head
│   ├── text_encoder.py  # Text conditioning module
│   ├── dataset.py       # Variable-size room grid dataset
│   ├── train.py         # Training loop (joint dim predictor + ELIT-DiT)
│   └── sample.py        # Inference with CCFG + guidance
├── guidance/
│   ├── collision.py     # Obstacle overlap penalty
│   ├── boundary.py      # Room boundary constraint
│   ├── connectivity.py  # Spawn-to-spawn pathfinding proxy
│   ├── cover.py         # Cover density targeting
│   ├── blocked.py       # Blocked percentage targeting
│   └── critic.py        # PCGRL critic guidance (M3)
└── eval/
    ├── metrics.py       # Tactical metric computation
    ├── validate.py      # Full validation pipeline
    └── compare.py       # Generated vs proc-gen distribution comparison
```

---

### Milestone 2 — PhyScene-Style Guidance Functions

**Goal:** Implement differentiable constraint functions that steer diffusion sampling toward valid, tactically interesting layouts.

Guidance operates at the Write output (spatial domain) where gradients can be computed on actual grid cell values, then propagated back through Write into the latent domain. This is more natural than computing constraints on abstract latent tokens — you're penalizing actual spatial violations.

#### M2.1 — Collision Guidance

Penalize overlapping obstacle regions. In the grid representation, this means penalizing cells where the model predicts high probability for multiple non-floor obstacle types simultaneously (before argmax).

```python
def collision_loss(obstacle_logits):
    """Penalize cells with ambiguous obstacle predictions.
    obstacle_logits: (W, D, 9) — pre-softmax type logits
    """
    probs = softmax(obstacle_logits, dim=-1)
    # Entropy of type distribution — high entropy = ambiguous
    non_floor_probs = probs[..., 1:]  # exclude floor channel
    non_floor_mass = non_floor_probs.sum(dim=-1)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
    # Penalize high entropy in cells with high non-floor probability
    return (non_floor_mass * entropy).mean()
```

#### M2.2 — Boundary Guidance

Enforce that perimeter cells remain blocked and interior obstacles stay off the perimeter.

```python
def boundary_loss(obstacle_logits, W, D):
    """Penalize non-wall predictions on perimeter, wall predictions on interior."""
    probs = softmax(obstacle_logits, dim=-1)
    perimeter_mask = build_perimeter_mask(W, D)  # True for edge cells
    
    # Perimeter cells should be wall (type=1)
    perimeter_loss = -probs[perimeter_mask, 1].log().mean()
    
    # Cells adjacent to perimeter should prefer floor
    inner_ring = build_inner_ring_mask(W, D)
    inner_loss = probs[inner_ring, 1:].sum(dim=-1).mean() * 0.1
    
    return perimeter_loss + inner_loss
```

#### M2.3 — Connectivity Guidance

Ensure paths exist between player and enemy spawn zones.

**Approach:** Differentiable soft distance field.

```python
def connectivity_loss(obstacle_logits, player_spawn, enemy_spawn):
    """Soft pathfinding penalty using iterative Bellman relaxation."""
    probs = softmax(obstacle_logits, dim=-1)
    floor_prob = probs[..., 0]  # probability each cell is walkable
    
    # Soft occupancy: high = likely blocked
    occupancy = 1.0 - floor_prob
    
    # Initialize distance field from player spawn
    dist = torch.full((W, D), 1e6)
    for (px, py) in player_spawn:
        dist[px, py] = 0.0
    
    # Iterative Bellman updates (20-50 iterations)
    for _ in range(30):
        neighbors = [roll(dist, dx, dy) + cost for dx, dy, cost in NEIGHBORS_4]
        min_neighbor = torch.stack(neighbors).min(dim=0).values
        # Blocked cells have infinite cost
        dist = torch.where(occupancy > 0.5, torch.tensor(1e6), 
                          torch.minimum(dist, min_neighbor))
    
    # Penalize if enemy spawn is unreachable
    enemy_dists = [dist[ex, ey] for (ex, ey) in enemy_spawn]
    return torch.stack(enemy_dists).mean() / (W + D)  # normalize
```

**Simpler fallback:** Penalize when the ratio of blocked cells in the rectangular corridor between spawn centroids exceeds 0.8.

#### M2.4 — Cover Density Guidance

Target a specific cover density range.

```python
def cover_density_loss(obstacle_logits, target_density=0.3):
    """Penalize deviation from target cover density."""
    probs = softmax(obstacle_logits, dim=-1)
    blocked = 1.0 - probs[..., 0]  # non-floor probability per cell
    walkable = probs[..., 0]
    
    # Soft "near cover" via max-pooling of blocked probability
    near_blocked = max_pool_2d(blocked.unsqueeze(0).unsqueeze(0), 
                               kernel_size=3, padding=1).squeeze()
    near_cover = walkable * near_blocked
    density = near_cover.sum() / walkable.sum().clamp(min=1)
    
    return (density - target_density) ** 2
```

#### M2.5 — Blocked Percentage Guidance

Keep total blocked area within [2%, 35%].

```python
def blocked_pct_loss(obstacle_logits, target_range=(0.02, 0.35)):
    """Soft blocked percentage constraint."""
    probs = softmax(obstacle_logits, dim=-1)
    # Exclude perimeter (always blocked, not model's choice)
    interior = probs[1:-1, 1:-1]
    blocked_pct = (1.0 - interior[..., 0]).mean()
    
    low_penalty = F.relu(target_range[0] - blocked_pct)
    high_penalty = F.relu(blocked_pct - target_range[1])
    return low_penalty + high_penalty
```

#### M2.6 — Guidance Integration During Sampling

At each reverse diffusion step, guidance gradients are computed on the spatial domain output (after Write) and propagated back into the latent domain:

```python
for t in reversed(range(T)):
    # Standard ELIT-DiT denoising step
    spatial_out = elit_dit.forward(x_t, text_cond, dim_cond, t)
    # spatial_out shape: (W, D, C) where C = 9 + 1 + 1
    
    obstacle_logits = spatial_out[..., :9]
    height = spatial_out[..., 9]
    elevation = spatial_out[..., 10]
    
    # Compute guidance gradients
    spatial_out.requires_grad_(True)
    loss = (lambda_coll  * collision_loss(obstacle_logits)
          + lambda_bound * boundary_loss(obstacle_logits, W, D)
          + lambda_conn  * connectivity_loss(obstacle_logits, p_spawn, e_spawn)
          + lambda_cover * cover_density_loss(obstacle_logits)
          + lambda_block * blocked_pct_loss(obstacle_logits))
    
    # Optionally add PCGRL critic guidance (M3)
    # loss -= lambda_critic * critic_model(latent_tokens, t)
    
    grad = torch.autograd.grad(loss, spatial_out)[0]
    spatial_out = spatial_out - guidance_scale(t) * grad
```

**Guidance weight schedule:** Ramp schedule `λ_t = λ * (1 - ᾱ_t)` — stronger guidance at noisier (early) steps, relaxed at later steps when fine detail is resolved.

---

### Milestone 3 — PCGRL Critic Integration

**Goal:** Train a learned tactical quality critic that operates on ELIT's latent tokens and provides a guidance signal during diffusion sampling.

#### M3.1 — Critic Architecture

The critic operates on the latent interface tokens, which are a **fixed-size, resolution-independent** representation. This is the key advantage of combining PCGRL with ELIT — the critic doesn't need to handle variable grid sizes.

**Input:**
- Latent tokens: K tokens × d_model (from Read output, before core blocks)
- Diffusion timestep: t (so critic knows noise level)
- Dimension conditioning: (width, depth)

**Architecture:** Small transformer (2-3 layers, d=128, 4 heads) with a scalar output head.

**Output:** Predicted quality score in [0, 1].

#### M3.2 — Critic Training Data

Generate training data by:
1. Take ground-truth room grids from proc-gen corpus
2. Encode through ELIT-DiT's head blocks + Read to get latent tokens
3. Add diffusion-process noise at various timesteps (t ∈ {0.1, 0.3, 0.5, 0.7, 0.9})
4. Compute ground-truth tactical quality score for each clean room

**Quality score formula:**
```python
quality = (
    (1.0 if connected else 0.0)
    * clip(1 - abs(blocked_pct - target_blocked) / 0.1, 0, 1)
    * clip(1 - abs(cover_density - target_cover) / 0.15, 0, 1)
    * clip(1 - abs(chokepoints - target_chokepoints) / 2, 0, 1)
    * clip(1 - spawn_quality_diff / 3.0, 0, 1)
)
```

Target values derived from text caption or room type defaults.

#### M3.3 — Critic as Guidance Function

During reverse sampling, compute critic gradient on latent tokens:

```python
# After Read, before core blocks
latent_tokens.requires_grad_(True)
quality = critic_model(latent_tokens, t, width, depth)
critic_grad = torch.autograd.grad(quality, latent_tokens)[0]
latent_tokens = latent_tokens + lambda_critic * critic_grad  # maximize quality
```

This runs in the latent domain (fixed cost, resolution-agnostic) while the PhyScene guidance in M2 runs in the spatial domain (resolution-dependent but on simpler per-cell computations). The two guidance layers are complementary.

#### M3.4 — PCGRL-Guided Dimension Refinement (optional)

After the critic is trained, use its quality predictions to fine-tune the dimension predictor:

```python
# For each text prompt, try several candidate dimensions
for (w, d) in candidate_dimensions:
    latent_tokens = elit_dit.sample_latents(text, w, d, num_steps=10)  # fast low-budget
    quality = critic_model(latent_tokens, t=0.0, w, d)
    
# REINFORCE update on dimension predictor
reward = quality - baseline
dim_predictor_loss = -reward * log_prob(predicted_w, predicted_d)
```

This teaches the dimension predictor to choose sizes that maximize tactical quality for a given prompt, going beyond the proc-gen's rigid size-per-type mapping.

---

### Milestone 4 — Rust Integration & Bevy Pipeline

**Goal:** Connect the Python ML inference to the Rust game runtime.

#### M4.1 — Export & Inference Options

**Option A — ONNX export (recommended for production):**
- Export ELIT-DiT denoiser + text encoder + dimension predictor to ONNX
- Run inference via `ort` crate (ONNX Runtime Rust bindings)
- Guidance functions reimplemented in Rust (simple grid math)
- Full diffusion loop in Rust
- Pros: No Python dependency, fast, deterministic
- Cons: Export complexity

**Option B — Python subprocess (recommended for prototyping):**
- Python script: (text_prompt, room_type, seed) → JSON output
- Rust calls via `std::process::Command` or Unix socket
- Pros: Fast iteration, full PyTorch flexibility
- Cons: Python dependency, IPC overhead (acceptable for pre-generation)

**Recommendation:** Start with Option B. Port to Option A for release.

#### M4.2 — NavGrid Construction from ML Output

The ELIT-DiT outputs a multi-channel grid. Post-process to NavGrid:

```rust
fn ml_grid_to_navgrid(
    obstacle_types: &[Vec<u8>],   // W×D grid of type IDs
    heights: &[Vec<f32>],          // W×D height values
    elevations: &[Vec<f32>],       // W×D elevation values
    width: usize,
    depth: usize,
    cell_size: f32,
) -> (NavGrid, Vec<ObstacleRegion>) {
    let mut nav = NavGrid::new(width, depth, cell_size);
    let mut obstacles = Vec::new();
    
    // Enforce perimeter walls
    nav.set_walkable_rect(0, 0, width - 1, 0, false);
    nav.set_walkable_rect(0, depth - 1, width - 1, depth - 1, false);
    nav.set_walkable_rect(0, 0, 0, depth - 1, false);
    nav.set_walkable_rect(width - 1, 0, width - 1, depth - 1, false);
    
    // Process interior cells
    for r in 1..depth-1 {
        for c in 1..width-1 {
            let obs_type = obstacle_types[r][c];
            let height = heights[r][c];
            let elevation = elevations[r][c];
            
            if obs_type == 0 {
                // Floor — walkable, possibly elevated
                if elevation > 0.1 {
                    nav.set_elevation_rect(c, r, c, r, elevation);
                }
            } else if obs_type == 8 {
                // Ramp — walkable with elevation
                nav.set_elevation_rect(c, r, c, r, elevation);
            } else {
                // Obstacle — blocked
                let idx = nav.idx(c, r);
                nav.walkable[idx] = false;
            }
        }
    }
    
    // Extract contiguous ObstacleRegion blocks via connected components
    obstacles = extract_obstacle_regions(&obstacle_types, &heights);
    
    (nav, obstacles)
}
```

#### M4.3 — Bevy Visual Spawning

The extracted `Vec<ObstacleRegion>` feeds directly into the existing `spawn_room` function in `visuals.rs`. The `ObstacleRegion` struct is unchanged:

```rust
pub(crate) struct ObstacleRegion {
    pub col0: usize,
    pub col1: usize,
    pub row0: usize,
    pub row1: usize,
    pub height: f32,
}
```

The ML model's continuous height values produce more varied visuals than the proc-gen's fixed height tiers. The existing color palette derivation from the seed works unchanged.

For ramps, extract contiguous elevated walkable regions as `RampRegion` entries.

#### M4.4 — Spawn Zone Placement

After NavGrid construction, spawn zones use the existing `build_spawn_zone` logic:

1. Dimension predictor determines room size
2. ELIT-DiT generates the grid
3. Pick two anchor columns with minimum separation (from room config)
4. Find walkable cells in each zone
5. Select spawn positions from walkable candidates

Spawn placement remains procedural — the ML model generates terrain, not spawn logic. The separation constraint from `generate_room` is preserved.

#### M4.5 — Validation & Fallback

```rust
fn generate_ml_room(prompt: &str, room_type: RoomType, seed: u64) -> RoomLayout {
    for attempt in 0..3 {
        let (width, depth) = predict_dimensions(prompt, room_type);
        let ml_grid = run_elit_inference(prompt, room_type, width, depth, seed + attempt);
        let (nav, obstacles) = ml_grid_to_navgrid(&ml_grid, width, depth, 1.0);
        
        if validate_layout(&nav) {
            let ramps = extract_ramp_regions(&ml_grid);
            return RoomLayout {
                width: width as f32,
                depth: depth as f32,
                nav,
                player_spawn: build_spawn_zone(...),
                enemy_spawn: build_spawn_zone(...),
                room_type,
                seed,
                obstacles,
                ramps,
            };
        }
    }
    // Fallback to proc-gen
    generate_room(seed, room_type)
}
```

#### M4.6 — MissionRoomSequence Integration

Update `MissionRoomSequence` and `advance_room_system` to use the ML generator:

```rust
// In types.rs — add optional text prompt
pub struct MissionRoomSequence {
    pub rooms: Vec<RoomType>,
    pub prompts: Vec<Option<String>>,  // NEW: per-room text prompts
    pub current_index: usize,
    pub seed: u64,
    pub current_room_origin: Vec3,
}

// In systems.rs — use ML generation when prompt is available
let new_layout = match seq.prompts.get(seq.current_index) {
    Some(Some(prompt)) => generate_ml_room(prompt, new_room_type, new_seed),
    _ => generate_room(new_seed, new_room_type),
};
```

---

### Milestone 5 — Evaluation & Iteration

**Goal:** Quantify the ML generator's quality relative to the proc-gen baseline.

#### M5.1 — Metric Comparison

Generate 1000 rooms from each system per room type. Compare distributions:

- Blocked percentage (target: 2–35%, 100% pass rate)
- Connectivity (target: 100% pass rate)
- Cover density distribution
- Chokepoint count distribution
- Spawn quality balance
- Elevation usage
- Obstacle type diversity
- **Dimension distribution** (ML only): verify room sizes are reasonable and varied

#### M5.2 — Text Compliance

For 100 hand-written tactical prompts, generate rooms and have Qwen3-VL-4B assess whether the generated room matches the prompt. Compute compliance rate.

Also test dimension compliance: "tight corridor" should produce narrow rooms, "open arena" should produce large rooms.

#### M5.3 — Playtest Metrics

Run `run_scenario` on ML-generated rooms:
- Match duration distribution (avoid degenerate fast kills or timeouts)
- Hero vs enemy win rates by room type
- Ability usage diversity
- **Compare across generated room sizes** — verify that varied dimensions improve gameplay diversity

#### M5.4 — Diversity

Pairwise distance metrics between generated rooms:
- Grid-level Hamming distance (walkable cell agreement, normalized by grid area)
- Obstacle-set Chamfer distance (position/type similarity)
- A* trajectory edit distance (gameplay-relevant)
- **Dimension variance** — verify the model doesn't collapse to a single size per room type

#### M5.5 — Multi-Budget Quality Curve

Evaluate generation quality at different latent token budgets (25%, 50%, 75%, 100%) and plot the quality-compute tradeoff. Verify that ELIT's importance ordering preserves room connectivity and global structure even at 25% tokens, with fine detail (small cover positions, precise edges) degrading gracefully.

---

## Hardware & Compute Requirements

| Component | VRAM | Time | Notes |
|-----------|------|------|-------|
| Qwen3-VL-4B captioning | ~8 GB | 3-7 hours | Batch inference on 30-50K rooms |
| ELIT-DiT training | ~14-18 GB | 24-72 hours | Single 4090, gradient checkpointing for large rooms |
| PCGRL critic training | ~4-8 GB | 4-12 hours | Small model, large dataset |
| Inference (1 room) | ~4-6 GB | 2-10 seconds | DDIM/Euler 40 steps + CCFG + guidance |

All components fit on a single 4090 (24 GB). Run M0.3, M1, M3 sequentially.

**Multi-budget inference on 4090:**
- Full budget (100% tokens): ~8-10s per room
- Fast validation (25% tokens): ~3-4s per room
- CCFG saves ~33% vs standard CFG

---

## Key Dependencies & References

| Component | Source | License |
|-----------|--------|---------|
| ELIT | `snap-research.github.io/elit` (arXiv:2603.12245) | — |
| PhyScene | `github.com/PhyScene/PhyScene` | Apache 2.0 |
| MiDiffusion (reference) | `github.com/MIT-SPARK/MiDiffusion` | MIT |
| Discrete Diffusion Guidance | `github.com/kuleshov-group/discrete-diffusion-guidance` | MIT |
| Qwen3-VL-4B | `huggingface.co/Qwen/Qwen3-VL-4B-Instruct` | Apache 2.0 |
| PCGRL reference | `github.com/smearle/control-pcgrl` | MIT |
| Moonshine (captioning pipeline ref) | arXiv:2408.09594 | — |
| MarioGPT (text encoder finding) | `github.com/shyamsn97/mario-gpt` | MIT |

**Key papers:**
- Haji-Ali et al., "ELIT: One Model, Many Budgets" (arXiv:2603.12245, 2026) — Core architecture
- Yang et al., "PhyScene" (CVPR 2024) — Guidance functions
- Hu et al., "MiDiffusion" (arXiv:2405.21066, 2024) — Mixed diffusion reference
- Schiff et al., "Simple Guidance for Discrete Diffusion" (2024) — CFG for discrete tokens
- Nie et al., "Moonshine" (AAAI 2025) — Synthetic captioning pipeline
- Sudhakaran et al., "MarioGPT" (NeurIPS 2023) — Text encoder finding
- Earle et al., "Controllable PCGRL" (IEEE CoG 2021) — RL critic

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ELIT-DiT doesn't converge on small game grid data (very different from ImageNet) | Start with DiT-B scale (~12 blocks, ~100M params). The spatial structure of game grids is much simpler than natural images — less capacity needed. Augment data with rotations/mirrors (8× augmentation). |
| Room dimensions collapse to proc-gen defaults | Augment proc-gen with ±30% size perturbation. Fine-tune dimension predictor with PCGRL reward. Add dimension diversity loss during training. |
| Connectivity guidance is non-differentiable | Use soft distance field proxy (M2.3). Fallback: blocked-corridor-ratio heuristic. Worst case: hard validation + retry (already in the pipeline). |
| Text compliance is poor | Start with structured conditioning (room_type + metric targets) alongside text. Upgrade text encoder if needed. The text is a soft signal; the guidance functions enforce hard constraints. |
| PCGRL critic doesn't generalize to noisy latent inputs | Train with explicit noise augmentation at multiple diffusion timesteps. The latent tokens' importance ordering (from tail dropping) helps — early latents are stable even under noise. |
| Generated rooms are too homogeneous | Add diversity loss: penalize low variance of obstacle type distribution across a batch. Use temperature scaling on obstacle type logits during sampling. |
| 4090 VRAM pressure with large rooms (50×50+) | Gradient checkpointing on core blocks. Cap training rooms at 48×48. Rely on ELIT's resolution generalization for larger rooms at inference (Read/Write are resolution-independent). |
| Inference too slow for gameplay | Pre-generate room pool offline. DDIM with fewer steps (20 vs 40). Multi-budget: use 50% tokens for standard rooms, 100% only for Climax/Setpiece. |

---

## Open Questions for Future Work

1. **Room sequence conditioning.** Can the model condition on the previous room's layout to ensure thematic continuity across a mission? Pass the previous room's latent tokens as additional context.

2. **Interactive editing via masking.** ELIT's Read/Write architecture supports partial constraints naturally. Fix some spatial regions (designer-placed landmarks) and let the model fill the rest. This enables mixed-initiative design.

3. **RL fine-tuning from playtests (Pattern 2).** Fine-tune the ELIT-DiT end-to-end using DDPO with match outcome rewards from `run_scenario`. The dimension predictor and layout generator are jointly optimized for gameplay quality.

4. **Multi-resolution generation.** For very large rooms (Open, 100×100), generate a coarse 25×25 layout at full budget, then use a second ELIT pass to upsample to 100×100. The coarse pass captures global structure; the fine pass adds detail. ELIT's multi-budget training naturally supports this.

5. **Prop primitive conditioning.** Instead of (or in addition to) text prompts, condition on a desired set of obstacle primitives: "use 2 L-shapes and 1 elevated platform." This maps to the existing template vocabulary and gives designers fine-grained control.

6. **Scenario-aware generation.** Condition on the enemy wave composition (4 melee rushers vs 3 ranged + 1 healer) to generate rooms that create interesting matchups for the specific encounter.
