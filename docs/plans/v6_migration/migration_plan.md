# AbilityActorCriticV6 Migration Plan

## Context

Current model (V4): 134K params, d=32, 4 heads, h_dim=64. Training time ~30min.
Target model (V6): ~1.7M params, d=128, 8 heads, h_dim=256. Estimated training time ~2-3hr.

The GPU (RTX 4090) is massively underutilized at ~5-10% during episode generation.
The bottleneck is SHM round-trip and Rust sim tick rate, not GPU compute.
The model can be 10-20x larger before GPU becomes the bottleneck.

Key motivation: at d=32 with 4 heads, each head has only 8 dimensions — too narrow
for attention to represent meaningful tactical relationships. d=128 / 8 heads = 16d/head
is where transformer attention becomes genuinely expressive.

---

## Scene Representation: Information Capacity as a Design Knob

### The Problem

Entity/threat slot truncation is a **data-level** bottleneck, not a model-level one.
Architectural improvements (d=128, latent interface, CfC) only improve how the model
reasons about what it sees — they cannot recover information discarded before the model
runs. For a 12-enemy chokepoint push with 7 entity slots, the model sees 3 enemies and
has no signal that 9 more exist. "Mass push" and "3 isolated scouts" look identical.

This is distinct from capacity limits inside the model (which are intentional — see below).

### Design Philosophy: Intentional Capacity Limits

The slot limit is not purely a bug. A model that always has complete information never
learns to operate under uncertainty. The goal is:

- **Identified individual slots** for tactically significant units (full detail)
- **Aggregate summary** for the unrepresented crowd (meta-awareness without full info)
- **Importance-based selection** in Rust to decide which units earn individual slots

This means the model knows "12 enemies exist, centroid is northeast, aggregate threat
is high" without knowing each individual's details. It learns to condition on that
uncertainty. This is more realistic and more robust than full-information policies,
and it enables a natural scaling axis: allocate more powerful units more individual
compute by giving them guaranteed slots.

### Aggregate Summary Token

Add 1 aggregate token per tick summarizing all entities that did not earn individual slots.
Projects to 128d via `agg_proj(N_agg_features → 128)`, appended to the token sequence.
The latent interface naturally routes attention to it when the crowd signal matters.

**Proposed feature vector (~16 features):**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `n_enemies_total` | Total enemies in scene (including truncated) /20 |
| 1 | `n_allies_total` | Total allies in scene (including truncated) /10 |
| 2 | `n_enemies_truncated` | How many enemies didn't get a slot /15 |
| 3 | `n_allies_truncated` | How many allies didn't get a slot /8 |
| 4 | `enemy_centroid_x` | Mean X of all enemies /20 |
| 5 | `enemy_centroid_y` | Mean Y of all enemies /20 |
| 6 | `ally_centroid_x` | Mean X of all allies /20 |
| 7 | `ally_centroid_y` | Mean Y of all allies /20 |
| 8 | `mean_enemy_hp_pct` | Mean HP fraction of all enemies |
| 9 | `min_enemy_hp_pct` | Lowest HP enemy (focus fire signal) |
| 10 | `max_enemy_threat` | Highest individual threat score /1.0 |
| 11 | `aggregate_enemy_dps` | Sum of auto DPS of truncated enemies /200 |
| 12 | `n_projectiles_total` | Total projectiles in flight (including truncated threats) /10 |
| 13 | `enemy_spread` | Std dev of enemy positions (tight=push, wide=spread) /10 |
| 14 | `dominant_enemy_type` | One-hot or ordinal: melee/ranged/caster/mixed |
| 15 | `aggregate_cc_threat` | Sum of CC durations of truncated enemies /5000 |

**Type ID:** `5` (aggregate) — new type embedding added to encoder.

**SHM addition:** 16 × f32 = 64 bytes appended to request. Small.

### Importance-Based Slot Selection (Rust Side)

The current selection heuristic (presumably nearest/highest threat) is replaced with a
priority scoring system. Units are scored and the top-K earn individual entity slots.
The rest contribute only to the aggregate token.

**Priority score components:**

```rust
fn unit_priority_score(unit: &Unit, context: &BattleContext) -> f32 {
    let mut score = 0.0;

    // Guaranteed slots — always earn individual representation
    if unit.is_boss || unit.tier >= UnitTier::Elite {
        return f32::MAX;  // always gets a slot
    }

    // Threat to self
    score += unit.auto_dps / 30.0;
    score += unit.ability_damage / 50.0;
    score += if unit.has_ready_cc() { 0.3 } else { 0.0 };

    // Proximity (closer = higher priority, but not the only signal)
    let dist = unit.distance_from_caster;
    score += (1.0 - (dist / 20.0).min(1.0)) * 0.4;

    // Low HP enemies are high priority (focus fire opportunity)
    if unit.is_enemy && unit.hp_pct < 0.25 {
        score += 0.5;
    }

    // Allies in danger are high priority
    if unit.is_ally && unit.hp_pct < 0.3 {
        score += 0.4;
    }

    // Active casts are high priority (dodge/interrupt signal)
    if unit.is_casting {
        score += 0.6;
    }

    score
}
```

**Slot allocation order:**
1. Self (always slot 0)
2. Guaranteed units (boss/elite tier) — consume slots in score order
3. Remaining slots filled by score descending
4. All remaining units → aggregate token only

**Rust implementation location:** `src/ai/core/ability_eval/game_state.rs` — modify
entity serialization to score and sort before slot assignment, compute aggregate stats
over the truncated remainder.

### Future Direction: Per-Unit Compute Allocation

The natural extension of importance-based selection is importance-based compute
allocation inside the model. A boss enemy's entity token could pass through more
latent interface blocks before the Write step — effectively giving it more
representational depth. This requires a variable-depth forward pass and is nontrivial
to implement cleanly in PyTorch (dynamic computation graph per token), but is the
correct long-term direction. The priority scoring system in Rust is the prerequisite
— once units have explicit importance scores, routing them to different compute paths
is architecturally natural.

### SHM Changes for Aggregate Token

```
aggregate_features:  16 × f32 = 64 bytes  (new, appended to entity block)
sample_size:         +64 bytes
```

New token count in header or derived from existing counts — aggregate is always
present (1 token always), so no mask needed. Zero-fill if scene has ≤ slot count
entities (nothing was truncated).

---

## Spatial Awareness: Geometry Corner Tokens

### Overview

Structural features of room geometry are extracted and encoded as spatial tokens.
Each unit gets a per-unit view: only corners visible from its current position
are included. This gives the model awareness of cover positions, choke points,
flanking angles, and room shape — information that is currently invisible to the
entity encoder.

**Corner extraction** (`extract_corners`): Scans the grid for walkable cells
adjacent to walls where the boundary changes direction. Each corner has position,
convex/concave type, blocked neighbor count, opening direction, passage width, and
elevation. Corners sorted by passage width (narrowest = most tactically relevant).
Capped at 16 per room.

**Precomputed visibility** (`VisibilityMap`): Built once per room (~500μs). Per-cell
u32 bitset of visible corners via Bresenham LOS. Tick-time lookup is a single
HashMap get (~20ns cache hit).

**Dynamic obstacles** (`update_obstacle_placed` / `update_obstacle_removed`):
Barricade placement creates new corners at obstacle edges, updates passage widths,
recomputes local visibility. Geometry changes bump a generation counter that
invalidates the spatial cache.

### Corner Token Format (11 floats)

| Index | Feature | Range |
|-------|---------|-------|
| 0-1 | Corner position (normalized 0-1 in room) | [0, 1] |
| 2 | Convex (1) vs concave (0) | {0, 1} |
| 3 | Enclosure level (blocked_neighbors / 3) | [0, 1] |
| 4-5 | Opening direction (unit vector) | [-1, 1] |
| 6 | Passage width (normalized, capped) | [0, 1] |
| 7 | Elevation (normalized) | [0, 1] |
| 8-9 | Relative direction from unit to corner | [-1, 1] |
| 10 | Normalized distance from unit to corner | [0, 1] |

Features 0-7 are static per corner. Features 8-10 are per-unit (relative to the
querying unit's position).

### Threat Token Format (10 floats)

Ephemeral spatial tokens extracted from live game state every decision tick.
Unlike geometry corners (near-static, cached), these represent active threats
and opportunities that units must react to immediately.

**Sources:** Active zones (damage fields, healing circles, barricade obstacles),
cast indicators (AoE telegraphs with ground target position), and projectiles
in flight.

| Index | Feature | Range |
|-------|---------|-------|
| 0-1 | Position (normalized 0-1 in room) | [0, 1] |
| 2-3 | Relative direction from unit | [-1, 1] |
| 4 | Normalized distance from unit | [0, 1] |
| 5 | Radius (normalized) | [0, 1] |
| 6 | Hostile flag | {0, 1} |
| 7 | Duration remaining (seconds, capped) | [0, 1] |
| 8 | Kind (zone=0.25, obstacle=0.5, cast=0.75, projectile=1.0) | [0, 1] |
| 9 | Line-of-sight visibility from unit | {0, 1} |

**Note:** Features 2-4 and 9 are per-unit relative features. Threat tokens must
be serialized per-unit in SHM, not globally. Corner tokens have the same property
(features 8-10).

### Integration Strategy

Spatial information integrates across two phases to separate the "does it help at
all" question from the "how much expressiveness do we need" question.

**Phase 1 — Per-unit summary features (validate signal):**
Aggregate visible corners into 4 fixed-size features per entity slot, appended to
the existing entity feature vector. No architectural changes — just a wider input.

**Phase 2 — Dedicated cross-attention (full expressiveness):**
Entity tokens (Q) attend to corner tokens (K/V) in a dedicated cross-attention
layer before the latent interface. Per-unit visibility masking. Each unit gets its
own spatial attention pattern over specific corners it can see.

See Phase Plan below for implementation details.

### Tactical Behaviors Enabled

- **Cover usage:** Low-HP unit sees concave corner nearby → move toward it. Convex
  flag distinguishes cover (concave) from exposed positions (convex).
- **Choke control:** Corners with low passage_width are choke points. Holding
  position near a narrow concave corner is high-value in corridor rooms.
- **Barricade placement:** Engineer sees corner with low passage_width + concave +
  enemies approaching → place barricade to seal the choke.
- **Formation selection:** Distribution of visible corners encodes room shape
  implicitly (many narrow = corridor, few wide = open arena).
- **Zone placement:** AoE abilities most effective at choke points — passage_width
  directly indicates AoE value at a location.
- **Flanking detection:** Convex corner + enemy nearby → can be attacked from
  around the wall. Opening direction points toward the exposed side.

### Spatial Evaluation Metrics

Beyond win rate, spatial awareness should improve:
- **Damage taken in corridor rooms** — units should use cover more effectively
- **Barricade effectiveness** — barricades at choke points should block more enemies
- **Formation spread** — should correlate with room openness (corner distribution)
- **Choke control time** — time spent controlling narrow passages
- **Out-of-sight deaths** — should decrease (units aware of blind spots)

---

## Target Architecture (V6)

```
Input:
  7 entity slots × 34 features      (importance-selected, +4 spatial summary features)
  6 threat tokens × 10 features     (zones, casts, projectiles — per-unit relative)
  8 position tokens × 8 features
  1 aggregate token × 16 features   (crowd summary for truncated entities)
  8 ability CLS embeddings × 128d   (no longer projected down)

                    ┌──────────────────────────────────┐
                    │       Entity Encoder V6           │
                    │   d=128, 8 heads, 4 layers        │
                    │   pre-norm, same structure as V4  │
                    │                                   │
  Entities ────────►│   entity_proj(34→128)             │
  Threats ─────────►│   threat_proj(10→128)             │
  Positions ───────►│   position_proj(8→128)            │
  Aggregate ───────►│   agg_proj(16→128)                │
                    │   + type embeddings (6 types)     │
                    │   → TransformerEncoder(4 layers)  │
                    └────────────┬─────────────────────┘
                                 │
                         tokens (B, 22, 128)
                                 │
                    ┌────────────▼─────────────────────┐
                    │   Spatial Cross-Attention          │
                    │   (Phase 2 — zero-init at start)  │
                    │                                   │
  Corner tokens ───►│   corner_proj(11→128)             │
    (up to 8)       │   Entity tokens (Q) ×             │
                    │   Corner tokens (K,V)             │
                    │   Per-unit visibility masking      │
                    │   → spatially-enriched tokens     │
                    └────────────┬─────────────────────┘
                                 │
                         tokens (B, 22, 128)
                                 │
                    ┌────────────▼─────────────────────┐
                    │     Latent Interface (ELIT-style) │
                    │   (Phase 2)                       │
                    │                                   │
                    │   K=12 learned latent tokens      │
                    │   Read: latents attend to tokens  │
                    │   2× latent transformer blocks    │
                    │   Write: tokens updated by latents│
                    │                                   │
                    │   → pool latents → (B, 128)       │
                    └────────────┬─────────────────────┘
                                 │
                         pooled (B, 128)
                                 │
                    ┌────────────▼─────────────────────┐
                    │       CfC Temporal Cell           │
                    │   (Phase 3)                       │
                    │   input=128, hidden=256           │
                    │   replaces GRUCell                │
                    │   → proj(256→128)                 │
                    └────────────┬─────────────────────┘
                                 │
                         pooled_enriched (B, 128)
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
    Move Head              Combat Pointer            Value Head
    d=128 → 9              d=128, pointer             d=128 → 1
                           over 128d entity tokens
```

**Token count:** 22 tokens (7 entity + 6 threat + 8 position + 1 aggregate).
Increased from 20 by expanding threat slots from 4 to 6.

**Threat slot expansion rationale:** With the formalized threat token taxonomy
(zones, cast indicators, projectiles, obstacles), 4 slots is tight. A single
AoE-heavy fight can produce 2-3 zones + a cast telegraph + a projectile
simultaneously. 6 slots provides headroom. The same importance-scoring logic
used for entity slots applies: closest/most-dangerous threats get individual
slots, remainder contributes to aggregate features (n_projectiles_total etc.).

**Ability CLS embeddings:** 128d matches d_model — `external_cls_proj` (Linear 128→32)
is removed entirely. Ability embeddings feed directly into cross-attention. This removes
a lossy bottleneck that was discarding 75% of the ability representation capacity.

**Corner tokens in cross-attention:** Up to 8 visible corners per acting unit,
projected from 11d to 128d. These are NOT part of the main token sequence — they
appear only as K/V in the spatial cross-attention layer. This keeps the latent
interface sequence length at 22, not 30.

---

## Parameter Estimates

| Component | V4 | V6 (estimated) |
|-----------|-----|----------------|
| Entity Encoder | 46,848 | ~770,000 |
| Aggregate Proj | — | ~2,000 |
| Spatial Cross-Attention + corner_proj | — | ~70,000 |
| Latent Interface (K=12) | — | ~200,000 |
| CfC Cell | — | ~350,000 |
| GRU | 20,896 | removed |
| External CLS Proj | 4,224 | removed |
| Cross-Attention (ability) | 6,560 | ~50,000 |
| Move Head | 1,097 | ~17,000 |
| Combat Pointer Head | 22,122 | ~200,000 |
| Value Head | 1,089 | ~17,000 |
| **Total** | **134,548** | **~1,676,000** |

---

## Phase Plan

### Phase 1 — Scale d: 32 → 128, remove CLS bottleneck, add aggregate token, spatial summary features, update threat tokens

**Core model changes:**
- All projections widened to d=128
- All transformer blocks: d=32→128, 4 heads→8 heads
- Remove `external_cls_proj: Linear(128→32)` — ability CLS feeds directly at 128d
- Add `agg_proj: Linear(16→128)` — aggregate token projection
- Add type_id=5 embedding for aggregate token
- GRU input/output updated: GRUCell(128→64), proj Linear(64→128)
- All heads: updated to d=128

**Spatial summary features (Option 3 — validates spatial signal):**
- Entity features widened from 30→34. Four spatial summary features appended per
  entity slot: `[visible_corner_count, avg_passage_width, min_passage_width,
  avg_corner_distance]`
- `entity_proj(34→128)` instead of `entity_proj(30→128)`
- Rust side: during entity serialization, query `SpatialCache` for each slotted
  unit, compute summary features from visible corners, append to feature vector
- SHM: +16 bytes per entity slot (4 × f32) = +112 bytes total for 7 slots

**Unified zone tokens (supersedes threat token update):**

The original plan called for expanding threat tokens from 8→10 features and 4→6 slots.
During implementation, threats evolved into a **unified zone token system** that covers
all spatial points of interest in a single 12-dim representation. This is strictly more
expressive than the original plan and eliminates the need for separate threat/position
token types.

- **Zone feature dimension: 12** (ZONE_DIM=12 in `game_state_zones.rs`)
- **Up to 10 tokens per unit** (priority-tiered allocation)
- **8 zone kinds** (linearly spaced [0,1]):
  - KIND_DAMAGE=0.0, KIND_CC=0.14, KIND_OBSTACLE=0.29, KIND_PROJECTILE=0.43,
    KIND_CAST=0.57, KIND_HEALING=0.71, KIND_COVER=0.86, KIND_OBJECTIVE=1.0
- **6 hint categories** (linearly spaced [0,1]):
  - AVOID, DODGE, APPROACH, HOLD, CONTEST, NEUTRAL
- **12 features per token:** dx, dy, distance, radius, intensity, urgency, kind,
  hint, friendliness, has_cc, terrain_quality, exists
- **Priority-tiered slot allocation:**
  - Tier 0 (imminent threats): 4 slots — hostile damage/CC zones, projectiles, enemy casts
  - Tier 1 (objectives): 2 slots — capture points, extraction areas
  - Tier 2 (friendly): 2 slots — healing circles (when HP < 80%)
  - Tier 3-4 (other): remaining slots — non-imminent threats, cover positions
- **Per-unit relative features:** dx/dy/distance are relative to the querying unit
- `zone_proj(12→128)` in entity encoder, type_id=3, participates in full self-attention
- Implemented in `src/ai/core/ability_eval/game_state_zones.rs`

This replaces the legacy `threats` and `positions` fields in GameStateV2 (those fields
still exist for backward compat but are empty in V6 extraction paths).

**Rust / Burn inference changes (SHM eliminated):**

The original plan assumed a Python GPU inference server communicating via shared memory.
This was replaced by **Burn in-process GPU inference** (`BurnInferenceClientV6`), which
eliminates all SHM serialization, byte alignment, and cross-language protocol concerns.

- Priority-based entity slot selection + aggregate computation: `select_entity_slots()`,
  `compute_aggregate_features()` in `game_state_v2.rs`
- Spatial summary feature computation: `rich_entity_features_spatial()` via
  `extract_game_state_v2_cached_spatial()` — passes `VisibilityMap` + `GridNav`
- Zone token extraction: `extract_zone_tokens()` in `game_state_zones.rs`
- Corner token extraction: `VisibilityMap::spatial_tokens_for_unit()` → 8 × 11-dim
- CLS embeddings: packed per-ability from `InferenceRequest.ability_cls`
- Burn tensors built directly from request structs — no byte packing
- Double-buffered batching in `inference_v6.rs` (same pattern as V5 Burn client)
- Checkpoint save/load: `checkpoint.rs` using `BinFileRecorder`
- CLI: `--burn-v6 [--burn-checkpoint PATH]`

**Training notes:**
- Train from scratch — no weight transfer possible from V4
- Lower LR: if V4 used 1e-3, target ~5e-4 (LR ∝ 1/√d_model)
- Watch combat head early: ability CLS now feeds raw 128d instead of projected 32d.
  Expect slightly noisier pointer head for first few thousand steps during calibration.
- Profile training step time at d=128 — may be able to increase batch size further

**Validation:**
- Confirm policy quality clearly exceeds V4 within equivalent wall-clock time
- Confirm attention weights are non-uniform (heads are specializing)
- **Spatial signal check:** Compare corridor room damage taken with and without
  the 4 spatial summary features. If no improvement, skip Phase 2 cross-attention
  and save the complexity. If improvement, the ceiling is higher with per-corner
  detail.

---

### Phase 2 — Add Spatial Cross-Attention + Latent Interface

Two components added to the model. Neither requires SHM changes relative to each
other, but spatial cross-attention adds a new SHM block for corner tokens.

#### Spatial Cross-Attention (Option 4 — full spatial expressiveness)

Inserted between entity encoder output and latent interface. Entity tokens attend
to visible corner tokens via dedicated cross-attention. This is gated on Phase 1
spatial validation — only proceed if summary features showed measurable improvement.

**SHM addition:**
- Corner tokens for acting unit: up to 8 corners × 11 floats × f32 = 352 bytes
- Zero-padded to MAX_CORNERS=8, masked in attention for units seeing fewer corners
- Separate SHM block from entity/threat data (corner tokens are near-static,
  can skip writes on cache hits via generation counter check)

**Python changes:**
```python
class SpatialCrossAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=8):
        super().__init__()
        self.corner_proj = nn.Linear(11, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        # Zero-init output proj → identity passthrough at init
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, entity_tokens, corner_tokens, corner_mask=None):
        # entity_tokens: (B, 22, 128) — main token sequence
        # corner_tokens: (B, 8, 11) — visible corners for acting unit
        # corner_mask: (B, 8) — True for padded/invisible corners
        corners = self.corner_proj(corner_tokens)  # (B, 8, 128)
        updated, attn_weights = self.cross_attn(
            entity_tokens, corners, corners,
            key_padding_mask=corner_mask
        )
        return self.norm(entity_tokens + updated), attn_weights
```

**Zero-init rationale:** At initialization, the cross-attention output projection
is all zeros, so the residual connection passes entity tokens through unchanged.
The spatial cross-attention trains from identity — same principle as the latent
interface Write step. This means adding it cannot regress performance at init.

**Rust changes:**
- `game_state.rs`: Serialize acting unit's visible corner tokens into SHM
- `gpu_client.rs`: Write corner token block to SHM
- `gpu_inference_server.py`: Parse corner token block from SHM

**Design decision: Corner tokens stay separate from the main sequence.** Corners
appear only as K/V in cross-attention, not as additional tokens in the entity
encoder self-attention. This keeps the latent interface sequence at 22 tokens
(no growth) and maintains clean separation: the latent interface compresses
entity-level tactical information from spatially-enriched entity tokens, rather
than also having to compress raw geometry.

#### Latent Interface

Inserted after spatial cross-attention output, before pool→CfC step.
No SHM or Rust changes required.

```python
class LatentInterface(nn.Module):
    def __init__(self, d_model=128, n_latents=12, n_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)

        self.read = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.write = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.latent_block_1 = TransformerBlock(d_model, n_heads)
        self.latent_block_2 = TransformerBlock(d_model, n_heads)

        self.read_norm = nn.LayerNorm(d_model)
        self.write_norm = nn.LayerNorm(d_model)

        # Zero-init write output proj → identity at init, trains from there
        nn.init.zeros_(self.write.out_proj.weight)
        nn.init.zeros_(self.write.out_proj.bias)

    def forward(self, entity_tokens, n_latents_override=None):  # (B, 22, 128)
        B = entity_tokens.shape[0]
        L = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, 12, 128)

        # Tail dropping: use subset of latents (training: random, inference: budget)
        if n_latents_override is not None:
            L = L[:, :n_latents_override, :]
        elif self.training:
            J = random.choice([4, 6, 8, 10, 12])
            L = L[:, :J, :]

        # Read: latents attend to (spatially-enriched) entity tokens
        L, _ = self.read(L, entity_tokens, entity_tokens)
        L = self.read_norm(L)
        L = self.latent_block_1(L)
        L = self.latent_block_2(L)

        # Write: entity tokens updated by latents
        updated, _ = self.write(entity_tokens, L, L)
        entity_tokens = self.write_norm(entity_tokens + updated)

        # Pool latents for temporal cell input (not entity tokens)
        pooled = L.mean(dim=1)  # (B, 128)
        return entity_tokens, pooled
```

**Why pool latents (not entity tokens):** The latent tokens are a coordinated tactical
summary — they've already compressed the important cross-entity information. Entity tokens
still go to the pointer heads for targeting (they need per-entity resolution).

**Why K=12:** 22 input tokens → 12 latents is a moderate compression ratio. At d=128
the latent self-attention is cheap and more slots allow more specialized tactical
representations to emerge. ELIT found diminishing returns at high K — 12 is a reasonable
starting point to tune from.

#### Tail Dropping

Tail dropping is a training trick that makes the latent interface work at variable
compute budgets using a single set of weights.

**During training:** each forward pass, randomly sample J̃ from {4, 6, 8, 10, 12}
and drop the last (K - J̃) latent tokens before the Read/Write steps. One batch
might run with 12 latents, the next with 6, the next with 10.

```python
# In LatentInterface.forward():
J = random.randint(J_min, K)  # sampled once per training iteration
L = L[:, :J, :]               # drop tail latents
# then proceed with Read/Write using only J latents
```

**At inference:** pick J̃ based on compute budget.

```python
J = self.inference_budget  # fixed or dynamically chosen
L = L[:, :J, :]
```

**Why it works:** Early latent tokens get trained on every batch regardless of J̃,
while later tokens only train when J̃ is large. The model is forced to front-load
the most important information into early slots because those are always present.
Later slots learn to refine and add detail, but the representation is valid without
them. No auxiliary losses or special architecture — the asymmetric training frequency
does all the work.

**Tactical priority mapping:** The importance ordering that emerges maps naturally
onto tactical priority. Early latents will learn to represent high-urgency information
(immediate threats, low-HP units, active casts) because that information is always
penalized if missing. Later latents pick up contextual nuance. This can be verified
by inspecting Read attention weights per latent slot to see what each one specializes
on.

**Inference spectrum:** 4 latents = fast and coarse (suitable for low-priority units
or high-load ticks), 12 latents = full quality (for critical decisions). A single
set of weights serves the entire compute spectrum.

**Validation:**
- **Spatial cross-attention weights:** Should show interpretable per-unit patterns.
  Engineer tokens should attend heavily to low-passage-width concave corners.
  Low-HP unit tokens should attend to nearby concave corners (cover-seeking).
  If attention is flat/uniform, spatial signal is not being used effectively.
- **Latent Read attention weights:** Should be non-uniform and interpretable
  (e.g. one latent attending heavily to low-HP allies, another to nearby enemies).
  If flat after several thousand steps, increase K or add diversity regularization.
- **Spatial metrics:** Corridor room damage taken, barricade effectiveness,
  choke control time (see Spatial Evaluation Metrics above).

---

### Phase 3 — Replace GRU with CfC, scale h_dim: 64 → 256

**Changes:**
- Python: swap GRUCell for CfCCell
- SHM: h_dim 64→256, hidden_state_in/out 256→1024 bytes
- Rust: update h_dim in header parsing, update SHM buffer sizes

```python
class CfCCell(nn.Module):
    def __init__(self, input_size=128, hidden_size=256):
        super().__init__()
        total = input_size + hidden_size
        self.f_gate = nn.Linear(total, hidden_size)
        self.h_gate = nn.Linear(total, hidden_size)
        self.t_a    = nn.Linear(total, hidden_size)
        self.t_b    = nn.Linear(total, hidden_size)
        self.proj   = nn.Linear(hidden_size, input_size)

        # Init: start near full memory retention, let it learn to forget
        nn.init.constant_(self.f_gate.bias, 1.0)
        nn.init.constant_(self.t_b.bias, 1.0)

    def forward(self, x, h, delta_t=1.0):
        combined = torch.cat([x, h], dim=-1)
        f = torch.sigmoid(self.f_gate(combined))
        candidate = torch.tanh(self.h_gate(combined))
        t = torch.sigmoid(self.t_a(combined)) * delta_t + self.t_b(combined)
        h_new = torch.tanh(f * h + (1 - f) * candidate * t)
        return self.proj(h_new), h_new
```

**SHM layout changes:**
```
hidden_state_in:  256 × f32 = 1024 bytes  (was 64 × f32 = 256 bytes)
hidden_state_out: 256 × f32 = 1024 bytes  (was 64 × f32 = 256 bytes)
sample_size:      increases by 1536 bytes
h_dim header:     256 (was 64)
```

**Rust changes:**
- Update h_dim constant / header read
- Update SHM buffer allocation for hidden_state_in/out
- `transformer_rl.rs`: per-unit hidden state buffer grows from 256→1024 bytes

**delta_t:** Keep at 1.0 for now. The input-dependent time constant is the main
immediate benefit — true delta_t scheduling is Phase 4.

---

### Phase 4 — delta_t via SHM (optional, post-validation)

Add 1 × f32 to request for ticks-since-last-meaningful-event.

**Meaningful event definition (Rust side):**
- Damage taken or dealt above threshold
- Ability fired
- CC applied or received
- Unit death in scene

**SHM change:** +4 bytes to request. Append after `hidden_state_in` or use a
dedicated field in the header region.

**Payoff:** A unit that has been kiting for 20 ticks sends delta_t=20. The CfC
cell naturally decays short-term tactical memory, reducing the signal from stale
positional history. Tactically relevant for: tracking cooldown cycles, remembering
ability timing, recognizing re-engagement vs. sustained engagement.

---

## SHM Layout Summary (Historical — superseded by Burn)

> **Note:** The SHM protocol was eliminated by the Burn migration. The model runs
> in-process via LibTorch — no shared memory, no byte packing, no Python server.
> `InferenceRequest` fields are converted directly to Burn tensors in `inference_v6.rs`.
> This section is retained for historical reference.

| Phase | Change | Delta |
|-------|--------|-------|
| 1 | Aggregate features (16 × f32) | +64 bytes |
| 1 | Spatial summary features (7 slots × 4 × f32) | +112 bytes |
| 1 | Zone tokens (10×12 features × f32) | +480 bytes |
| 2 | Corner tokens (8 × 11 × f32) | +352 bytes |
| 3 | Hidden state expansion (2 × (256-64) × f32) | +1536 bytes |
| 4 | delta_t (1 × f32) | +4 bytes |

**With Burn, caching is simpler:** The `VisibilityMap` is built once per scenario and
stored on `ActiveSim`. Corner tokens are computed per-unit per-tick via
`spatial_tokens_for_unit()` — a single HashMap lookup (~20ns) for the visibility
bitset plus feature composition for visible corners.

---

## Risk / Rollback Summary

| Phase | Changes | Rollback path |
|-------|---------|---------------|
| 1 — d=128, aggregate, spatial summary, zones | Burn entity encoder, zone extraction, aggregate computation | Use ActorCriticV5 (V5 Burn model still exists alongside V6) |
| 2 — Spatial Cross-Attention + Latent Interface | New Burn modules, corner token pipeline | Zero-init output projections = identity at init; pass `None` for corners to skip spatial |
| 3 — CfC + h_dim=256 | CfC cell config, hidden state buffers | Change V6_H_DIM back to 64 in ActorCriticV6Config |
| 4 — delta_t | Not yet implemented | N/A (hardcoded delta_t=1.0) |

---

## Open Questions / Decisions Deferred

- **K (latent count):** Starting at 12, tune based on Read attention diversity
- **n_latent_blocks:** Starting at 2, could increase if GPU headroom confirms
- **Pointer head LR:** May need separate lower LR or delayed unfreeze if it collapses
  early due to raw 128d CLS input (vs. previously projected 32d)
- **Batch size at d=128:** Profile training step time — may be able to increase
  beyond 1024 for better gradient estimates given the underutilized GPU
- ~~**Whether to do Phases 1+2 together or sequentially**~~ → **RESOLVED:** All phases
  implemented together. Zero-init output projections on spatial cross-attn and latent
  interface write step mean they start as identity — no regression risk at init.
- ~~**Threat slot count**~~ → **RESOLVED:** Unified zone token system with 10 slots,
  priority-tiered (4 imminent + 2 objective + 2 friendly + 2 other). 12-dim features
  cover all zone kinds including cover positions and objectives.
- ~~**Corner token caching**~~ → **RESOLVED:** `VisibilityMap` built once per scenario
  on `ActiveSim` init. Per-unit corner tokens computed via bitset lookup (~20ns) +
  feature composition. No per-tick caching needed.
- **Spatial cross-attention for all units vs. acting unit only:** Current implementation
  serializes corners only for the acting unit. If the combat pointer head needs
  spatial context for targeting (e.g., "target the enemy behind the corner"),
  all entity tokens may need spatial enrichment. Defer until training evaluation.
- **Burn training loop:** Whether to implement IMPALA V-trace entirely in Rust using
  `Autodiff<LibTorch>`, or use a hybrid approach with Python for gradient updates.
  The Burn infrastructure supports either path.

---

## Key Files (Burn Implementation)

The original plan targeted Python model definitions + SHM protocol. The actual
implementation uses **Burn (Rust ML framework)** for both inference and model
definition, eliminating the Python GPU server and SHM entirely.

### Burn Model (new files)

| File | Purpose |
|------|---------|
| `src/ai/core/burn_model/actor_critic_v6.rs` | V6 top-level model: encoder → spatial cross-attn → latent interface → CfC → heads |
| `src/ai/core/burn_model/latent_interface.rs` | ELIT-style latent interface: K=12 latents, Read/Write cross-attention, tail dropping |
| `src/ai/core/burn_model/spatial_cross_attn.rs` | Corner token cross-attention (11-dim corners as K/V, zero-init output) |
| `src/ai/core/burn_model/value_head.rs` | Two-headed value prediction (attrition ratio + survival ticks) |
| `src/ai/core/burn_model/checkpoint.rs` | Save/load via Burn's BinFileRecorder |
| `src/ai/core/burn_model/inference_v6.rs` | Double-buffered GPU inference client for V6 |

### Existing files modified

| File | Change |
|------|--------|
| `src/ai/core/burn_model/mod.rs` | Added V6 modules, checkpoint, inference_v6 |
| `src/ai/core/burn_model/config.rs` | Constants: D_MODEL=128, N_HEADS=8, H_DIM=64, N_LATENT_TOKENS=12 |
| `src/ai/core/burn_model/inference.rs` | Added CLS embedding packing (was stubbed as vec![None]) |
| `src/ai/core/ability_eval/game_state_v2.rs` | Added `extract_game_state_v2_cached_spatial()` with VisibilityMap support |
| `src/ai/core/ability_transformer/gpu_client.rs` | Added `corner_tokens` field to InferenceRequest |
| `src/ai/goap/spatial.rs` | Corner extraction, VisibilityMap, spatial_tokens_for_unit() (11-dim) |
| `src/bin/xtask/oracle_cmd/rl_gpu.rs` | Wired spatial features + corner tokens + zone tokens into InferenceRequest |
| `src/bin/xtask/oracle_cmd/rl_gpu_sim.rs` | Added `visibility_map` field to ActiveSim, built at init |
| `src/bin/xtask/oracle_cmd/rl_generate.rs` | Added `--burn-v6` and `--burn-checkpoint` policy paths |
| `src/bin/xtask/oracle_cmd/transformer_rl.rs` | Added Policy::BurnServerV6 variant |
| `src/bin/xtask/cli/scenario.rs` | Added `--burn-v6` and `--burn-checkpoint` CLI flags |
| `Cargo.toml` | Added `std` feature to burn dependency (for file-based checkpoints) |
| `training/impala_learner_v6.py` | IMPALA training loop using `--burn-v6` for episode generation |

---

## Implementation Status

### Phase 1 — COMPLETE

| Component | Status | Notes |
|-----------|--------|-------|
| d=128, 8 heads | ✅ | D_MODEL=128, N_HEADS=8 in burn_model/config.rs |
| Entity features 30→34 (spatial summary) | ✅ | rich_entity_features_spatial() via VisibilityMap |
| Unified zone tokens (12-dim, 10 slots) | ✅ | Supersedes original threat token plan; 8 kinds, tiered priority |
| Aggregate token (16-dim) | ✅ | compute_aggregate_features(), agg_proj(16→128), type_id=4 |
| Priority-based entity slot selection | ✅ | select_entity_slots() with threat/proximity/HP/casting scoring |
| Ability CLS at 128d (no projection) | ✅ | CLS embeddings packed directly in inference, external_cls_proj optional |
| Burn in-process GPU inference | ✅ | BurnInferenceClientV6, double-buffered, no SHM |

### Phase 2 — COMPLETE

| Component | Status | Notes |
|-----------|--------|-------|
| Spatial cross-attention | ✅ | Corner tokens (8 × 11-dim) as K/V, zero-init output proj |
| Latent interface (ELIT) | ✅ | K=12 latents, Read/Write cross-attention, 2 latent blocks |
| Tail dropping | ✅ | n_latents_override parameter for variable compute budget |
| Corner token extraction | ✅ | VisibilityMap::spatial_tokens_for_unit(), cached per scenario |
| Per-unit visibility masking | ✅ | Bresenham LOS, u32 bitset per cell |

### Phase 3 — COMPLETE

| Component | Status | Notes |
|-----------|--------|-------|
| CfC temporal cell | ✅ | CfCCell in burn_model/cfc_cell.rs, replaces GRU |
| h_dim=256 | ✅ | V6_H_DIM=256 in actor_critic_v6.rs (V5 retains H_DIM=64) |

### Phase 4 — NOT STARTED (optional, post-validation)

| Component | Status | Notes |
|-----------|--------|-------|
| delta_t via request | ❌ | Hardcoded delta_t=1.0; event-based timing deferred |

### Infrastructure — COMPLETE

| Component | Status | Notes |
|-----------|--------|-------|
| Checkpoint save/load | ✅ | BinFileRecorder, roundtrip test passing |
| CLI integration | ✅ | --burn-v6, --burn-checkpoint flags |
| Value head (curriculum) | ✅ | Two-headed: attrition ratio + survival ticks |
| 12 Burn model tests | ✅ | Including checkpoint roundtrip + V-trace |

### Training — COMPLETE (all Rust, no Python)

Staged pretraining (0a-0e) and graduated unfreezing (1a-1e) abandoned — pretraining
didn't converge despite hours of training. Straight to IMPALA V-trace RL with full
unfreeze from step 1. Only regularization: latent interface tail dropping.

**Python eliminated entirely.** No PyTorch model, no SHM protocol, no export scripts.
Episode generation + gradient updates both happen in-process via Burn `Autodiff<LibTorch>`.

| Component | Status | Notes |
|-----------|--------|-------|
| ~~Stages 0a-0e: pretraining~~ | SKIPPED | Didn't converge in practice |
| ~~Stages 1a-1d: graduated unfreezing~~ | SKIPPED | Going straight to full unfreeze |
| End-to-end IMPALA RL | ✅ | `training.rs`: V-trace, policy+move+value loss, AdamW, tail dropping |
| Episode generation | ✅ | `--burn-v6 --burn-checkpoint` CLI ready, records continuous target_move_pos |
| Burn training loop | ✅ | `impala_train.rs`: full loop with no-grad value prediction for V-trace |
| CLI command | ✅ | `xtask scenario oracle transformer-rl impala-train` |
| Continuous movement | ✅ | MSE loss on target_pos [B, 2]; no discrete 9-way move head |
| Value prediction for V-trace | ✅ | No-grad forward pass via model.valid() (inner backend) |
| Python training code | DELETED | `training/impala_learner_v6.py` removed |
