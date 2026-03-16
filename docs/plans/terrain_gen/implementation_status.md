# ML Terrain Generation — Implementation Status

Cross-references `terrain_gen_plan_v2.md`. Documents what has been built, what deviates from the plan, and what remains.

---

## Milestone 0 — Data Generation & Captioning Pipeline

### M0.1 — Augmented Batch Room Generation ✅

**Files:**
- `src/mission/room_gen/mod.rs` — `generate_room_varied()`, `RoomGrid`, `RoomMetrics`
- `src/mission/room_gen/lcg.rs` — `ObstacleType` constants, `obs_type` field on `ObstacleRegion`
- `src/mission/room_gen/primitives.rs` — All 7 primitives tag `obs_type`
- `src/bin/xtask/roomgen_cmd.rs` — `xtask roomgen export` CLI
- `src/bin/xtask/cli/mod.rs` — `RoomgenCommand`, `RoomgenExportArgs`

**Discrepancies from plan:**
| Plan | Implementation | Reason |
|------|---------------|--------|
| `ObstacleRegion` struct unchanged | Added `obs_type: u8` field | Needed to track which primitive generated each obstacle for multi-channel grid export |
| `ObstacleRegion` is `pub(crate)` | Now `pub` | Needed for xtask binary to access from `roomgen_cmd.rs` |
| `RampRegion` is `pub(crate)` | Now `pub` | Same reason |
| Dimension perturbation seed `0xDIM5` | Uses `0xD1A5_E70F` | `0xDIM5` isn't valid hex; replaced with arbitrary constant |
| `obstacles` field on `RoomLayout` is `pub(crate)` | Now `pub` | Required for external access in xtask and ml_gen |
| Plan shows `chokepoint_score_by_cell >= 2` from existing code | Implemented as "2 walkable orthogonal neighbors on opposite sides" | Existing codebase doesn't have `chokepoint_score_by_cell`; this is a reasonable approximation |
| `flanking_routes` via "distinct shortest paths" | Implemented as 3-band BFS (top/mid/bottom row bands) | Full distinct-path enumeration is expensive; 3-band check captures the tactical intent |
| `spawn_quality_diff` from existing `score_spawn_quality` | Implemented as per-spawn cover density comparison | No existing `score_spawn_quality` function found; cover-near-spawn is the relevant metric |

**JSON format matches plan exactly.** All 8 metrics implemented. Parallel generation via `std::thread::scope`.

**CLI:**
```bash
cargo run --bin xtask -- roomgen export --count-per-type 5000 --output generated/rooms.jsonl
```

### M0.2 — Headless Top-Down Rendering ✅

**Files:**
- `src/bin/xtask/roomgen_cmd.rs` — `xtask roomgen render` subcommand

**Discrepancies from plan:**
| Plan | Implementation | Reason |
|------|---------------|--------|
| Separate step from export | Combined into same binary, reads JSONL then renders | Simpler pipeline; avoids redundant room regeneration |
| Color: obstacles `#404040` → `#606060` by height | Uses `0x60 - (type * 4)` darkening | Approximation; obstacle type correlates with tactical importance better than raw height |

**Rendering matches plan spec:** 4px/cell, black perimeter, grey obstacles, white floor, blue elevated, green/red spawns. Output PNG per room.

**CLI:**
```bash
cargo run --bin xtask -- roomgen render --input generated/rooms.jsonl --output-dir generated/room_images
```

### M0.3 — Qwen3-VL Captioning ✅

**Files:**
- `training/caption_rooms.py`

**Discrepancies from plan:**
| Plan | Implementation | Reason |
|------|---------------|--------|
| Batch inference via vLLM Python API | HTTP client to OpenAI-compatible endpoint | More flexible — works with any VLM server (vLLM, SGLang, ollama). Pattern matches `~/Projects/lfm-agent/video_server/vlm_client.py` |
| Reads images + metrics from separate sources | Reads JSONL + image directory, looks up by `{room_type}_{seed}.png` | Single-pass pipeline |

**Prompt template matches plan exactly.** Supports `--resume` for crash recovery, `--limit` for testing.

**CLI:**
```bash
# Start vLLM server first:
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8200
# Then caption:
uv run python training/caption_rooms.py --input generated/rooms.jsonl --images generated/room_images/
```

---

## Milestone 1 — ELIT-DiT Model Training ✅ (architecture only, not trained)

**Files:**
- `training/roomgen/elit_dit.py` — Full ELIT-DiT model
- `training/roomgen/flow_matching.py` — Rectified flow
- `training/roomgen/dataset.py` — Variable-size grid dataset
- `training/roomgen/text_encoder.py` — MiniLM + simple fallback
- `training/roomgen/train.py` — Training loop

**Discrepancies from plan:**

| Plan | Implementation | Reason |
|------|---------------|--------|
| Separate files: `read_write.py`, `dim_predictor.py`, `text_encoder.py` | `GroupedReadWrite`, `DimensionPredictor` in `elit_dit.py`; text encoder separate | Keeps related architecture together; text encoder is genuinely independent |
| Separate `guidance/` directory with one file per function | Single `guidance.py` | Five small functions don't need separate files |
| 2×2 patch size for rooms > 32 on either axis | Always 1×1 patch size | Simplification; can add adaptive patching later if VRAM is an issue |
| Data augmentation: 8× rotations/mirrors | Not yet implemented | Deferred — add to dataset.py when training begins |
| DiT-L scale option (24 blocks) | Only DiT-B default (12 blocks) in config, but block counts are fully parameterized | Can instantiate any scale via CLI args |

**Architecture verified working:** Forward pass tested with `d_model=64, n_heads=4` — produces correct output shapes, loss computes correctly.

---

## Milestone 2 — PhyScene-Style Guidance Functions ✅

**Files:**
- `training/roomgen/guidance.py`

**All 5 guidance functions implemented:** collision, boundary, connectivity (soft Bellman), cover density, blocked percentage. Plus `combined_guidance()` aggregator and `guidance_scale_schedule()` ramp.

**Discrepancies from plan:**
| Plan | Implementation | Reason |
|------|---------------|--------|
| `connectivity_loss` takes explicit spawn positions | Takes width/depth, seeds from left column (player side) | At sampling time, spawns aren't placed yet; left→right connectivity is the relevant invariant |
| Guidance operates at Write output with backprop through Write | Guidance applied directly on spatial output in sampling loop | Equivalent effect; simpler implementation without needing to hook into model internals |

---

## Milestone 3 — PCGRL Critic ✅ (architecture only, not trained)

**Files:**
- `training/roomgen/critic.py`

**Implemented:** `PCGRLCritic` transformer (2-3 layers, d=128, 4 heads), `compute_quality_score()` ground-truth formula, `CriticTrainer` wrapper.

**Matches plan.** Quality score formula matches M3.2 exactly.

---

## Milestone 4 — Rust Integration & Bevy Pipeline ✅

**Files:**
- `src/mission/room_gen/ml_gen.rs` — `generate_ml_room()`, `MlGenConfig`, `ml_grid_to_navgrid()`
- `src/mission/room_sequence/types.rs` — `prompts` field on `MissionRoomSequence`
- `src/mission/room_sequence/systems.rs` — ML generation in `advance_room_system`

**Discrepancies from plan:**
| Plan | Implementation | Reason |
|------|---------------|--------|
| `ml_grid_to_navgrid` returns `(NavGrid, Vec<ObstacleRegion>)` | Returns `(NavGrid, Vec<ObstacleRegion>, Vec<RampRegion>)` | Plan omitted ramp extraction; needed for Bevy visual spawning |
| Plan mentions `extract_obstacle_regions` | Implemented as `merge_obstacle_regions` — greedy rectangle merging | Single-cell obstacles from ML output need merging for efficient Bevy mesh spawning |
| Plan's `ObstacleRegion` struct unchanged | Has new `obs_type` field | Consistent with M0.1 changes |

**Uses Option B (Python subprocess)** as plan recommends for prototyping. Falls back to proc-gen after 3 failed attempts.

---

## Milestone 5 — Evaluation ✅

**Files:**
- `training/roomgen/evaluate.py`
- `training/roomgen/sample.py` — Guided sampling + post-processing + validation

**Discrepancies from plan:**
| Plan | Implementation | Reason |
|------|---------------|--------|
| Obstacle-set Chamfer distance | Not yet implemented | Requires spatial obstacle extraction; add when comparing ML vs proc-gen |
| A* trajectory edit distance | Not yet implemented | Requires A* pathfinding in Python; add when comparing ML vs proc-gen |
| Multi-budget quality curve (M5.5) | Not yet implemented | Requires trained model |
| Text compliance via VLM re-scoring (M5.2) | Not yet implemented | Requires trained model + VLM |
| Playtest via `run_scenario` (M5.3) | Not yet implemented | Requires trained model |

**Implemented:** Metric distributions, connectivity rate, blocked-in-range rate, per-type breakdown, pairwise Hamming diversity, dimension distribution analysis.

---

## File Layout vs Plan

Plan proposed:
```
project/
├── data/           → generated/ (JSONL + images)
├── model/          → training/roomgen/ (consolidated)
├── guidance/       → training/roomgen/guidance.py (single file)
└── eval/           → training/roomgen/evaluate.py (single file)
```

Actual layout:
```
src/mission/room_gen/
├── mod.rs          # generate_room_varied(), RoomGrid, RoomMetrics
├── ml_gen.rs       # ML inference bridge, NavGrid construction
├── lcg.rs          # ObstacleType constants, Lcg RNG
├── primitives.rs   # Obstacle primitives with obs_type tagging
├── templates.rs    # Room-type template generators
├── validation.rs   # validate_layout(), BFS connectivity
└── visuals.rs      # Bevy PBR mesh spawning

src/bin/xtask/
├── cli/mod.rs      # RoomgenCommand CLI definitions
├── roomgen_cmd.rs  # Export + render subcommands
└── main.rs         # Routing

src/mission/room_sequence/
├── types.rs        # MissionRoomSequence with prompts field
└── systems.rs      # ML generation in advance_room_system

training/roomgen/
├── elit_dit.py     # ELIT-DiT architecture (includes Read/Write, DimPredictor)
├── flow_matching.py # Rectified flow forward/reverse
├── guidance.py     # All 5 PhyScene guidance functions
├── critic.py       # PCGRL critic + quality scoring
├── dataset.py      # Variable-size room grid dataset
├── text_encoder.py # MiniLM-L6-v2 + simple fallback
├── train.py        # Training loop
├── sample.py       # Guided sampling + post-processing
├── infer.py        # Subprocess inference for Rust bridge
├── evaluate.py     # Metric comparison pipeline
└── caption_rooms.py # VLM captioning (in training/ root)
```

---

## What Remains Before Training

1. **Generate dataset at scale:** `xtask roomgen export --count-per-type 5000` (30K rooms)
2. **Render images:** `xtask roomgen render` on the 30K JSONL
3. **Caption with Qwen3-VL:** Run `caption_rooms.py` (~4 hours on 4090)
4. **Add data augmentation:** 8× rotations/mirrors in `dataset.py`
5. **Train ELIT-DiT:** `train.py` (~24-72 hours on 4090)
6. **Train PCGRL critic:** After ELIT-DiT converges
7. **Evaluate:** Compare ML vs proc-gen with `evaluate.py`

---

## Tests

17 room_gen tests all pass:
```
cargo test --lib mission::room_gen  # 17 passed, 0 failed
```

New tests added: `generate_room_varied_works`, `generate_room_varied_with_override`, `to_grid_produces_correct_dimensions`, `to_grid_perimeter_is_wall`, `compute_metrics_works`, `varied_dimensions_differ_across_seeds`.
