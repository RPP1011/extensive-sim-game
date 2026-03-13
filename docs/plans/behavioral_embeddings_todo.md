# Behavioral Embeddings TODO

## Status: Step 3 complete — 54.4% HvH win rate

### Step 1: Behavioral Profiling Data Generator ✅
- [x] **1.1** Add `AbilityProfile` subcommand to CLI
- [x] **1.2** Create `src/bin/xtask/oracle_cmd/ability_profile.rs`
  - Loads from `dataset/abilities/**/*.ability`, `assets/hero_templates/*.toml`, `assets/lol_heroes/*.toml`
  - 943 unique abilities loaded, deduplicated by name
- [x] **1.3** Controlled sim runner
  - 1 caster + N targets, condition grid: HP×distance×targets×armor = 144 combos
  - SelfAoe uses `AbilityTarget::Position` so cones/lines get direction
  - Peak status tracking across all ticks (stun, slow, root, silence, etc.)
  - Caster state changes tracked (self-heal, dash, self-buff)
- [x] **1.4** Outcome vectors: 119-dim (4 targets × 23 + caster × 23 + 4 aggregates)
- [x] **1.5** npz output: ability_id, condition(4), outcome(119), ability_names, dsl_texts
- [x] **1.6** Validated: 135,792 samples from 943 abilities

### Step 2: Behavioral Ability Encoder ✅
- [x] **2.1** ~~Lookup-based encoder~~ → Pivoted to transformer-based
  - Initial: `nn.Embedding(943, 32)` + MLP predictor. Worked (val_mse=0.60) but can't handle new abilities.
  - Pivot: Finetune transformer CLS with behavioral objective. DSL → transformer → CLS(128d) → outcome(119d).
  - Any new ability gets a behavioral embedding from a single forward pass. Known abilities cached in registry.
- [x] **2.2** Train d=128 transformer with curriculum: MLM first, then behavioral
  - Phase 1: MLM-only → 97.5% acc, 99.2% recon, 99.6% hint (20K steps, 5 min)
  - Phase 2: Frozen encoder + behavioral head with z-normed outcomes + Huber loss
  - Behavioral MSE: 0.01 z-normed (99% variance explained vs naive baseline of 4610)
  - Key learnings:
    - Joint training (MLM+behavioral) destroys MLM — behavioral MSE dwarfs MLM loss
    - Unfrozen encoder with behavioral loss → catastrophic forgetting (97.5% → 46%)
    - Solution: curriculum (MLM first) + frozen encoder + z-norm + Huber loss
  - Added `--freeze-encoder` flag to `pretrain.py`
  - Checkpoint: `generated/ability_transformer_pretrained_v6_base.pt` (MLM), `_v6.pt` (+ behavioral)
- [x] **2.3** Export embeddings registry
  - Script: `training/export_embedding_registry.py`
  - Output: `generated/ability_embedding_registry.json` (2.5 MB, 943 abilities × 128d)
  - Includes model hash (`f20edd76dfe80c9e`) and outcome normalization stats
- [x] **2.4** Evaluate CLS quality
  - kNN@5 hint classification: **99.4%**
  - Nearest-neighbor same-hint: **99.5%** (938/943)
  - Behavioral MSE: 0.01 z-normed (99% variance explained)

### Step 3: Policy Integration (in progress)
- [x] **3.1** Update `AbilityActorCritic` to accept behavioral embeddings
  - Added `external_cls_dim` param → creates `external_cls_proj` Linear(128→32)
  - `project_cls()` method projects external embeddings before cross-attention
  - Updated `forward()` and `forward_policy()` to call `project_cls()`
- [x] **3.2** Episode format: added `unit_ability_names` to `RlEpisode` for registry lookup
  - Rust: `transformer_rl.rs` stores `slot.def.name` per ability
  - Python: `train_rl.py` looks up CLS from registry by name (with space→underscore normalization)
- [x] **3.3** Rust runtime: `EmbeddingRegistry` in `weights.rs`
  - `from_json()`/`from_file()` loads registry, `get(name)` → `Option<&[f32]>`
  - `project_external_cls()` on `ActorCriticWeights` projects 128→d_model
  - CLI: `--embedding-registry` flag on `transformer-rl generate/eval`
  - Episode gen/eval: prefer registry lookup, fall back to transformer CLS
- [x] **3.4** PPO iteration 1 with behavioral embeddings
  - Generated 2040 HvH episodes (56.3% win rate in bootstrap data)
  - Trained with `--embedding-registry` — 28/28 unique abilities matched from registry
  - Exported: `generated/actor_critic_weights_beh.json`
  - **Results (204 HvH scenarios):**

| System | Wins | Losses | Timeouts | Win% |
|--------|------|--------|----------|------|
| Baseline V2 (d=32 transformer CLS) | 26 | 178 | 0 | **12.7%** |
| **Behavioral embeddings (128d→32d)** | **102** | **100** | **2** | **50.0%** |

- [x] **3.5** Iterative RL training (PPO → REINFORCE)
  - PPO collapses every time (value head trained on biased data → garbage GAE advantages)
  - DAgger (iterative BC on wins): 33.8% → 25% (overfits on wins-only data)
  - **REINFORCE** (reward-weighted CE, no value function): best approach
  - Best recipe: BC 30ep on expert wins → REINFORCE 1ep (lr=5e-7, entropy=0.05, unfrozen transformer)
  - **Result: 54.4% HvH** (106W/92L/1T on 204 scenarios)
  - On-policy iterations don't help (~48.5%) — near-50% WR gives weak gradient signal
  - Reward shaping (HP differential `--reward-shaping`): implemented but didn't help
  - Script: `scripts/reinforce_hvh.sh`
- [x] **3.6** Evaluate on 28 attrition scenarios
  - **12% win rate** — model is HvH-specialized, doesn't transfer
  - Previous "96.4%" V2 result was incorrect/unreproducible (all V2 models get 8-12%)

### Step 4: Push Beyond 54% (next)
- [x] **4.1** Larger model: d_model=64 cross-attention (102K → 354K params)
  - d=64 BC: 94.9% acc, 37.7% HvH (vs d=32: 79.1%, 33.8%) — better BC
  - d=64 + REINFORCE: 44-49% HvH (vs d=32: **54.4%**) — worse RL
  - Root cause: d=64 overfits BC → too peaked (entropy 0.23 vs 0.46) → REINFORCE overshoots
  - Tried: lower lr (1e-7), fewer BC epochs, label smoothing, all-episodes BC — none helped
  - **Conclusion: d=32 is optimal for current REINFORCE pipeline**
- [ ] **4.2** Self-play / league training
  - **Composition analysis**: 76 always-win, 65 always-lose, 3 mixed base scenarios
  - Results are composition-dependent, not tactical — ceiling for single-policy vs default AI
  - Requires asymmetric policies (hero team vs enemy team controlled by different models)
  - Modify `run_rl_episode` to support separate enemy policy weights
  - Train against pool of past checkpoints to avoid cycling
  - **This is the critical next step for >60% win rate**
- [x] **4.3** V3 pointer action space integration
  - Full pipeline working: BC + REINFORCE + engagement heuristic
  - 128d behavioral embeddings → `external_cls_proj` Linear(128→32)
  - BC on oracle data: 97.5% type acc, 98.3% pointer acc (but 0% move in data)
  - Engagement heuristic patches missing movement: move toward nearest enemy when holding out of range
  - **Result: ~30% HvH** (BC=28.9%, +REINFORCE=30.4% peak)
  - REINFORCE unstable: pg_loss grows exponentially after 3-4 iters
  - Root cause: pointer target selection is O(N) search space, REINFORCE too noisy
  - PPO broken: temperature mismatch between Rust (log_prob with temp) and Python (raw logits)
  - V2's 54.4% used flat actions where target selection was implicit — much easier optimization
  - **Conclusion: V3 pointer needs better training approach (self-play, curriculum, or imitation from V2)**

## Key Files
- Profiler: `src/bin/xtask/oracle_cmd/ability_profile.rs`
- Profiles data: `dataset/ability_profiles.npz` (135,792 samples, 943 abilities)
- Pretrain: `training/pretrain.py` (MLM + behavioral, `--freeze-encoder`)
- Export registry: `training/export_embedding_registry.py`
- Registry: `generated/ability_embedding_registry.json` (943 × 128d, model hash)
- Model: `training/model.py` (`BehavioralHead`, `AbilityActorCritic.external_cls_proj`)
- Train RL: `training/train_rl.py` (`--embedding-registry`)
- Export actor-critic: `training/export_actor_critic.py` (`--external-cls-dim`)
- Rust registry: `src/ai/core/ability_transformer/weights.rs` (`EmbeddingRegistry`)
- Episode gen/eval: `src/bin/xtask/oracle_cmd/transformer_rl.rs` (`--embedding-registry`)

## Log

### Run 1 (2026-03-11): Data generator built
- Built full profiler pipeline: CLI → ability loading → controlled sim → npz output
- 943 abilities, 135,792 samples, 119-dim outcome vectors

### Run 2 (2026-03-11): Behavioral encoder trained (lookup-based)
- Architecture: Embedding(943,32) + MLP predictor, 75K params
- val_mse=0.60, kNN@5 75.8% — but can't handle new abilities

### Run 3 (2026-03-11): Transformer CLS with behavioral finetuning
- Curriculum: MLM-only (97.5%) → frozen encoder + behavioral head
- Z-normed outcomes + Huber loss → stable 0.01 MSE (99% variance explained)
- Registry: 943 abilities × 128d CLS, kNN@5 99.4%

### Run 4 (2026-03-11): Policy integration — PPO iter 1
- Behavioral embeddings (128d) projected to d=32 via learned Linear layer
- Single PPO iteration on 2040 HvH bootstrap episodes
- **4x win rate improvement**: 12.7% → 50.0% on 204 HvH scenarios
- Next: iterative PPO to push higher, test on attrition scenarios

### Run 5 (2026-03-11): RL training exploration
- PPO: collapses every time (value head → garbage advantages → policy collapse)
- DAgger: overfits on wins-only (33.8% → 25%)
- **REINFORCE**: 1 epoch at lr=5e-7, unfrozen transformer → **54.4% HvH**
- On-policy iterations regress to ~48.5% (weak signal at 50% WR)
- Reward shaping (HP differential): 52-53% (slightly worse than no shaping)
- Attrition: 12% (HvH-specialized)
- Best model: `generated/actor_critic_weights_hvh_best.json`
