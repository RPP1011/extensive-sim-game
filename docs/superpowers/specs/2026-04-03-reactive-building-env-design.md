# Reactive Building Environment — Design Spec

**Date:** 2026-04-03
**Parent:** NPC Building Intelligence V2
**Goal:** Turn the world sim into an rl4burn `Env` so the building AI makes decisions mid-simulation in response to world events, producing RL-ready (state, action, reward, next_state) trajectories.

---

## 1. Problem

The building AI oracle currently runs once at scenario init. The sim then runs thousands of ticks with zero building decisions. This means:

- No reactive building in response to events (attacks, deaths, resource crises)
- No training signal for when to build vs. when to wait
- No outcome measurement for whether a decision actually helped
- BC dataset pairs are static snapshots, not situated decisions

The env must expose decision points during simulation so an agent (oracle, BC policy, or RL policy) can build reactively.

---

## 2. Approach

**Thin Env Wrapper (Approach A):** `BuildingEnv` owns a `WorldSim` and implements `rl4burn_core::env::Env`. The sim ticks forward internally between `step()` calls. The agent only sees decision points, never individual sim ticks.

The env interface is stage-agnostic: BC, DAgger, D-REX, and RL all use the same env. What changes between stages is who picks the action and how reward is computed.

---

## 3. BuildingEnv Struct

```rust
pub struct BuildingEnv {
    sim: WorldSim,
    challenge: Challenge,
    tick_budget: u64,
    heartbeat_interval: u64,
    ticks_since_decision: u64,
    actions_taken: usize,
    max_actions: usize,
    pre_decision_scores: ObjectiveScores,
    scenario_axes: ScenarioAxes,
    seed: u64,
    curriculum_level: u8,
}

impl Env for BuildingEnv {
    type Observation = Vec<f32>;
    type Action = usize;
    // ...
}
```

---

## 4. Env Trait Implementation

### `reset() -> Vec<f32>`

1. Sample scenario axes weighted by curriculum level
2. `compose_world_state()` from mass_gen axes
3. Inject challenge (category + severity from curriculum)
4. `WorldSim::new(state)` with `skip_resource_init = true`
5. Tick sim until first trigger fires or heartbeat interval reached
6. Snapshot pre-decision objective scores
7. Return Track A observation flattened to `Vec<f32>`

### `step(action: usize) -> Step<Vec<f32>>`

1. **Decode action:**
   - `0` = pass/no-op (done building for this trigger)
   - `1..=N` = strategic placement: `cell_index = (action - 1) / num_building_types`, `building_type = (action - 1) % num_building_types`
2. **Apply action** (if not pass):
   - Create `BuildingAction` from decoded (cell, type)
   - Auto-fill structural details via structural oracle
   - `apply_actions(&mut state, &actions)`
   - Increment `actions_taken`
3. **Advance sim** — tick until one of:
   - An event trigger fires (see §5)
   - `ticks_since_decision >= heartbeat_interval`
   - Challenge resolves → `terminated = true`
   - `sim.state().tick >= tick_budget` → `truncated = true`
   - `actions_taken >= max_actions` → `truncated = true`
4. **Compute reward:**
   - Snapshot post-decision objective scores
   - Chebyshev scalarization delta (§7)
   - Update `pre_decision_scores` for next step
5. **Return** `Step { observation, reward, terminated, truncated }`

### `action_space() -> Space`

```rust
Space::Discrete(grid_cells * num_building_types + 1)
```

For a 128x128 grid with ~10 building types: `Discrete(163841)`. Large but masked PPO handles this — most actions are masked invalid.

### `action_mask() -> Option<Vec<f32>>`

Returns `Vec<f32>` of length `action_space.flat_dim()`:
- Index 0 (pass): always `1.0`
- Index `1 + cell * B + type`: `1.0` if cell is unoccupied AND resources sufficient for building type, `0.0` otherwise

### `observation_space() -> Space`

```rust
Space::Box {
    low: vec![0.0; obs_dim],
    high: vec![1.0; obs_dim],
}
```

Observation dimension depends on Track A channel count and grid resolution. All features normalized to [0, 1].

---

## 5. Decision Point Triggers

Checked each sim tick inside `step()`'s inner tick loop.

### Event Triggers (immediate decision point)

| Event | Detection | Rationale |
|-------|-----------|-----------|
| NPC death | `WorldEvent::EntityDied` for NPC entity | Settlement lost a defender or worker |
| Monster enters settlement grid | `WorldEvent::BattleStarted` or grid fidelity escalated to High | Active threat requires defensive response |
| Building health < 50% | Entity HP check on building entities | Infrastructure failing, may need replacement |
| Resource crisis | Any stockpile commodity drops below 20% capacity | Economic pressure requires production buildings |
| Housing overflow | Population exceeds housing capacity by > 20% | Growth pressure requires residential construction |

### Heartbeat Trigger

`ticks_since_decision >= heartbeat_interval` (default: 200 ticks). Fires even when no events occurred. Allows the agent to proactively build during calm periods. The agent can pass (no-op) if nothing needs doing.

### Priority

If multiple event triggers fire on the same tick, the decision point fires once. The observation captures all current pressures — the agent decides which (if any) to address.

---

## 6. Episode Structure

### Boundaries

- **Start:** `reset()` composes a scenario with an injected challenge
- **Terminated:** Challenge resolves naturally:
  - Military: all attack-wave monsters dead
  - Environmental: disaster passed (flood receded, fire out)
  - Economic: resource crisis stabilized (stockpiles above critical threshold for 100+ ticks)
  - Population: housing pressure below 1.0 for 100+ ticks
- **Truncated:** Tick budget exhausted or max actions reached

### Within an Episode

Multiple decision points per episode. At each, the agent takes one action (place or pass). Pass with no pending triggers advances the sim by the heartbeat interval. The agent can take multiple actions in rapid succession if triggers keep firing (e.g., cascading monster attacks).

### Multi-Step Response to Single Trigger

When a trigger fires, the agent gets one `step()` call. If it places a building, the sim ticks once (to apply the placement's effects), then re-checks triggers before continuing the tick loop. If a new trigger fires from the placement's effects, the loop breaks immediately and returns to the agent. If no new trigger fires, the sim continues ticking toward the next trigger or heartbeat. This naturally allows multi-action responses without special-casing.

---

## 7. Reward

### Chebyshev Scalarization (V2 §6)

Per-step reward computed as delta between pre-decision and post-decision objective scores:

```
r = -max_i(w_i * |R_i(post) - R*_i| / R*_i) - (-max_i(w_i * |R_i(pre) - R*_i| / R*_i))
```

Positive reward when the worst-case objective improved. Negative when it degraded.

**R\*_i(scenario):** Per-objective ideal, cached per scenario seed by running the oracle with single-objective focus.

**w_i:** Challenge-category-dependent weights per V2 §6 table.

### Objective Components

Per V2 §6.1:
- Defensive: breach count, casualty ratio, NPC survival, response time
- Environmental: damage prevented, recovery time, cascading failure prevention
- Economic: throughput, commute efficiency, ROI
- Population: housing coverage, class accessibility, growth headroom
- Connectivity: resilience, redundant pathing, chokepoint quality
- Garrison: perimeter coverage, patrol efficiency
- Spatial: dead space, expansion provision

### Potential-Based Shaping (future, RL stage)

Track B oracle-derived features as potential function: `F(s, s') = gamma * Phi(s') - Phi(s)`. Dense early signal, decayed over training. Not needed for BC/DAgger stages.

---

## 8. Curriculum

Curriculum controls `reset()` parameters. All curriculum logic is external to the env.

| Level | Settlement Size | Challenges | Severity | Max Actions | Tick Budget | Gate |
|-------|----------------|-----------|----------|-------------|-------------|------|
| 1 | Small (pop 5-50) | Single | 0.3-0.5 | 10 | 2000 | Mean outcome > 0.6 |
| 2 | Medium (pop 50-150) | Single | 0.5-0.8 | 20 | 5000 | Mean outcome > 0.5 |
| 3 | Medium (pop 50-200) | 2-3 compound | Mixed | 25 | 8000 | Mean outcome > 0.4 |
| 4 | All sizes | Compound + cascading | Full range | Unlimited | 15000 | Beat oracle baseline |

Advance when gate metric achieved for 3 consecutive eval windows. Drop back if performance falls below 80% of gate metric.

---

## 9. Integration with rl4burn

### SyncVecEnv

```rust
let envs: Vec<BuildingEnv> = (0..num_envs)
    .map(|i| BuildingEnv::new(config, seed + i as u64))
    .collect();
let mut vec_env = SyncVecEnv::new(envs);
```

Auto-reset on episode completion. Each env runs independent scenarios.

### Masked PPO

Use `rl4burn_algo::base::ppo_masked` with `action_mask()` from the env. Invalid placements are masked to -inf before softmax. The pass action is always available.

### Model Architecture

Model-agnostic from the env's perspective. The env outputs `Vec<f32>` observations and consumes `usize` actions. The model (CNN, spatial attention, transformer, MLP) reshapes the flat observation vector as needed in its forward pass.

The V2 doc (§11) describes an IMPALA ResNet with per-cell logits. Multi-resolution spatial attention over gridmaps is also compatible — the env doesn't constrain architecture.

---

## 10. Structural Auto-Fill

When the agent places a building (strategic decision), structural details are auto-filled by the structural oracle:

1. Agent chooses `(cell, building_type)` via `step()`
2. Env creates a strategic `BuildingAction`
3. Env runs `structural_oracle()` conditioned on the placement context
4. Combined strategic + structural actions applied via `apply_actions()`

This keeps the RL action space to a single discrete choice per step while preserving structural detail. The structural oracle can later be replaced by a learned structural head (V2 §11 hierarchical decomposition) without changing the env interface.

---

## 11. Future Work (Not In Scope)

These are defined by the V2 doc but not part of the initial env implementation:

- **BC data collection pipeline** — oracle acts in the env, records trajectories to JSONL
- **DAgger collection** — policy rollouts with oracle labeling at visited states
- **D-REX reward learning** — ranked trajectory collection from noised policies
- **RL fine-tuning** — masked PPO against learned reward
- **Probe environments** — minimal test envs for pipeline validation (V2 §12)
- **Track B oracle-derived features** — reward shaping potential function
- **Dynamic reward weighting** — DyLam-style auto-adjustment during RL (V2 §6.2)
- **Adversarial validity testing** — exhaustive placement validity checks (V2 §12)
- **Feature ablation** — channel importance analysis (V2 §3)
