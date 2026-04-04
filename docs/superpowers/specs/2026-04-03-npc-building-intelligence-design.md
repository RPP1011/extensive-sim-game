# NPC Building Intelligence — V2

**Date:** 2026-04-03  
**Goal:** Train spatial construction AI for NPCs via oracle-bootstrapped BC → DAgger → RL pipeline. Each stage is gated on measurable criteria before advancing.

**Aspiration:** The best and most efficient spatial AI that exists. We will fall short, but it is worth trying for regardless.

---

## 1. Pipeline Overview

Four stages, each with explicit go/no-go gates. No stage is skipped.

```
Stage 1: Oracle BC        → baseline policy, ~70% oracle agreement
Stage 2: DAgger (3-5 rounds) → close distribution shift gap, ~85% oracle agreement on own rollouts
Stage 3: D-REX reward learning → learn reward that extrapolates beyond oracle ceiling
Stage 4: RL fine-tuning   → surpass oracle on learned reward, evaluated on held-out scenarios
```

**Why this ordering:** Pure BC compounds errors quadratically in episode length (Ross & Bagnell 2010). DAgger reduces this to linear by collecting oracle labels at the policy's own visited states. D-REX learns a reward function from ranked BC rollouts that can extrapolate beyond oracle quality (Brown et al. 2019). RL then optimizes against the learned reward without being bounded by oracle ceiling.

Each stage produces a checkpoint. If a stage degrades performance on the prior stage's metrics, roll back and diagnose before proceeding.

---

## 2. Memory Architecture — Importance-Filtered Ring Buffers

*Unchanged from V1. This is solid.*

### Tiers

| Tier | Slots | Importance Threshold | Decay | Contents |
|------|-------|---------------------|-------|----------|
| Short-term | 64 | None (all events) | Circular overwrite | Raw events: "wall segment (12,7) took 3 damage from ram," "NPC#42 couldn't path to market" |
| Medium-term | 256 | > 0.3 | Halves every 500 ticks | Aggregated patterns: "east wall breached 3 times in 40 ticks," "flooding in low-ground district" |
| Long-term | 64 | > 0.7 | None (permanent until contradicted) | Structural lessons: "east wall insufficient against jumping enemies," "stone survived fire that destroyed wood district" |

### Importance Scoring

```
importance = severity × recency_weight × novelty_bonus
```

- **Severity:** damage/deaths/cost normalized 0-1
- **Recency:** exponential decay from event tick
- **Novelty:** high if no similar event exists in medium-term buffer (prevents redundant promotion)

Promotion happens when short-term events cross the medium-term threshold (frequency or severity). Medium-term patterns promote to long-term when confirmed repeatedly or involve catastrophic outcomes.

---

## 3. Observation Space — Two-Track Design

The observation space is split into **raw spatial features** (primary policy inputs) and **oracle-derived features** (used only for reward shaping and curriculum gating, never as policy inputs). This prevents causal confusion where the policy learns to read oracle answers instead of learning spatial reasoning (de Haan et al. 2019).

### Track A: Raw Spatial Features (Policy Inputs)

Multi-channel grid representation processed by CNN encoder. Each channel is a 2D map over the settlement grid.

**Terrain channels:**
- Elevation (continuous, per cell)
- Terrain type (categorical: flat, slope, water, cliff, swamp, forest, rock)
- Flood risk (binary or continuous based on water table proximity)
- Wind direction (2 channels: x, y components)

**Structure channels:**
- Building presence (categorical: wall, residential, military, economic, religious, empty)
- Material (categorical: wood, stone, iron, mixed, none)
- Wall height (continuous, per cell)
- Wall thickness (continuous, per cell)
- Structural health (0-1 per cell)
- Roof type (categorical: flat, pitched, reinforced, none)
- Foundation depth (continuous, per cell)

**Connectivity channels:**
- Road network (binary per cell)
- Path distance to nearest gate (continuous, flood-filled)
- Path distance to nearest market (continuous, flood-filled)
- Connected component ID (integer per cell)

**Unit channels:**
- Friendly NPC density (continuous per cell)
- Friendly combat effectiveness density (weighted by NPC level/class)
- Enemy threat direction heatmap (continuous, computed from challenge injection)
- Garrison assignment map (which cells have assigned defenders)

**Population channels:**
- Occupancy density (continuous per cell)
- Housing capacity (continuous per cell)

**Scalar context (concatenated after CNN, before policy head):**
- Settlement level (1-5)
- Tech tier (1-4)
- Resource stockpiles: wood, stone, iron, food (each 0-1 normalized to settlement capacity)
- Worker counts by skill tag
- Population / housing capacity ratio
- Tick count / seasonal phase
- Challenge type embedding (learned, from challenge category)
- Memory buffer summary (fixed-size encoding of long-term lessons)

### Track B: Oracle-Derived Features (Reward Shaping Only)

These are computed by the same heuristics as the oracle, but are **never** fed to the policy network. They're used for:
- Reward shaping during RL (potential-based, preserves optimal policy)
- Curriculum gating (advance difficulty when these metrics cross thresholds)
- Evaluation dashboards

**Oracle-derived metrics:**
- Wall coverage % of perimeter
- Evacuation reachability %
- Chokepoint quality score
- Watchtower coverage %
- Fire propagation risk score
- Garrison effectiveness map
- Defensive value model output (structural + garrison + synergy)
- Building utilization ratios
- Commute efficiency
- Connectivity resilience (remove-one-structure test)

### Feature Ablation Protocol

Before full training, run ablation study on the raw feature set:

1. Train BC on full feature set → baseline
2. Remove each channel group independently → measure deployment performance change
3. Any channel group whose removal *improves* deployment performance is a causal confusion candidate → investigate or remove permanently
4. Document which features the policy actually uses (gradient-based attribution or permutation importance)

This catches the "more features = worse performance" trap documented across BC literature.

---

## 4. Challenge Categories

*Unchanged from V1. The 10-category matrix is comprehensive.*

### 4.1 Military/Defensive

- **Siege:** rams, catapults, sustained assault
- **Raid:** fast hit-and-run, wall-jumpers, climbers
- **Infiltration:** enemies bypassing walls via tunnels, stealth
- **Aerial threat:** flying enemies, projectile arcs over walls
- **Multi-vector:** simultaneous attack from multiple directions

### 4.2 Environmental/Natural Disaster

- **Flood:** river overflow, rain accumulation in low areas
- **Fire:** spreading between adjacent wood structures, lightning
- **Earthquake:** structural collapse, wall breaches, rubble blocking paths
- **Erosion/landslide:** terrain changes destroying foundations
- **Extreme weather:** blizzard blocking construction, heat cracking stone

### 4.3 Economic/Resource

- **Resource scarcity:** iron depleted, forest clear-cut, drought
- **Trade boom:** sudden need for market capacity, warehousing, road throughput
- **Resource discovery:** new mine opens — where to route access
- **Supply line disruption:** road destroyed, bridge lost
- **Inflation/cost pressure:** achieve same defense with fewer materials

### 4.4 Population/Social

- **Population surge:** refugee wave, baby boom, immigration
- **Population decline:** plague aftermath, emigration — consolidate, demolish
- **Class conflict:** workers need housing near industry, nobles want separation
- **Religious demand:** temple needed, sacred ground constraints
- **Faction split:** settlement divides — who gets the forge?

### 4.5 Temporal/Seasonal

- **Winter preparation:** insulation, food storage, fuel stockpiling before deadline
- **Harvest season:** temporary farm capacity, drying racks, storage surge
- **Festival/event:** temporary structures, crowd management, public space
- **Decay/maintenance:** buildings degrading, prioritize repair vs. new construction

### 4.6 Terrain/Geography

- **River crossing:** bridge placement, ford management
- **Hill/cliff settlement:** build up vs. terrace, defensive advantage vs. accessibility
- **Coastal:** dock placement, storm surge protection, sea wall
- **Chokepoint exploitation:** fortify natural bottleneck vs. route around
- **Hostile terrain expansion:** swamp, dense forest, rocky ground

### 4.7 Multi-Settlement/External

- **Trade route optimization:** road connecting two settlements
- **Allied defense coordination:** shared wall, mutual watchtower coverage
- **Territory dispute:** build to claim contested land
- **Outpost establishment:** minimal viable settlement in frontier
- **Road network planning:** connecting N settlements efficiently

### 4.8 Unit Capability Awareness

- **Friendly composition:** mages need towers, archers need elevation, melee need wall-adjacent barracks
- **Enemy capability profiles:** orcs have siege engines, bandits climb, dragons fly
- **Level disparity response:** underleveled defenders need more structural compensation
- **Class-specific infrastructure:** mage towers, archery platforms, cavalry stables, training yards sized to unit count
- **Garrison capacity planning:** barracks/watchtower sizing based on defender roster

### 4.9 High-Value NPC Awareness

- **Leader/noble protection:** manor placement, escape routes, safe rooms
- **Specialist housing priority:** master blacksmith near forge, healer near barracks
- **Visiting dignitaries:** temporary high-security accommodations
- **Threat-target awareness:** structurally protect assassination/kidnap targets
- **Succession planning:** courthouse accessible to next-in-line

### 4.10 Level-Scaled Construction

- **Settlement level tiers:** hamlet - village - town - city - capital, each with different building types and layout demands
- **Tech/skill progression:** wood-only through stone+iron, gated by settlement capability
- **Scaling defensive requirements:** fence at level 1, layered walls with murder holes at level 5
- **Upgrade vs. rebuild decisions:** patch wood wall with stone, or tear down and rebuild?
- **Workforce scaling:** scope projects to available labor (3 builders vs. 30)

---

## 5. Decision Types

*Unchanged from V1. The 5 strategic + 13 structural decision type taxonomy is good.*

### Tier 1: Strategic Decisions (Grid-Level)

1. Placement
2. Prioritization/Sequencing
3. Routing/Connectivity
4. Zone Composition
5. Demolition/Rebuild

### Tier 2: Structural Decisions (Tile-Level)

6-18 as in V1.

### Matrix: 10 × 18 = 180 cells

---

## 6. Reward Architecture — Chebyshev Scalarization with Per-Scenario Normalization

### Why Not Linear Scalarization

Linear scalarization (`r = Σ w_i R_i`) provably cannot recover Pareto-optimal solutions in non-convex regions of the objective trade-off surface. With 5+ objectives, the Pareto front almost certainly has significant non-convex regions. Interesting trade-off policies (sacrifice a little defense and a little economy for emergent cascading resilience) are exactly in those regions.

### Chebyshev Scalarization

```
r = -max_i(w_i × |R_i - R*_i(scenario)| / R*_i(scenario))
```

Negative because we minimize the worst-case weighted deviation from the ideal point.

**R\*_i(scenario)** = per-objective maximum, computed by running the oracle on each scenario with w_i = 1, all others = 0. Cached per scenario seed. Per-scenario normalization ensures a defense score of 0.8 means the same thing whether it's an easy flat-terrain scenario or a multi-vector siege.

**w_i** = objective weights, shifted by active challenge category:

| Challenge Category | Defense | Economy | Population | Connectivity | Garrison | Spatial |
|---|---|---|---|---|---|---|
| Military | 0.4 | 0.05 | 0.1 | 0.15 | 0.2 | 0.1 |
| Environmental | 0.15 | 0.1 | 0.15 | 0.2 | 0.05 | 0.35 |
| Economic | 0.05 | 0.4 | 0.1 | 0.25 | 0.05 | 0.15 |
| Population | 0.05 | 0.1 | 0.4 | 0.2 | 0.05 | 0.2 |
| Compound | 0.2 | 0.15 | 0.15 | 0.2 | 0.15 | 0.15 |

These are starting points. DyLam-style dynamic reweighting (see §6.2) adjusts them during RL training.

### 6.1 Objective Components

**Defensive scores (military challenges):**
- Breach count (0 = perfect), depth of deepest penetration
- Casualty ratio (friendly / enemy)
- High-value NPC survival (binary per important NPC, weighted by value)
- Response time (ticks between breach and first defender engagement)
- Resource efficiency (defensive outcome / material cost)

**Environmental scores:**
- Damage prevented vs. no-build counterfactual
- Recovery time to pre-disaster functionality
- Cascading failure prevention (did fire spread? did flood isolate districts?)

**Economic scores:**
- Throughput (trade volume, storage utilization)
- Commute efficiency (average NPC travel time to workplace)
- ROI (economic output per material invested)

**Population scores:**
- Housing coverage (unhoused after / before)
- Class accessibility (workers near industry, healer near wounded)
- Growth headroom

**Connectivity scores:**
- Connectivity resilience (remove any single structure — does settlement stay connected?)
- Redundant pathing between key building pairs
- Chokepoint quality (intentional at threat vectors, none on civilian routes)

**Garrison scores:**
- % of perimeter within response range of combat-capable NPCs
- Weighted by NPC effectiveness × structure synergy
- Patrol coverage efficiency

**Spatial quality scores:**
- Dead space % (context-scaled: tight hamlet = good, zero open space in city = bad)
- Expansion provision quality (stub walls, oversized foundations present for growth)

### 6.2 Dynamic Reward Weighting (RL Stage Only)

During RL fine-tuning, use DyLam-style automatic weight adjustment:

```
At each evaluation interval:
  for each objective i:
    if R_i is below its running baseline:
      increase w_i proportionally to the deficit
    if R_i is above its running baseline:
      decrease w_i (saturating at a floor)
  renormalize weights to sum to 1
```

This creates an implicit curriculum: the agent naturally focuses on its weakest objective, then shifts attention as that objective improves. No manual curriculum staging needed for the reward weights.

### 6.3 Potential-Based Reward Shaping (RL Stage Only)

Use Track B oracle-derived features as a potential function for reward shaping:

```
F(s, s') = γ × Φ(s') - Φ(s)
```

where Φ(s) is a composite of oracle-derived metrics (wall coverage, garrison effectiveness, connectivity resilience, etc.). This is the only form of reward shaping that is proven to preserve optimal policy invariance (Ng et al. 1999). The oracle metrics provide dense signal early in training without distorting what the agent ultimately optimizes for.

---

## 7. Oracle Heuristics

*Two-layer oracle preserved from V1, with modifications to address systematic blind spots.*

### Strategic Oracle (Grid-Level)

**Rule Layer (unchanged):**
- Threat from direction D + no wall coverage on D → place wall segment, priority = threat severity
- Housing pressure > 1.2 → place residential building, weighted away from industrial zone
- Population > 1.5× housing + leader exists → evaluate settlement founding
- Seasonal deadline approaching + insufficient storage → prioritize storage building
- Fire destroyed wood cluster → zone for stone rebuild
- High-value NPC unprotected + threat > 0.5 → defensive structure near them
- High-level combat NPCs present → bias guild/barracks placement toward weakest defensive segment

**Utility Layer — Chebyshev (modified):**

Replace linear weighted sum with:

```
score = -max_i(w_i × |predicted_R_i(action) - R*_i| / R*_i)
```

where `predicted_R_i(action)` is the oracle's estimate of objective i after taking the action. Same weights as §6, shifted by challenge category.

### Structural Oracle (Tile-Level)

**Rule Layer (unchanged from V1).**

**Utility Layer — Chebyshev (modified):** same transformation as strategic oracle.

### Oracle Consistency Enforcement

To prevent contradictory labels for equivalent states (which cause mode-averaging in BC):

1. **Canonical state representation:** before oracle evaluation, normalize grid orientation (rotate so primary threat is always from the north). This eliminates rotational duplicates that might get different labels due to iteration order.
2. **Deterministic tie-breaking:** when utility scores are within ε (0.01), always select the action with lowest grid-cell index. Arbitrary but consistent.
3. **Label confidence annotation:** tag each (observation, action) pair with the utility gap between the chosen action and the second-best. Low-gap pairs (< 0.05) are flagged as "ambiguous" — these get lower training weight in BC and are prime targets for DAgger round collection.

---

## 8. Data Generation Pipeline

### Stage 1: Oracle BC Dataset

**Step 1: Scenario Seed Generation** *(unchanged from V1)*

Randomize: settlement level (1-5), population (5-500), resource levels, terrain type, existing building stock, NPC roster, tech tier.

**Step 2: Challenge Injection** *(unchanged from V1)*

Single, compound, cascading, slow-burn challenges. Parameters randomized within plausible ranges.

**Step 3: Memory Buffer Population** *(unchanged from V1)*

**Step 4: Feature Computation**

Compute Track A features (raw spatial → policy input) and Track B features (oracle-derived → reward shaping only). Track B features are stored in metadata, not in the observation tensor.

**Step 5: Oracle Labeling — Modified**

1. Canonicalize state (rotate to threat-north).
2. Run strategic oracle → emit 1-N labeled actions with Chebyshev utility scores.
3. Run structural oracle for each strategic action → emit component-level specs.
4. For each action, compute confidence = utility gap to second-best.
5. Each (observation, action, confidence, utility_score) = one row.

**Step 6: Diversity Monitoring**

Track coverage of the 10×18 challenge×decision matrix. After each generation batch:

```
for each (challenge_category, decision_type) cell:
  count = pairs generated for this cell
  if count < min_threshold:
    increase sampling weight for this cell in next batch
```

Target: no cell has fewer than 100 pairs. Cells with < 50 are priority generation targets. Report coverage heatmap after each batch.

Additionally, monitor **state-space coverage** — cluster observations by PCA/UMAP embedding and flag under-represented regions. Generate targeted scenarios to fill gaps.

**Step 7: Quality-Weighted Dataset**

Each pair gets a training weight:

```
weight = confidence^α × utility_score^β × rarity_bonus
```

- `confidence^α`: high-confidence oracle decisions get more weight (α ≈ 0.5, not too aggressive — ambiguous decisions are still valuable signal)
- `utility_score^β`: higher-scoring actions contribute more
- `rarity_bonus`: pairs from under-represented matrix cells or state-space clusters get upweighted

### Volume Target

- 750 seeds × 3-5 challenges = 2,250-3,750 scenarios
- × 5-30 pairs each = ~11K-112K pairs
- Target minimum: 50K pairs before diversity-based pruning
- After DAgger (Stage 2): additional 20K-50K pairs from policy rollout states

---

## 9. Stage 2: DAgger

The oracle is a queryable heuristic, not a human expert. DAgger is therefore essentially free — no human labeling cost.

### DAgger Protocol

```
for round in 1..5:
  1. Roll out current BC policy on all scenario seeds → collect visited states S_π
  2. For each state in S_π:
     a. Query oracle for action label
     b. Record (state, oracle_action, confidence, utility_score)
  3. Aggregate new data into training set (β-weighted mixing with prior data)
  4. Retrain BC policy on aggregated dataset
  5. Evaluate:
     - Oracle agreement on policy-visited states (target: ≥85% by round 3-5)
     - Deployment performance on held-out scenarios
     - Distribution shift metric: KL(state distribution under policy || state distribution under oracle)
  6. Go/no-go:
     - If oracle agreement improved: continue to next round
     - If oracle agreement plateaued for 2 rounds: advance to Stage 3
     - If oracle agreement degraded: diagnose (likely dataset quality issue), fix, retry
```

### Key Implementation Details

- **β-schedule:** β starts at 1.0 (pure oracle policy) and decays to 0.0 (pure learned policy) over rounds. This gradually shifts state distribution from oracle to learner.
- **Noise injection (DART variant):** Add calibrated Gaussian noise to oracle actions during demonstration collection. This forces the dataset to include near-miss states and recovery behaviors. Calibrate noise level so oracle performance degrades by ~10% — this is the sweet spot where recovery signal is strong but demonstrations aren't garbage.
- **Parallel rollouts:** Run rollouts across multiple environment instances. Bottleneck is rollout time, not oracle query cost. Target: 1000+ rollouts per DAgger round.

---

## 10. Stage 3: D-REX Reward Learning

### Motivation

The BC policy is bounded by oracle quality. D-REX learns a reward function from ranked demonstrations that extrapolates beyond the oracle ceiling.

### Protocol

1. Take the DAgger-trained BC policy.
2. Create N ranked policy variants by injecting increasing noise:
   - π_0 = DAgger policy (best)
   - π_1 = DAgger policy + noise σ=0.1
   - π_2 = DAgger policy + noise σ=0.3
   - π_3 = DAgger policy + noise σ=0.6
   - π_4 = random policy (worst)
3. Roll out each variant on scenario seeds → collect trajectories.
4. Rank trajectories: π_0 > π_1 > π_2 > π_3 > π_4 (by construction).
5. Train a reward network to predict rankings from (state, action) pairs.
6. Validate: the learned reward should correlate with oracle outcome scores on held-out scenarios, but may assign high reward to states the oracle never reached.

### Reward Network Architecture

Input: same Track A features as policy network.  
Output: scalar reward per (state, action).  
Architecture: shared CNN encoder (can be initialized from BC encoder) → MLP head → scalar.

### Go/No-Go Gate

- Learned reward ranking accuracy on held-out trajectory pairs ≥ 90%
- Correlation between learned reward and oracle outcome scores ≥ 0.7 on held-out scenarios
- Spot-check: manually inspect top-10 highest-reward states the learned reward identifies that the oracle scored low → are these genuine improvements or reward hacking?

---

## 11. Stage 4: RL Fine-Tuning

### Reward

```
r_total = r_learned (from D-REX)
        + λ × F(s, s')  (potential-based shaping from oracle features, §6.3)
```

λ decays over training. Early on, oracle-derived shaping provides dense signal. Later, the learned reward dominates.

### Curriculum

Introduce challenge complexity gradually. This is separate from dynamic reward weighting (§6.2), which handles objective balancing.

**Level 1: Single challenges, small settlements**
- Settlement levels 1-2, population 5-50
- Single challenge category, severity 0.3-0.5
- 5-10 placement decisions per episode
- Gate: mean outcome score > 0.6 on Level 1 eval set

**Level 2: Single challenges, medium settlements**
- Settlement levels 2-3, population 50-150
- Single challenge, severity 0.5-0.8
- 10-20 placements per episode
- Gate: mean outcome score > 0.5 on Level 2 eval set

**Level 3: Compound challenges, medium settlements**
- Settlement levels 2-4, population 50-200
- 2-3 simultaneous challenge categories
- 15-25 placements
- Gate: mean outcome score > 0.4 on Level 3 eval set (lower threshold because compound is harder)

**Level 4: Full complexity**
- All settlement levels, all populations
- Compound + cascading challenges, full severity range
- No placement limit
- Gate: outperform oracle baseline on held-out scenario set

**Level transitions:** advance when the agent achieves the gate metric for 3 consecutive evaluation windows. If performance drops below 80% of the gate metric at any level, drop back one level and retrain.

### Action Masking

The policy outputs logits over all grid cells × building types. Invalid actions (occupied cell, insufficient resources, tech tier locked) are masked to -∞ before softmax. This is critical — without masking, the agent wastes capacity learning which actions are valid vs. which are good.

### Architecture

```
Input: Track A feature channels (H × W × C)
  ↓
CNN Encoder: IMPALA ResNet (3 blocks, channels 32→64→64)
  ↓
Spatial feature map (H × W × 64) ──────────────────────────────┐
  ↓                                                             │
Global pool + scalar context → MLP → 256-dim context vector     │
  ↓                                                             │
Tile via broadcast to (H × W × 256)                             │
  ↓                                                             │
Concat with spatial features → (H × W × 320)                   │
  ↓                                                             │
1×1 Conv → per-cell action logits (H × W × num_building_types)  │
  ↓                                                             │
Action mask → Softmax → Categorical sample                      │
                                                                │
Value head: spatial features (from ┘) → global pool → MLP → V(s)
```

**Why this architecture:**
- Fully convolutional preserves spatial information end-to-end (no flattening)
- IMPALA ResNet is proven for grid-based game AI (GridNet, StarCraft II)
- Per-cell logits = pointer-network-like output without attention overhead
- Scalar context (resources, settlement level, challenge embedding, memory encoding) is broadcast to all cells, giving each cell access to global state
- Shared encoder between policy and value reduces parameters

**Hierarchical decomposition (strategic → structural):**

The strategic tier (which building, which cell) and structural tier (wall thickness, material, footprint) are separate policy heads sharing the same encoder.

```
Strategic head: CNN encoder → per-cell building-type logits → sample (cell, building_type)
Structural head: CNN encoder + (cell, building_type) embedding → MLP → structural parameters
```

Strategic decisions are made first. The chosen (cell, building_type) is fed as conditioning to the structural head, which outputs continuous parameters (thickness, height, material probabilities, etc.) for that specific placement.

During BC, both heads are trained simultaneously. During RL, the strategic head is trained with policy gradient; the structural head can be trained with policy gradient or supervised from oracle structural decisions (hybrid approach — let RL discover *where* to build, let oracle expertise guide *how* to build).

---

## 12. Simulation Gym Hardening

### Adversarial Validity Testing

Before any training, run automated adversarial tests on the simulation:

1. **Placement validity exhaustion:** attempt every possible (cell, building_type) combination on 100 random settlement states. Verify that every "valid" placement produces a consistent post-state and every "invalid" placement is correctly rejected. Log any placement that passes validity but creates pathological outcomes (blocks all paths, overlapping influence zones, resources double-counted).

2. **Determinism check:** run the same scenario with the same actions 100 times. Assert bitwise-identical outcomes. If any non-determinism exists (random tie-breaking, float imprecision), fix or seed explicitly.

3. **Observation-action timing:** after each action, verify the returned observation reflects the post-action state, not the pre-action state. Off-by-one here is extremely common and silently degrades training.

4. **Reward consistency:** compute reward for the same (state, action, next_state) triple 1000 times. Assert identical results. Any stochasticity in reward computation is a training poison.

### Probe Environments

Build minimal test environments before training on the full game (Andy Jones methodology):

```
Probe 1: Single empty cell, one building type, reward = 1 if placed. (Tests: does the agent learn to take any action at all?)
Probe 2: Two cells, one good one bad, reward = +1/-1. (Tests: can the agent distinguish cells?)
Probe 3: Resource constraint — can only afford 1 of 2 buildings, one is better. (Tests: conditional decision-making)
Probe 4: Two-step sequence — wall then gate, reward only if both placed correctly. (Tests: credit assignment over horizon > 1)
Probe 5: Threat-response — enemy from north, must place wall on north edge. (Tests: spatial reasoning w.r.t. challenge)
```

Each probe should be solved to >95% optimality before proceeding to full environment training. If any probe fails, the bug is in the learning pipeline, not in the task complexity.

### Feature Computation Parity

Track A and Track B features must be computed by the **exact same code path** during training and deployment. No "training uses Python, deployment uses Rust" discrepancies. If the simulation is in Rust and the training pipeline uses Python bindings, verify numerical parity on 1000 random states with assertions on every feature channel.

---

## 13. Evaluation Protocol

### Held-Out Scenario Sets

Reserve 20% of scenario seeds as held-out evaluation. Never used for training data, DAgger collection, or curriculum tuning.

**Evaluation sets by difficulty:**
- Easy: single challenges, severity < 0.5, adequate resources
- Medium: single challenges, severity 0.5-0.8, mixed resources
- Hard: compound challenges, high severity, scarce resources
- Adversarial: hand-designed scenarios targeting known oracle weaknesses (compound cascading events, extreme resource constraints, unusual terrain)

### Baselines

1. **No-build baseline:** run challenge against settlement as-is. Measures what would have happened.
2. **Random valid policy:** N random valid building actions. Measures scenario difficulty.
3. **Oracle baseline:** apply oracle actions. This is the ceiling for BC, but RL should exceed it.
4. **BC-only baseline:** Stage 1 policy without DAgger. Measures DAgger's contribution.
5. **DAgger baseline:** Stage 2 policy without D-REX/RL. Measures RL's contribution.

### Metrics

Per-scenario:
- Composite outcome score (Chebyshev, §6)
- Per-objective breakdown (defense, economy, population, connectivity, garrison, spatial)
- Oracle agreement % (what fraction of actions match oracle recommendations)
- Novel-action rate (actions the oracle would not have taken — high rate + high outcome = genuine improvement)

Aggregate:
- Mean outcome score across held-out sets by difficulty
- Worst-case outcome score (min across held-out scenarios — tests robustness)
- Coverage of Pareto front (hypervolume indicator across objectives)
- Scenario-difficulty-adjusted score (outcome / random_baseline_outcome — how much harder-than-random scenarios does the agent handle?)

### Failure Mode Monitoring

Continuously monitor during RL training:

- **Reward hacking detector:** flag any episode where learned reward is high but oracle outcome score is low. Inspect manually. If rate exceeds 5%, pause training and investigate.
- **Objective collapse detector:** if any single objective drops below 50% of its R*_i while others are high, the agent is exploiting the reward structure. Trigger: increase that objective's Chebyshev weight by 2× and retrain.
- **Distribution shift monitor:** track KL divergence between training state distribution and policy rollout state distribution. Spikes indicate the policy is visiting unfamiliar territory — may need additional DAgger rounds.
- **Mode collapse detector:** track entropy of action distribution. If entropy drops near zero, the policy has collapsed to a single strategy — increase exploration or diversify training scenarios.

---

## 14. Scenario Template Data Format

### Observation (Input) — Modified

```json
{
  "challenge": {
    "category": "military/defensive",
    "sub_type": "raid_wall_jumpers",
    "severity": 0.8,
    "direction": [0.0, -1.0]
  },
  "memory": {
    "short_term": ["...64 raw events..."],
    "medium_term": ["...256 aggregated patterns..."],
    "long_term": ["...64 structural lessons..."]
  },
  "track_a_spatial": {
    "terrain_channels": {},
    "structure_channels": {},
    "connectivity_channels": {},
    "unit_channels": {},
    "population_channels": {}
  },
  "scalar_context": {
    "settlement_level": 3,
    "tech_tier": 2,
    "resources": { "wood": 0.6, "stone": 0.3, "iron": 0.1, "food": 0.8 },
    "workers": { "construction": 5, "masonry": 2, "labor": 10 },
    "housing_pressure": 1.3,
    "seasonal_phase": 0.7,
    "challenge_embedding": [],
    "memory_encoding": []
  },
  "track_b_oracle_features": {
    "_note": "NOT fed to policy. Used for reward shaping and evaluation only.",
    "wall_coverage": 0.72,
    "garrison_effectiveness": {},
    "connectivity_resilience": 0.85,
    "evacuation_reachability": 0.60,
    "fire_risk_score": 0.3
  },
  "units": {
    "friendly_roster": [{ "id": 1, "level": 5, "class": "archer", "combat_eff": 0.7 }],
    "enemy_profiles": [{ "type": "orc", "can_jump": true, "jump_height": 3, "has_siege": false }],
    "high_value_npcs": [{ "id": 12, "role": "leader", "level": 8, "location": [12, 8] }]
  },
  "decision_tier": "strategic"
}
```

### Action Label — Strategic Tier *(unchanged from V1)*

```json
{
  "decision_type": "placement",
  "action": {
    "building_type": "barracks",
    "grid_cell": [14, 3],
    "priority": 0.9,
    "reasoning_tag": "threat_proximity"
  },
  "metadata": {
    "confidence": 0.82,
    "utility_score": 0.91,
    "second_best_gap": 0.09,
    "training_weight": 0.87
  }
}
```

### Action Label — Structural Tier *(unchanged from V1, plus metadata)*

```json
{
  "decision_type": "wall_composition",
  "action": {
    "target": 42,
    "changes": [
      { "component": "north_wall", "thickness": 3, "material": "stone", "height": 5 },
      { "component": "crenellations", "add": true, "spacing": 2 }
    ],
    "reasoning_tag": "jump_counter"
  },
  "metadata": {
    "confidence": 0.95,
    "utility_score": 0.88,
    "second_best_gap": 0.22,
    "training_weight": 0.93
  }
}
```

---

## 15. Representative Scenario Templates

*Unchanged from V1. The examples are good and cover the matrix well.*

### Military x Structure

- Adventurer guild at weakest wall segment (garrison > stone when resource-constrained)
- U-shaped courtyard barracks creating kill zone
- Wall height matched to enemy jump height + 2
- Arrow slits facing threat vector, only if garrison includes ranged units
- Connected rooftop paths on perimeter buildings for defender repositioning
- Siege engine threat: buttressed stone walls. Climbers: smooth stone + overhang. Jumpers: height. Tunnelers: deep foundation.

### Environmental x Structure

- River-adjacent buildings on raised platforms, drainage channels routing water away
- Post-fire rebuild in stone, fire breaks (stone road/plaza) between wood clusters
- Earthquake zone: wide shallow foundations, square footprints, low profile
- Flood zone: narrow buildings perpendicular to flow, sacrificial wood ground floor + stone upper

### Economic x Strategic

- Iron scarce: wood buildings first, iron reserved for weapons. Compensate with garrison placement.
- Trade boom: market + warehouse before housing (revenue enables housing later)
- Partial material upgrades: stone-face only the threat-facing wall

### Population x Strategic

- Refugee wave: emergency tent district with clear upgrade path to permanent housing
- Noble/worker separation that maintains commute access to shared industry
- Housing near workplace to reduce commute, noisy industry buffered from residential

### Terrain x Structure

- Hillside: terraced buildings stepping down slope
- Cliff edge: setback + retaining wall
- Bridge as defensive chokepoint (fortify, don't just span)
- Natural bottleneck: fortify instead of building full perimeter wall

### Unit Capability x Placement

- Mage tower where spell range covers most of settlement
- Archer platforms at max effective range from engagement points
- Training yard near threat direction (pre-positioned defenders)
- Watchtower chain enabling patrol route force multiplication

### High-Value NPC x Structure

- Leader manor with escape route to settlement center
- Master blacksmith housed adjacent to forge (specialist productivity)
- Assassination target: reinforced door, interior chokepoint, no ground-floor windows facing exterior

### Level-Scaled x Renovation

- Wood palisade → stone wall → crenellated wall → wall + tower chain (upgrade progression)
- Tech tier gates material access: oracle never labels stone actions for wood-only settlements
- Scope projects to available labor: 3 builders get a fence, 30 builders get layered walls

---

## 16. Defensive Value Model — Garrison-Aware

Effective defense at a position is not purely structural. NPCs *are* defenses.

```
effective_defense = structural_defense + garrison_defense + synergy_bonus
```

- **structural_defense:** wall height, thickness, material, openings — the passive component
- **garrison_defense:** sum of combat effectiveness of NPCs likely to respond to that position (building assignments, patrol routes, proximity)
- **synergy_bonus:** structures that amplify unit effectiveness (archer on tower > archer on ground, chokepoint lets one fighter hold a corridor)

### Implications for Oracle Decisions

- **Resource-constrained settlements:** lean into garrison placement over structural investment. Adventurer guild at weakest wall segment is correct when stone is scarce but high-level fighters are present.
- **Diminishing structural returns:** `min(structural_defense, threshold) + garrison_defense × synergy_multiplier`. Past the threshold, more wall gives less value than positioning a strong NPC there.
- **Threshold scales with enemy type:** siege engines don't care about garrison — need structural answers. Raiders care a lot about garrison.
- **Placement decisions include garrison effect:** barracks near gate = fast response. Training yard toward threat = pre-positioned defenders. Adventurer guild at weak perimeter = garrison compensates for structure.
- **Patrol route awareness:** NPCs who patrol cover more ground than stationary ones. Watchtower chains with line-of-sight create force multiplication through patrol loops.

---

## 17. Risk Registry

| Risk | Severity | Likelihood | Mitigation | Detection |
|------|----------|------------|------------|-----------|
| Compounding BC error | High | Near-certain without DAgger | DAgger rounds 1-5 | Oracle agreement on policy-visited states |
| Causal confusion from oracle features | High | Likely if Track B leaks into policy | Two-track observation design, feature ablation | Feature ablation shows removal improves perf |
| Mode averaging on tied actions | Medium | Likely in ambiguous scenarios | Confidence-weighted training, temperature sampling | Action entropy monitoring |
| Reward hacking | High | Likely with composite reward | Chebyshev scalarization, manual spot-checks | Learned reward vs. oracle outcome divergence |
| Oracle ceiling limiting RL | Medium | Certain without D-REX | D-REX reward learning | RL performance vs. oracle baseline comparison |
| Simulation bugs exploited by RL | High | Likely in complex sim | Probe environments, adversarial testing | Anomalous action patterns, reward spikes |
| Non-convex Pareto blind spots | Medium | Certain with linear scalarization | Chebyshev scalarization | Hypervolume indicator tracking |
| Dataset imbalance across matrix | Medium | Likely without monitoring | Diversity monitoring, rarity upweighting | Matrix coverage heatmap |
| Objective collapse (one objective dominates) | Medium | Possible | DyLam dynamic weighting, collapse detector | Per-objective tracking during training |
| Feature computation train/deploy mismatch | High | Possible with multi-language stack | Parity assertions, shared code path | Numerical comparison on random states |

---

## 18. Implementation Sequence

### Phase 1: Foundation (weeks 1-3)
- Implement Track A feature computation in the simulation
- Implement Track B feature computation (oracle-derived)
- Build probe environments (5 probes)
- Implement oracle heuristics with Chebyshev scalarization
- Run adversarial validity testing on simulation
- Verify feature computation parity

### Phase 2: BC Dataset (weeks 3-5)
- Scenario seed generator
- Challenge injection system
- Memory buffer population
- Oracle labeling pipeline with consistency enforcement
- Diversity monitoring dashboard
- Generate 50K+ labeled pairs
- Feature ablation study

### Phase 3: BC + DAgger (weeks 5-8)
- Train BC baseline on dataset
- Evaluate on probe environments (must pass all 5)
- Evaluate on held-out scenarios
- Run 3-5 DAgger rounds
- Go/no-go: ≥85% oracle agreement on policy-visited states

### Phase 4: D-REX + RL (weeks 8-14)
- Generate ranked trajectory set from DAgger policy + noise variants
- Train D-REX reward network
- Go/no-go: ranking accuracy ≥90%, correlation with oracle outcomes ≥0.7
- RL training with curriculum levels 1-4
- Continuous failure mode monitoring
- Go/no-go: outperform oracle baseline on held-out scenarios

### Phase 5: Integration (weeks 14-16)
- Deploy trained model in game
- A/B test against oracle heuristics on player-facing scenarios
- Collect deployment metrics
- Iterate
