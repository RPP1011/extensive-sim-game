# Mass Scenario Generation — Compositional World State Generator

**Date:** 2026-04-03  
**Goal:** Generate 50K+ labeled (observation, action) pairs covering the 10x18 challenge-decision matrix via compositional world state generation + oracle labeling.

**Parent spec:** `2026-04-03-npc-building-intelligence-design.md` (V2, Section 8)

---

## Architecture

Two independent layers composed at runtime:

- **Layer 1: World State Generators** — 5 independent axes producing diverse settlement starting states
- **Layer 2: Pressure Injectors** — 24 event types that force specific oracle decisions

Variety comes from combinatorial explosion (8 × 5 × 5 × 6 × 5 = 6,000 seed configs × 24+ pressures), not parameter sweeps within fixed templates.

The oracle is observation-driven — it reads spatial features, memory, unit rosters, and responds to what it sees. The generator's job is to create genuinely different world states that force different decisions.

---

## Layer 1: World State Axes

### Terrain (8 types)

| Type | Buildable % | Key properties | Forces decisions |
|---|---|---|---|
| Flat open | ~90% | No constraints, baseline | Pure placement optimization |
| River bisect | ~70% | Needs bridges, flood risk on banks | Routing, foundation, drainage |
| Hillside | ~75% | Elevation gradient, terracing | Vertical design, foundation, terrain adaptation |
| Cliff edge | ~60% | One side impassable, natural defense | Placement (leverage natural wall), expansion |
| Coastal | ~65% | Dock access, storm surge risk | Environmental adaptation, routing to port |
| Swamp | ~50% | Difficult ground, limited foundations | Foundation (pilings), material selection, routing |
| Forest clearing | ~55% | Wood abundant, fire risk, clearing cost | Material selection, environmental adaptation |
| Mountain pass | ~40% | Natural chokepoint, limited space | Chokepoint fortification, vertical design |

Each terrain generator produces: elevation map, water cells, buildable mask, resource modifiers (forest → wood +3, mountain → stone +2).

### Settlement Maturity (5 levels)

| Level | Existing buildings | Existing walls | Layout quality | Primary pressure |
|---|---|---|---|---|
| Empty (1) | 0-2 | None | N/A | Greenfield placement |
| Sparse (1-2) | 3-8 | None or partial fence | Random | Placement + routing from scratch |
| Moderate (2-3) | 8-20 | Partial walls, 1-2 gates | Mixed | Infill, upgrade, extend defenses |
| Dense (3-4) | 20-40 | Full perimeter, towers | Mostly organized | Renovation, optimization, garrison |
| Overgrown (4-5) | 40+ | Full + inner walls | Organic sprawl | Demolish/rebuild, routing fixes |

### Resource Profiles (5 types)

| Profile | Wood | Stone | Iron | Food | Oracle bias |
|---|---|---|---|---|---|
| Abundant | High | High | High | High | Ideal structural choices |
| Mixed | High | Low | Mid | High | Material substitution, hybrid walls |
| Scarce | Low | Low | Low | Mid | Garrison-heavy, minimal structures |
| Specialized | Very high | None | None | Mid | All-wood with fire mitigation |
| Depleting | Mid→Low | Mid→Low | Mid→Low | Mid | Temporal urgency, prioritization |

### NPC Roster (6 compositions)

| Composition | Combat | Workers | HV NPCs | Oracle bias |
|---|---|---|---|---|
| Military-heavy | 60% | 20% | 1-2 commanders | Garrison-rich, slow construction |
| Civilian-heavy | 10% | 70% | 1 leader | Fast construction, weak garrison |
| Balanced | 30% | 40% | 1-2 mixed | Standard tradeoffs |
| Elite few | 3-5 high-level | 5-10 | 1 champion | Garrison compensation, targeted placement |
| Large low-level | 20% low | 60% low | 0-1 | Quantity-based garrison, simple structures |
| Specialist | 20% | 40% | 3-4 specialists | HV NPC protection, specialist housing |

### Existing Building Quality (5 types)

| Quality | Connectivity | Zoning | Wall condition | Decisions forced |
|---|---|---|---|---|
| Well-planned | All connected | Proper zones | Good | Optimization, expansion |
| Organic growth | Gaps, dead ends | Mixed use | Patchy | Routing fixes, zone composition |
| Battle-damaged | Broken paths | N/A | Breached | Repair vs rebuild, emergency defense |
| Under construction | Partial | Planned but incomplete | Incomplete | Prioritization, sequencing |
| Abandoned/decayed | Degraded | Former zones | Crumbling | Demolish/rebuild, renovation |

**Combinatorial yield:** 8 × 5 × 5 × 6 × 5 = 6,000 distinct seed configurations.

---

## Layer 2: Pressure Types (24)

### Military (8)

| # | Pressure | Key params | Decision pressure |
|---|---|---|---|
| 1 | Infantry raid | count, level, direction | Wall placement, garrison positioning |
| 2 | Siege assault | siege_damage, duration, direction | Wall thickness, buttressing, material |
| 3 | Wall jumpers | jump_height, count | Wall height >= H+2, or garrison compensation |
| 4 | Climbers | climb_speed, count | Smooth walls, overhangs, wall material |
| 5 | Tunnelers | tunnel_speed, count | Deep foundations, underground detection |
| 6 | Flyers | fly_speed, count | Rooftop defense, safe rooms, anti-air placement |
| 7 | Multi-vector | 2-3 directions, mixed types | Perimeter coverage, garrison distribution |
| 8 | Infiltrators | stealth, target_hv_npc | HV NPC protection, interior chokepoints |

### Environmental (5)

| # | Pressure | Key params | Decision pressure |
|---|---|---|---|
| 9 | Flood | severity, duration, affected_cells | Foundation height, drainage, routing |
| 10 | Fire outbreak | origin, spread_rate, wind | Material selection, fire breaks, rebuild zoning |
| 11 | Earthquake | magnitude, aftershocks | Foundation width, low profile, wide footprint |
| 12 | Landslide | direction, severity | Setback, retaining walls, relocation |
| 13 | Storm/blizzard | wind_speed, duration, cold | Roof reinforcement, insulation, sheltering |

### Economic (4)

| # | Pressure | Key params | Decision pressure |
|---|---|---|---|
| 14 | Resource depletion | which_resource, timeline | Material substitution, prioritization |
| 15 | Trade boom | demand_multiplier, duration | Market/warehouse placement, road throughput |
| 16 | Supply disruption | which_route, severity | Rerouting, bridge/road repair, alternate paths |
| 17 | Resource discovery | type, location, yield | Access routing, processing building placement |

### Population (4)

| # | Pressure | Key params | Decision pressure |
|---|---|---|---|
| 18 | Refugee wave | count, urgency | Housing placement, density, expansion |
| 19 | Population decline | severity, speed | Consolidation, demolish, repurpose |
| 20 | Class tension | faction_split, intensity | Zone composition, separation + access balance |
| 21 | Specialist arrival | role, level, needs | HV NPC housing, specialist building proximity |

### Temporal (3)

| # | Pressure | Key params | Decision pressure |
|---|---|---|---|
| 22 | Winter deadline | ticks_remaining, severity | Storage, insulation, prioritization/sequencing |
| 23 | Harvest surge | crop_volume, storage_need | Temporary structures, warehouse sizing |
| 24 | Building decay | rate, affected_buildings | Repair vs replace, renovation, material upgrade |

### Compound Pressures

Generated by sampling 2-3 compatible singles. Compatibility:
- Military + Environmental = always compatible
- Military + Economic = always compatible
- Environmental + Population = always compatible
- Military + Military = compatible if different directions
- Environmental + Environmental = compatible if different types
- Temporal + anything = always compatible

---

## Composition Engine

```rust
fn generate_scenario(
    axes: &SamplingWeights,
    rng: &mut StdRng,
) -> (WorldState, Vec<Pressure>, BuildingObservation)
```

Procedure:
1. Sample terrain type (weighted by coverage needs)
2. Sample maturity level (weighted)
3. Sample resource profile (weighted)
4. Sample NPC roster composition (weighted)
5. Sample building quality (weighted)
6. Generate WorldState from 5-axis composition
7. Sample 1-3 pressures (weighted by coverage + compatibility filter)
8. Inject pressures into WorldState
9. Populate construction memory from pressure effects
10. Compute spatial features, build observation
11. Run oracle, collect actions
12. Map actions to matrix cells, update coverage

Compatibility filter rejects and resamples incompatible combos (flood needs water, coastal needs coast, demolish needs existing buildings, etc.).

---

## Coverage Tracking

```rust
struct CoverageTracker {
    matrix: [[u32; 18]; 10],
    target_per_cell: u32,       // 100
    dead_cells: HashSet<(usize, usize)>,
    total_pairs: u64,
}
```

After each scenario, map oracle actions to (category, decision_type) cells. Recompute sampling weights every 100 scenarios, biasing toward under-represented cells.

Dead cells (inherently inactive combos like "multi-settlement × arrow slits") identified empirically — 0 hits after 500 scenarios.

---

## Output Format

JSONL, one line per (observation, action) pair:

```jsonl
{"obs": {...}, "action": {...}, "meta": {"confidence": 0.82, "utility_score": 0.91, "category": "military", "decision_type": "placement", "scenario_id": 1234, "pressures": ["siege", "resource_depletion"]}}
```

Coverage report JSON:

```json
{
  "total_scenarios": 2500,
  "total_pairs": 52340,
  "matrix": [[102, 87, ...], ...],
  "dead_cells": [[6, 11], [6, 12], ...],
  "min_active_cell": 87
}
```

---

## CLI Interface

```bash
cargo run --bin xtask -- building-ai generate \
    --pairs 50000 --min-cell 100 \
    --output generated/building_bc.jsonl \
    --coverage generated/building_coverage.json \
    --seed 42

cargo run --bin xtask -- building-ai coverage generated/building_bc.jsonl

cargo run --bin xtask -- building-ai fill-gaps \
    --dataset generated/building_bc.jsonl \
    --min-cell 100 \
    --output generated/building_bc_supplement.jsonl
```
