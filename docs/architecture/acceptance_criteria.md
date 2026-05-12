# Adversarial fixture acceptance criteria

Each fixture in `assets/sim/` paired with a pin test in `crates/sims/tests/`
needs explicit acceptance criteria. **A `soft NOTE` doesn't equal success.**
A behaviour blocked by a documented compiler gap is allowed to be soft-noted
ONLY IF:

1. The gap is identified by SHA + file + line in `docs/architecture/gaps_<fixture>.md`
2. The pin's assertion would be load-bearing IF the gap closed (it's commented
   out / behind `if` rather than absent — strengthen as soon as the gap closes)
3. The fixture's *observable contract* (kernels emit, ticks step without panic,
   topology stays valid) is asserted

If a fixture trips a gap and produces a static / NaN / zero result, we MUST
NOT call it passing. The gap goes in the doc; the pin reports the failure
as a documented limit; the criterion stays as the target.

## Per-fixture criteria

### plague_city
- Disease spread: ≥1 healthy → infected transition over 100 ticks  
- Recovery / death: ≥1 infected → recovered AND ≥1 infected → dead over 500 ticks  
- Cure dispatch: healers fire `Cure` ≥10 times; observable as a damage_taken-style view delta  
- Belief drift: per-detective's `believed_sick(self, target)` view diverges from ground truth (forgetting curve at work)  
- Conservation: `alive_count + dead_count == initial_population` at every checkpoint  
- Equilibrium: at tick 500, ≤30% still infected (the system reaches a bounded steady state)

### palace_coup
- King state: at least one `cast` block (RoyalDecree) completes without interrupt  
- Trust transitions: ≥3 distinct agents move through ≥2 trust states  
- LoS gating: at least one investigation event fires only when guard had LoS to suspect  
- Decisive event: either king alive at tick 300 (loyalists win) or king dead with ≥1 successful Assassinate event fired (conspirators win) — outcome must be ONE of those two; not "static, no kills"

### detective_investigation
- Evidence accumulates: `total_evidence_writes > 0` at tick 100  
- View storage stable: at tick 1000, view storage memory unchanged (no growth-related crash)  
- Accusations fired: ≥3 Accuse events over 1000 ticks  
- Accuracy meaningfully above chance: `(true_positives) / (total_accusations) > 0.4` (3-of-15 = 20% baseline; agents should beat random with belief accumulation)  
- Per-detective variance: at least one detective has ≥50% accuracy (some agents converge)

### pirate_fleet
- Cannons fire: ≥10 CannonFired events over 500 ticks  
- Boarding succeeds: ≥1 Boarded event fires (= ≥1 ship's `creature_type` flips)  
  - **If `agents.set_creature_type` is gated** (likely a documented gap), this becomes "Boarded events fire AND the consumer rule emits a write attempt" — the write itself can be soft-noted with the gap doc reference
- Treasure conservation: total gold across all ships at tick 500 == initial total ± rounding  
- Faction populations end-state: not 16/16 (no engagement); the engagement happened either via sinking, boarding, or both

### among_us  ⭐ user-prioritised
The user wants this one to ACTUALLY WORK and demonstrate emergent GOAP + ToM.
Acceptance is therefore stricter:

- **Kills happen**: Imposters fire ≥3 Kill events over 500 ticks  
- **Belief writes happen**: Crew with LoS to a kill set `belief_imposter(self, killer) = true` for the killer; verifiable as ≥3 belief-write events  
- **Vote events fire**: at tick 50, 100, 150, 200, 250 (vote phases); each phase tallies votes and ejects ≥1 suspect  
- **Belief accuracy beats chance**: of suspects voted out, `(true_imposter_ejections) / (total_ejections) > (3/20)` (15% baseline; with belief accumulation agents should hit ≥40%)  
- **Game terminates correctly**: either (a) all 3 Imposters voted out → Crew win, or (b) Imposters >= alive Crew → Imposter win. NOT "static, no votes, no resolution"  
- **GOAP-shaped behaviour**: Imposters target ISOLATED crew (no other Crew within witness radius), not the nearest crew. Verifiable via post-kill spatial analysis.

### caravan_wars (in-flight, will tag once .sim lands)
TBD per agent's chosen topology.

### forest_fire (in-flight)
TBD per agent's chosen topology.

### squad_skirmish (in-flight)
TBD per agent's chosen topology.

## Pin contract (universal)

Every pin must:
1. Initialize the GeneratedRuntime cleanly (try_new succeeds OR test gracefully skips with "no wgpu adapter")
2. Seed the topology deterministically (same seed → same outcome)
3. Run the declared tick budget without panic
4. Read back observable state via host-side staging buffers (no new methods on GeneratedRuntime)
5. Assert load-bearing criteria above; soft-NOTE only with gap doc reference
6. Report a one-line "verdict" summarising the outcome (CITY HOLDS / CREW WIN / etc.)

## Allowed soft-NOTE format

```
NOTE: <behaviour> didn't propagate — chronicle-consumer Indirect-dispatch gap
      from commit 353527e6 + gaps_<fixture>.md#chronicle-consumer-not-fired.
      When that gap closes, this NOTE becomes load-bearing assert at line N.
```

This format is the contract: gap is identified; future close path is named;
the criterion is preserved, not erased.
