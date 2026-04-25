# DSL compiler improvement opportunities — emitter patterns + IR optimizations

**Status:** research (deliverable: ranked roadmap, no code changes)
**Date:** 2026-04-24
**Branch:** `dsl-compiler-improvements-research` (off `world-sim-bench`)
**Author:** synthesis of tonight's emitter-pattern wins
**Predecessors:** Five 2026-04-24 measurement-crawl research docs (mask-kernel-subphase, scoring-kernel-row-decomposition, scoring-view-read-decomposition, cascade-rule-decomposition, gpu-per-dispatch-attribution). Their instrumentation has been reverted; numbers and conclusions are referenced inline below where load-bearing. Recover detail via `git log --diff-filter=D --all -- 'docs/superpowers/research/2026-04-24-*.md'`.

---

## Why this doc exists

Tonight's measurement crawl found a pattern that keeps repeating: large
wins come from changing **how a single DSL primitive lowers**, applied
automatically across every emitter kind. The headline:

> The alive bitmap (commit `1634e55a` / merge `d5d8211a`) lowered
> `agents.alive(x)` from a 64-byte `AgentSlot` cacheline read to a
> 4-byte L1-resident bitmap lookup. Saved **155 ms/tick wall-clock at
> N=100k — a 49% reduction**, just by re-targeting one primitive's
> lowering arm in `emit_physics_wgsl`, `emit_mask_wgsl`, and
> `emit_scoring_wgsl` simultaneously.

Other tonight wins followed the same shape — type-aware scalar literals
+ fold-body RHS lowering for `symmetric_pair_topk` (`5fbdb71a`) was a
narrow IR change that unblocked the entire `standing` view's i32 path
across CPU + GPU. Exhaustive walker (`4c47dc5f`) is a future-bug
prevention layer that will compound across every new emitter feature.

A reviewer's meta-observation crystallised what we kept finding:

> Most of these optimizations are really emitter-pattern changes rather
> than one-off WGSL tweaks. Worth thinking about which belong in the
> emitter's toolkit vs. which are point fixes, since the DSL presumably
> generates lots of similar kernels.

This doc enumerates opportunities for **compiler-level work** — things
that, if landed in the emitter, would compound across every `.sim`
declaration that fits the pattern. Each section ranks impact + cost + risk.

A top-10 across-area ranking sits at the bottom for the next quarter's
planning.

---

## Surface area inventory (context for impact estimates)

To calibrate the ranking, here's the codebase the emitter currently
produces from. Numbers are 2026-04-24 snapshots:

* **Compiler (LoC):** `crates/dsl_compiler/src/` — 30,311 lines across
  23 files. The big ones are `resolve.rs` (4,154), `emit_view.rs`
  (3,476), `parser.rs` (2,737), `emit_physics_wgsl.rs` (2,683),
  `emit_view_wgsl.rs` (2,234), `emit_scoring_wgsl.rs` (2,134),
  `emit_physics.rs` (1,812), `emit_mask_wgsl.rs` (1,836),
  `emit_scoring.rs` (1,490), `emit_mask.rs` (1,379), `ir.rs` (1,139),
  `schema_hash.rs` (1,088).
* **DSL surface (LoC):** `assets/sim/` — 1,719 lines across 8 files
  (config 80, entities 118, enums 18, events 163, masks 104,
  physics 611, scoring 297, views 328).
* **Emitted code (rough counts):**
  - 23 physics rules (`crates/engine/src/generated/physics/`)
  - 9 mask predicates (`.../generated/mask/`)
  - 12 view modules (`.../generated/views/`) — 5 lazy + 7 materialized
  - 7 entity / 4 enum / 4 config / many event modules
  - 3 GPU shader emitters: physics WGSL, mask WGSL, scoring WGSL,
    view WGSL — each currently dispatched as one fused kernel.
* **Hot loops (per tonight's measurement):** at N=100k, 50 ticks, RTX 4090:
  - **scoring kernel:** 16-22 ms/tick (post-alive-bitmap), dominated
    by MoveToward + Attack target-loops (~50/50 each, see
    the scoring-kernel row-decomposition research (archived)).
  - **scoring view reads:** 29.3M reads/tick across 100k agents;
    `threat_level` is 42% of all view reads.
  - **mask kernel:** 1.5 ms/tick (small relative to scoring).
  - **cascade phase:** see per-dispatch attribution doc; chronicle ring
    on dedicated path saved further wall-clock.

---

## Area 1 — Predicate-as-bitmap lowerings

### What it would do

Generalise the alive-bitmap pattern: any per-agent boolean predicate
that's read inside a tight inner loop gets a per-tick packed bitmap
(N/32 u32s, L1-resident) prepared by a small derive kernel, and the
emitter rewrites the predicate to a `bit(slot)` lookup.

The shape: derive once per tick (1 thread per slot), read O(N×L) times
in the consuming loops where L is loop-body density (every scoring
candidate walk, every cascade dispatch).

### Concrete instances

In rough rank-by-density order:

1. **`is_stunned(x)` — `views.sim:36-39`.** Currently lowers to two SoA
   reads (`world.tick` from sim_cfg + `agents.stun_expires_at_tick(a)`
   from agent_data) and a comparison. Per-tick bitmap derived by
   comparing `stun_expires_at_tick > tick` once. Used in `mask Cast`
   (`masks.sim:101`) and reachable via the cooldown / cast paths in
   physics. **Frequency:** lower than `alive` (only Cast / castable
   abilities) but the savings per call equal alive's because the read
   pattern is identical.

2. **`hp_pct < THRESHOLD` (constant T) — `scoring.sim:113, 231-233,
   260`** and physics chronicle/rally rules. Five distinct constant
   thresholds in scoring (0.3, 0.5, 0.8) and physics (0.5 in
   `chronicle_wound`, `rally_on_wound`). A single per-tick "below
   half" bitmap (the most common) would fold the
   `chronicle_wound` + `rally_on_wound` cascade fan-out (both fire
   on `AgentAttacked` and read the same `hp_pct < 0.5` test —
   `crates/engine/src/generated/physics/{chronicle_wound,
   rally_on_wound}.rs`). More speculatively, a per-threshold bitmap
   table would cover scoring's three thresholds.

3. **`engaged_with(x) != None` — implicit in scoring's Attack
   row + cast mask (`masks.sim:104`).** The view storage is already a
   per-entity slot map; deriving an "is engaged" bitmap from that is
   one bit per agent. Hot in scoring's Attack target loop.

4. **`agent.creature_type == K` (per-species bitmap).** The
   `is_hostile` check (`assets/sim/views.sim:23-26`) is a pairwise
   creature-type table, but `query.nearby_kin` (used by 3 physics
   rules: `fear_spread_on_death`, `pack_focus_on_engagement`,
   `rally_on_wound`) does a creature-type filter inline. With 4
   species, four bitmaps × N/32 × 4 B = ~12 KB at N=100k — still L1.
   Lets `nearby_kin` skip the cold creature_type cacheline read per
   neighbour.

5. **`agent_alive(x) && hp_pct(x) < T`** combined as a single
   "wounded-and-alive" bitmap. Used in `chronicle_wound` and
   `rally_on_wound` together. Pre-AND-ing on derive is free.

### Estimated impact

* `is_stunned` bitmap: **small** (~0.5 ms/tick at N=100k — Cast usage
  is rare in current scenarios, more impact in future scenarios with
  many casters).
* `hp_pct < T` bitmaps: **medium-large**. Scoring reads `hp_pct` ~3×
  per agent per tick across the 3 threshold tests, target-bound.
  At N=100k that's ~300k reads × ~50 candidate-loop multiplier =
  ~15M reads → potentially ~10-15 ms/tick if the load pattern
  matches alive's miss profile. **Uncertain — the read currently goes
  through `t.hp_pct` which is in the same 64 B cacheline as `alive`,
  so it's already paid once per cacheline pull.** Real savings only
  materialise where `hp_pct` is read without other agent_data fields
  in the same hot path. Needs measurement.
* `engaged_with != None`: **small-medium** (~1-2 ms/tick) — the view
  storage read is an atomic load on GPU and a hashmap lookup on CPU;
  bitmap is one cycle.
* `creature_type == Wolf` etc: **small** (~0.5 ms/tick) — already in
  the agent_data cacheline pulled by other reads in those loops.
* Combined wounded-and-alive: **small** standalone, but fits the
  cascade-CSE pattern (Area 9) — savings come from sharing across
  rules, not from a single rule.

### Implementation cost

* **Per-bitmap derive kernel:** ~50 lines WGSL + 30 lines Rust pack +
  one BGL slot. Compare to `crates/engine_gpu/src/alive_bitmap.rs` for
  the existing template.
* **Emitter rewrite:** for each predicate, ~10-30 lines per emitter
  (mask, scoring, physics WGSL). Pattern is already established — the
  alive-bitmap commit `1634e55a` is the template. Repeat 4-5 times.
* **CPU-side parity:** trivial; CPU pack/read costs nothing in the
  benchmarks (CPU is L1-bound everywhere already).
* **Slot budget:** scoring BGL is at slot 22 (alive bitmap) + 5 view
  slots + 4 base slots = 11 used of 16 max. Adding 4 more bitmaps
  pushes to 15 — uncomfortable. **Suggests bundling bitmaps into a
  single buffer indexed by predicate-id, OR passing a packed
  bool-vector via push constants when count grows.** This becomes a
  separate compiler-design question once usage grows.

### Risks

* Per-tick derive-kernel dispatch overhead (small N pessimism — the
  derive kernel costs more than the inline read at N<5k). Mitigation:
  conditional emit gated on `agent_cap` threshold (Area 7 specialisation).
* Bitmap-per-threshold proliferation. Need to choose a small set
  (~3-5 bitmaps total) or generalise to a "bitmap atlas" with one
  binding and a derive-kernel-per-threshold pre-pass.
* Threshold values are config-tunable (e.g. `combat.attack_range`); if
  a threshold becomes config-driven, the bitmap is stale across config
  changes. Limit the pattern to **constant thresholds** like 0.3, 0.5,
  0.8 — config-derived thresholds stay scalar.

### Priority recommendation

**4/5** for `hp_pct < 0.5` bitmap (covers cascade CSE simultaneously).
**3/5** for the rest individually. **High aggregate** if framed as a
"predicate-bitmap atlas" emitter feature rather than 5 separate
optimisations.

---

## Area 2 — Expression-level IR optimizations

### What it would do

Standard compiler hygiene applied to the IR before lowering: CSE for
repeated reads, constant folding for compile-time-known values, dead
code elimination for provably-false `when` clauses.

### Concrete instances

#### Common subexpression elimination (CSE)

Cross-rule CSE — multiple physics rules reading the same fields after
destructuring the same event:

* `crates/engine/src/generated/physics/mod.rs:33-46` —
  `dispatch_agent_attacked` calls 3 handlers (`chronicle_attack`,
  `chronicle_wound`, `rally_on_wound`). All 3 read `state.tick`. Two
  (`chronicle_wound`, `rally_on_wound`) recompute
  `cur_hp / max_hp` and re-test `< 0.5` independently. The
  generated bodies (lines 11-15 of each) are byte-equal in their
  hp/max_hp/hp_pct prelude.
* `dispatch_agent_died` calls 3 handlers
  (`chronicle_death`, `engagement_on_death`, `fear_spread_on_death`).
  The agent_id read is already destructured once but pos read for
  fear_spread (12 m radius) and engagement_on_death's
  `engaged_with_or` are independent; could share a "dead.pos"
  prelude.
* `damage` rule (`physics.sim:23-43`): reads `agents.alive(t)`,
  `agents.shield_hp(t)`, `agents.hp(t)` — all three pull the same
  AgentSlot cacheline today. Already cohesive; the issue is that
  `chronicle_wound`/`rally_on_wound` re-pull it fresh after the
  cascade dispatches.

Within-rule CSE:

* `engagement_on_move` (`physics.sim:438-471`): reads
  `agents.engaged_with_or(mover, mover)` once but the body uses
  `mover` ~8 times — the destructure already handles that.
* `damage`/`opportunity_attack`: both read `agents.hp(t)` and could
  share with the rally check, but they emit their `AgentAttacked`
  event before checking lethal — the rally rule then re-reads `hp`
  on the post-event side. **Cross-phase CSE has correctness
  implications** since events are emitted between reads.

#### Constant folding

* `config.movement.max_move_radius` (`config.sim:58`, default 20.0),
  `config.combat.attack_range` (default 2.0), `config.combat.kin_radius`
  (default 12.0): currently passed via the `SimCfg` storage buffer
  (`crates/dsl_compiler/src/emit_sim_cfg.rs:25-50`). Every shader
  invocation does a load. **Default values are known at emit time.**
  If we add an `@inline_default` pragma or detect that a config field
  is never overridden in the shipped TOML, the emitter could splice
  the literal directly into shader code.
* This pattern would also apply to `cascade.max_iterations = 8` (used
  in `physics cast` rule), and `max_announce_recipients = 32`. Some
  config knobs are **never tuned in practice**; baking them removes
  one buffer load per use.

#### Strength reduction

* `hp_pct = cur_hp / max_hp` in cascade physics rules (chronicle_wound,
  rally_on_wound). Already partially solved on GPU (`AgentData::hp_pct`
  is precomputed CPU-side, see `emit_scoring_wgsl.rs:372-381`). The
  CPU-side cascade rules still call `cur_hp / max_hp` per-rule — could
  use the same precomputed field if SoA exposes it. **Currently NOT
  shared between CPU cascade and GPU scoring's precompute.**
* No obvious sqrt/division candidates beyond hp_pct in the current
  surface.

#### Dead-code elimination

* Mask/scoring rows whose `when` clauses contain identifiers that
  resolve to `0.0` literals are stripped only at the IR level today.
  E.g. scoring's "personality dims 8..12" all return 0.0 in
  `read_field` (`emit_scoring_wgsl.rs:1310-1318`). If a scoring
  predicate compares against personality field-id 8, the comparison is
  provably `lhs > 0.0` for any positive RHS — DCE could drop the row.
  **No instances in the current scoring table** (no row uses
  personality field-ids), but the dead branches in `read_field` are
  always emitted. Tiny.
* `engaged_with` view is bound in scoring BGL but **never read** — see
  the scoring view-read research (now archived in git history) finding #4. DCE
  could detect "view spec emitted but never referenced by any
  predicate" and skip the binding.

### Estimated impact

* **CSE across cascade dispatches: medium**. The cascade dispatcher
  has the structure to make it nearly free (it already destructures
  the event once). Hoisting `cur_hp / max_hp` to the dispatcher saves
  ~1 division per attacked agent, not asymptotic. Probably ~0.1-0.3
  ms/tick at N=100k — small but free win once the pattern lands.
* **Constant folding configs**: small per-call but **per-shader-
  invocation**, so multiplied across 100k agents × ~5 reads/call =
  500k saves/tick. At ~1 ns each that's 0.5 ms/tick. Saves a couple
  binding slots if all SimCfg uses become inlined.
* **DCE: small** for the engaged_with-in-scoring-BGL case (one slot,
  no cycles). The personality DCE buys nothing today.

### Implementation cost

* **CSE:** medium. The IR walker already exists (`resolve.rs`); adding
  a hash-based deduplication pass on `IrExpr` nodes is well-trodden
  ground, ~300 LoC. Cross-rule CSE in the dispatcher needs a new
  emitter pass that knows about handler bodies — bigger, ~600 LoC.
* **Constant folding:** small for default values. ~50 LoC + a
  `@inline_default` annotation in `config.sim`. The `SimCfg` plumbing
  stays for runtime-tunable values.
* **DCE:** small. Walk `read_field` arms + scoring rows; flag unread
  views. ~100 LoC. (Note: removing engaged_with from scoring BGL is
  already a recommendation in the view-read-decomposition doc.)

### Risks

* CSE across emit phases (sync/batch) needs care — sync invokes a
  fresh body per call, batch fuses. Hoisting in batch but not sync
  drifts behaviour.
* Constant folding configs that the user *might* want to override
  later breaks the runtime balance-tuning loop. Need an explicit
  "this is bake-time" annotation.
* DCE of an "unused" view today might fire if a row is added later;
  needs to be re-runnable on every compile, not cached.

### Priority recommendation

**3/5** for cascade-dispatch CSE (compounds with predicate bitmaps —
the two together cover the AgentAttacked fan-out CSE end-to-end).
**2/5** for constant folding (small wins, but cheap to land).
**2/5** for DCE (reclaim BGL slots, prep for view growth).

---

## Area 3 — Kernel structural decisions

### What it would do

Today the emitter produces one structural shape per kernel kind: one
fused mask kernel, one fused scoring kernel, one physics shader. The
mask sub-phase measurement (archived research) found that **fusion was a 36% loss for masks** at
N=100k (split ran at 24,218 µs/tick vs fused 37,794 µs/tick). The
scoring row decomposition found a 10× split inflation (i.e. fusion
**won** for scoring rows). The compiler currently can't know which
direction is right per kernel.

### Concrete instances

* **Mask kernel split (potential ship)**. Per the mask sub-phase doc
  finding #5, splitting the fused mask kernel into self-only +
  movetoward + attack would save ~13.6 ms/tick at N=100k. The
  measurement code is reverted but the win is real. **Currently a
  point fix; would benefit from a "split-by-default at N≥X" emitter
  rule.**
* **Scoring kernel split (don't)**. Per the row-decomposition doc, it
  loses 10× at N=100k. Confirms: fusion stays.
* **Per-event dispatch vs candidate iteration**: scoring's MoveToward
  and Attack rows iterate `0..N` candidate slots
  (`emit_scoring_wgsl.rs` candidate walk), filtering by alive +
  radius + (for Attack) hostility. **Spatial-hash-backed candidates
  are already produced by the mask kernel** (`mask_attack_candidates`,
  `mask_move_toward_candidates` — see `crates/engine/src/generated/
  mask/{attack,move_toward}.rs`), but scoring re-walks `0..N` instead
  of consuming the mask's candidate output. This is the largest single
  scoring opportunity — see Area 1's "engagement bitmap" + Area 5's
  "view storage shape" for adjacent patterns.
* **Workgroup size selection**: fixed at 64
  (`emit_mask_wgsl.rs:84`, `emit_scoring_wgsl.rs:111`,
  `emit_view_wgsl.rs:1115/1242/1761/1884`). NVIDIA wave sizes are
  32; AMD are 64. 64 covers both reasonably but is suboptimal for
  small kernels (high latency wakeup) and for memory-bound kernels
  (occupancy is shared-memory-pressure-driven). Auto-selection would
  vary by kernel: tiny derive kernels probably want 256, scoring's
  big-register-set kernel probably wants 32-64.

### Estimated impact

* **Mask kernel split-by-default**: 13.6 ms/tick saved at N=100k
  (measured). At N=10k extrapolated to ~1.4 ms/tick. **Confirmed
  large.**
* **Scoring candidate-source rewrite**: large. The split decomp doc
  attributes 167 ms/tick to MoveToward + Attack candidate walks under
  the split (10× inflated) and 16.6 ms in the fused kernel. Reducing
  the candidate set from `0..N` (100k) to spatial-hash output (~10-50)
  is a 1000-5000× reduction in candidates. Even with 5× constant
  overhead from spatial-hash indirection, it's ~10× speedup → **could
  save 10+ ms/tick at N=100k**. Caveat: the mask kernel's existing
  candidate output isn't yet a single GPU buffer the scoring kernel
  can consume directly; that's a backend change, not just emitter.
* **Workgroup size**: small (~5-15% kernel-time tweak, varies). Mostly
  matters for tiny derive kernels where the absolute µs is low.

### Implementation cost

* **Split-by-default rule**: medium. Need an emit-time `agent_cap`
  hint OR runtime `if (agent_cap > X) { use_split() }` pattern.
  Easier: emit both, dispatch picker chooses at runtime. ~200 LoC
  emitter + dispatcher.
* **Spatial-hash candidate consumption in scoring**: large. The mask
  kernel writes candidate IDs to a per-agent ring/topk buffer; scoring
  needs to read that. New shared buffer schema, new BGL binding, new
  parity test. ~500-1000 LoC across emitter + backend + tests.
* **Workgroup size selection**: small. Heuristic-based emit-time
  decision per entry-point. ~50 LoC.

### Risks

* Split mask kernel adds 3 dispatches → ~30 µs of dispatch overhead.
  Confirmed measurable but smaller than the fusion loss at N=100k.
  Below N=10k the dispatch overhead may erase the savings — emitter
  needs N-aware dispatch (Area 7).
* Spatial-hash candidate output schema is the high-bandwidth interface
  between mask + scoring. Wrong shape locks future work in. Worth a
  spike before commit.
* Workgroup size affects shared-memory pressure; an auto-picker has
  to know about register count, which the emitter doesn't track today.

### Priority recommendation

**5/5** — the spatial-hash candidate rewrite is the single largest
identified scoring optimisation in the codebase. Worth a separate
plan+spike.
**4/5** for mask split-by-default (measured win, low risk).
**1/5** for workgroup size auto-selection (small wins, brittle).

---

## Area 4 — Memory layout optimizations

### What it would do

Hot/cold split of the agent data structure. Today every emitter shapes
its agent data the same way (or close to it) — a 64-byte struct in
GPU scoring (`scoring.rs:117-144`), various SoA arrays in mask /
physics. Reads of cold fields pull the same cacheline as hot fields,
wasting bandwidth.

### Concrete instances

* **`GpuAgentData` is 64 bytes** with 16 f32-equivalent slots (see
  `emit_scoring_wgsl.rs:361-389`). Hot fields (pos[3], hp, hp_pct,
  alive, creature_type, max_hp): ~28 B. Cold fields (shield_hp,
  attack_range, hunger, thirst, fatigue, target_hp_pct_unused, 2 pads):
  36 B of same-cacheline overhead per agent for paths that just want
  pos+alive (e.g. mask candidate walk).
* **Mask candidate walk** (`emit_mask_wgsl.rs` candidate loop) reads
  `agent_data[t].pos` + `agent_data[t].alive` + `agent_data[t]
  .creature_type` per candidate. Three fields, all in the first 16
  bytes of the struct — one cacheline pull, OK. BUT the emitter
  currently dispatches `agents.pos(x)` to the same `agent_data`
  buffer rather than to the dedicated `hot_pos` SoA buffer that the
  CPU-side mask kernel reads. **Confirmed in the alive-bitmap commit
  message** — the alive bitmap landed because that exact pattern (per-
  candidate cacheline pull just for the alive bit) was the dominant
  cost.
* **`per_entity_topk(K=8)` slot arrays** are 8 slots × 12 bytes = 96 B
  per agent (see `crates/engine/src/generated/views/threat_level.rs`,
  `pack_focus.rs`, etc.). If average occupancy is <3, that's ~65%
  waste in storage AND in the K-scan work — at 1.23M `threat_level`
  reads/tick (per the view-read decomposition doc), reducing the
  scan from 8 to 4 saves ~half the L2 atomic loads. **Sparsification
  would be a 2× win on threat_level alone (~4 ms/tick at N=100k).**
* **Always-zero-after-clamp fields**: `view::engaged_with` is K=1
  (`views.sim:95`) but is bound in scoring BGL and never read (see
  view-read decomposition #4). The "DCE binding" win is here.
* **Cold-cacheline split**: an emitter-driven hot/cold partitioning
  pass would identify the read-frequency of each field per kernel
  and produce split structs. The infra to count reads doesn't exist
  yet but the data is statically derivable from emit walks.

### Estimated impact

* **Hot/cold AgentData split**: medium-large (~3-5 ms/tick at N=100k).
  The mask kernel's per-candidate cacheline-pull is one of the
  remaining costs after the alive bitmap; halving the cacheline span
  for the `hot_pos + alive + ct` path is in scope. **Caveat: hp_pct
  is hot too (used in scoring's target-side reads), and it's
  precomputed — splitting needs to keep hp_pct in the hot half.**
* **K=4 sparsification of pair-keyed views**: medium-large
  (~3-4 ms/tick). Halves the K-scan worst-case for `threat_level`
  (the dominant view at 42% of reads). Risk: eviction churn in
  high-threat clusters (a wolf surrounded by 5+ attackers loses
  threat memory faster). Needs an eviction-pressure measurement
  before commit.
* **DCE of engaged_with from scoring BGL**: small (one slot, no µs).
  Reclaims BGL headroom for future bitmap atlases.
* **Auto-detect always-zero fields**: speculative. Needs flow analysis
  + clamp-bound propagation. Probably not worth standalone effort.

### Implementation cost

* **Hot/cold AgentData split**: large. Affects the GpuAgentData layout
  in `emit_scoring_wgsl.rs`, the matching Rust struct in
  `engine_gpu/src/scoring.rs`, the upload path in `pack_agent_data`,
  the WGSL `read_field` switch (which currently unpacks the unified
  struct), and every parity test. ~800-1200 LoC + a careful migration
  plan. **High risk of parity drift.**
* **K=4 sparsification**: small. Storage hint param change in
  `views.sim` (already supported syntax). Re-run the parity tests.
  Adversarial test: high-density combat cluster eviction churn.
* **DCE BGL slots**: small. Already enumerated above.

### Risks

* Hot/cold split changes the wire format between CPU pack + GPU
  consume; needs a schema-hash bump and regen of every parity baseline.
* K=4 sparsification: behavioural drift for high-density clusters.
  Bench the stress fixtures (`chronicle_batch_stress_n20k`) to confirm
  no change in qualitative outcomes.
* DCE BGL: zero risk if "unused" really means unused; the view-read
  decomposition doc proved engaged_with is unread today.

### Priority recommendation

**4/5** — K=4 sparsification of the dominant pair-keyed view is the
highest-value, lowest-risk single change here.
**3/5** — hot/cold AgentData split is potentially big but expensive +
risky; defer until other levers are exhausted.
**2/5** — DCE BGL slots is housekeeping.

---

## Area 5 — View storage shape suggestions / auto-detection

### What it would do

Today storage shape (`@symmetric_pair_topk`, `@per_entity_ring`,
`@per_entity_topk`, `pair_map`) is hand-annotated in `views.sim`. The
compiler could analyse access patterns and suggest the right shape, or
detect mis-annotations and warn.

### Concrete instances

* **`memory(observer, source)` is `@per_entity_ring(K=64)`**
  (`views.sim:323-328`). Used for FIFO hearsay. If a scoring row ever
  reads it with a specific `(observer, source)` pair, ring storage is
  ~K reads to find the entry; a hashmap lookup would be 1-2 reads.
  **No scoring row reads it today** — but the shape choice was
  intentional (FIFO order is part of the semantics). **Auto-detection
  would here say "ring is correct" because reads are wildcard
  cursor scans, not pair lookups.**
* **`engaged_with(a)` is `@per_entity_topk(K=1)`** (`views.sim:95`).
  Single-slot per agent — effectively a HashMap. The K=1 distinction
  exists but the storage is a special case in the emitter
  (`emit_view.rs:597-660`). A shape-suggestion pass could identify
  that K=1 + no decay = "use a flat vec<Option<AgentId>>" and rewrite.
  Currently it goes through the topk K=1 path, slightly slower than
  necessary.
* **`@symmetric_pair_topk(K=8)` for `standing`** (`views.sim:292`):
  reads use canonical pair canonicalisation. If reads in practice
  are always specific `(a, b)` pairs (no wildcards), a flat hashmap
  keyed on `(min, max)` would beat the K-scan for sparse populations.
  **Auto-detection: count specific-vs-wildcard read sites at emit
  time + measure populations at runtime.**
* **`threat_level` / `kin_fear` / `pack_focus` / `rally_boost`** are
  `@per_entity_topk(K=8)` + `@decay`. Wildcard sums are 8 atomic
  loads regardless of population. If average population is ~2-3, K=4
  or K=2 is fine. The view-read decomposition doc's recommendation #4
  ("Profile whether K=8 is too large") fits here.

### Estimated impact

* **Auto-suggest K=4 from access patterns**: medium (3-4 ms/tick on
  threat_level alone — same as Area 4's recommendation, framed as a
  compiler feature instead of a manual change).
* **K=1 → flat-vec specialisation for engaged_with**: small (~0.2-0.5
  ms/tick) — already efficient.
* **Suggest hashmap vs topk**: speculative. Needs runtime population
  measurement; the emitter alone can't know.

### Implementation cost

* **Access-pattern analysis** (count specific-vs-wildcard reads per
  view): small. ~150 LoC IR walker, prints recommendations to stdout
  on `compile-dsl --analyse`. **Doesn't change emitted code; just
  recommends.**
* **K=1 → flat-vec rewrite**: small-medium. Add a new storage
  shape-arm to `emit_view.rs` for the K=1 case (~200 LoC).
* **Runtime population instrumentation** (for K-suggestion): medium.
  Same shape as the per-view read counter from
  the scoring view-read research (archived) — count avg + p99
  occupancy per slot. ~300 LoC, opt-in via env var.

### Risks

* Auto-suggestions risk being too aggressive. Recommendations should
  ship as compile-time warnings, not silent rewrites.
* Population-driven K is workload-dependent — a benchmark that's K-
  sufficient might evict in a different scenario.

### Priority recommendation

**3/5** — the access-pattern analysis tool is cheap and would have
caught the engaged_with-not-read case the view-read decomp found.
**2/5** for the K=1 → flat-vec specialisation (small win).
**2/5** for runtime population instrumentation (preparatory; pairs
with K=4 sparsification when that lands).

---

## Area 6 — Type-aware primitive packing

### What it would do

Many fields use full f32/u32/i32 but the values fit in much narrower
ranges. Pack into smaller widths or bit-fields where lossless.

### Concrete instances

* **Standing is i32 clamped [-1000, 1000]** (`views.sim:296`). Fits in
  i16. Halves the StandingEdge struct (`generated/views/standing.rs:12-20`)
  from 12 bytes to 8 bytes (or 6 bytes if other-id packs to u24). At
  N=100k × K=8 slots = 800k slots × 4 B savings = 3.2 MB less in
  per-tick traffic. Cacheline-touch reduced by ~33%.
* **AbilityHint** (subsystem 3, 4 values): 2 bits. Currently
  represented as u32. 16× density possible.
* **AbilityTag** (Phase 1, 19 tags per ability per
  `assets/abilities/`): bit-array. Currently `Vec<(AbilityTag, f32)>`.
  Bit-array would be 19 bits = 3 B vs ~36 B per ability for the Vec.
  Combined with weights it's still ~25 B vs 36 B.
* **`creature_type: u32` in AgentData** — only 4 species. Fits in 2
  bits, but the cacheline savings are minimal because of struct
  alignment.
* **Slow factor (q8)** is i16 already; no improvement.
* **Decay anchor tick (u32)** in TopkSlot: u32 supports ~430M ticks at
  10 Hz = ~50 days. Could be u24 with rollover, but the savings (1 B
  in a 12 B struct) hurt alignment.

### Estimated impact

* **i16 standing**: small-medium (~3 MB less GPU memory, ~5-10% faster
  StandingEdge scans). Probably ~0.2-0.5 ms/tick if standing reads are
  hot — and they're not in the current scoring table. **Latent win,
  realised when standing-driven scoring rows arrive.**
* **AbilityTag bit-array**: medium (relevant when ability evaluator
  lands on GPU per the ability-eval-gpu-migration design doc). Reduces
  per-ability metadata footprint ~5×.
* **Creature type bit packing**: tiny.

### Implementation cost

* **i16 narrowing for standing**: small. Just change the IR type +
  emitter widening rules (already widened from i16 in commit
  `5fbdb71a` for the inverse direction). ~100 LoC + parity test.
* **AbilityTag bit-array**: medium. New IR primitive needed (`@bits`
  storage hint?). ~400 LoC + parser + emitter.
* **Cross-cutting bit-packing infra**: large. Probably not worth a
  dedicated lane unless multiple primitives sign up.

### Risks

* Narrowing past CPU's natural width adds shift+mask overhead per
  read. Modern x86 + GPU make this ~free, but the win has to
  outweigh the load that's already L1.
* Bit-array changes parity hash; one schema bump per change.

### Priority recommendation

**2/5** — interesting, but speculative wins. Standing narrowing fits
the existing widening infrastructure cleanly; bit-arrays are bigger
work for a future ability-eval-on-GPU push.

---

## Area 7 — Batch-vs-sync specialization

### What it would do

The emitter produces ONE WGSL per kernel that runs on both batch
(N=100k, 50-tick paths) and sync (N=50, parity tests) paths. Each path
has very different optimal shapes:

* Sync: small N, no spatial hash needed (direct N² beats hash setup),
  smaller K values, tighter workgroup, no per-tick derive kernels.
* Batch: large N, spatial hash mandatory, larger K, derive kernels pay
  off.

Today both paths share one WGSL.

### Concrete instances

* **Mask kernel candidate-walk threshold**: at N=50, walking 0..N is
  faster than spatial-hash setup. Per the mask sub-phase doc, the
  fused kernel was sized for the worst case. Specialising for N<X
  would skip spatial-hash overhead for parity tests.
* **Workgroup size 64**: at N=50, 64 threads → 1 workgroup; the
  remaining 14 lanes are idle. 32 would be better. At N=100k it
  doesn't matter.
* **Topk K**: K=8 makes sense for N=100k where many agents observe
  many threats. At N=50, the population is bounded by 49; K=4 is
  always enough. **No win at scale; just simpler code paths.**
* **Per-tick derive kernels** (alive bitmap, future predicate
  bitmaps): dispatch overhead ~10-30 µs each. At N=50 the consumer
  reads 50 cachelines anyway — derive kernel is pure overhead. Shrinks
  the win below threshold.

### Estimated impact

* **Sync-path bypass for derive kernels**: medium (sync tests are
  fast already; this is about closing the gap, not opening one).
  Probably ~5-15% sync-test wall-clock improvement; not a perf win on
  the hot path.
* **Specialised batch kernel choices**: small at scale (the batch
  path is already tuned for N=100k); the win is from removing the
  "dual-mode" complexity.

### Implementation cost

* **Two-shape emit per kernel**: medium. Add an "intent" param
  (`KernelIntent::Sync | Batch`) to emit fns. ~400 LoC across the
  emitters. Risk: dual maintenance burden.
* **Compile-time specialisation via const generics**: speculative for
  WGSL; the WGSL compiler doesn't have generics. Would have to
  produce two WGSL strings.

### Risks

* Code duplication between sync + batch shaders. Test surface doubles.
* Sync path is the parity reference; any divergence between the
  emitted shapes risks parity drift.

### Priority recommendation

**1/5** — the sync path is fast enough; this is mostly housekeeping.
Worth doing only if the hot path forces a kernel shape (e.g. spatial-
hash candidate consumption in scoring) that the parity tests can't
exercise without adapter-N machinery.

---

## Area 8 — Pipeline cache + AOT shader compilation

### What it would do

WGSL → SPIR-V → driver pipeline compilation happens at first dispatch.
Per the per-dispatch attribution doc, this costs measurable startup
time (NVIDIA SPIR-V JIT visible in samply). Could pre-compile at
backend init or ship pre-warmed pipeline binaries.

### Concrete instances

* **First-dispatch pipeline compile**: visible in samply traces per
  the per-dispatch attribution research. Adds ~50-200 ms to the first
  tick after a backend init. Currently amortised across the warmup
  sync step but visible in stress runs.
* **Adapter-specific binaries**: wgpu can serialise the compiled
  pipeline to disk and skip the SPIR-V compile next run. Would shave
  ~50-100 ms cold-start.
* **Cross-run cache**: Vulkan has `VkPipelineCache`; wgpu exposes
  `Adapter::pipeline_cache_data()` recently.

### Estimated impact

* **Cold-start improvement**: 50-200 ms once per run. **Doesn't help
  steady-state perf; helps developer iteration speed and CI.**
* **No batch-mode improvement**: pipeline cache is a constant cost,
  not a per-tick one. By tick 50 of a 50-tick batch run, it's
  amortised.

### Implementation cost

* **wgpu pipeline cache API**: small. Wire `pipeline_cache_data` into
  backend init + persist to `~/.cache/...`. ~150 LoC.
* **Pre-warm at startup**: small. Issue dummy dispatches against
  every entry-point during backend init so first real dispatch is
  cached. ~50 LoC.

### Risks

* Cache poisoning if the pipeline cache survives a binary change.
  Hash the compiled WGSL into the cache key (we already have schema
  hashes; reuse).
* Driver-version coupling — pipeline binaries are driver-specific.
  Already handled by wgpu but worth noting.

### Priority recommendation

**1/5** — cold-start only; doesn't move the needle on the steady-
state perf budget. Pick up only if developer iteration is annoying.

---

## Area 9 — Predicate-fusion across rules (cascade CSE)

### What it would do

Multiple physics rules in the cascade dispatcher fire on the same
event kind (e.g. `AgentAttacked` triggers `chronicle_attack` +
`chronicle_wound` + `rally_on_wound`). Each independently re-derives
its predicate. A pre-pass that hoists shared predicates to the
dispatcher level and passes the result to each rule body would
eliminate redundant work.

This is Area 2's CSE specialised to the cascade-dispatcher pattern.

### Concrete instances

* **`AgentAttacked` fan-out** (`generated/physics/mod.rs:33-46`):
  - `chronicle_attack`: just emits `ChronicleEntry`. No predicate.
  - `chronicle_wound`: reads `agent_alive(t)` + `hp/max_hp` + tests
    `< 0.5`.
  - `rally_on_wound`: reads `agent_alive(t)` + `hp/max_hp` + tests
    `< 0.5` + iterates `nearby_kin`.
  Rules 2 + 3 do the **exact same predicate**. A hoisted
  `let wounded_alive = state.agent_alive(t) && hp/max_hp < 0.5;`
  in the dispatcher passes the bool to both, eliminating one division
  + comparison per AgentAttacked event.
* **`AgentDied` fan-out**:
  - `chronicle_death`: emits ChronicleEntry.
  - `engagement_on_death`: reads `engaged_with_or(dead, dead)`.
  - `fear_spread_on_death`: iterates `nearby_kin(dead, 12.0)`.
  No obvious shared subexpression beyond the `dead` bind, which is
  already shared by destructure.
* **`EngagementCommitted` fan-out**:
  - `chronicle_engagement`: emits.
  - `pack_focus_on_engagement`: iterates `nearby_kin(actor, 12.0)`.
  No shared.

In short: only `AgentAttacked` has obvious fusion potential today.
But as more rules join the cascade (subsystem 3 ability-eval rules
are scheduled to fire on every event), the pattern compounds.

### Estimated impact

* **AgentAttacked CSE**: small per event (~1 division + 1 comparison
  saved). At ~215k attacks/tick (per the mask sub-phase doc's headline
  fixture), that's ~430k ops/tick saved. At 1 ns each = 0.4 ms/tick.
  Real but not headline.
* **Future fanout (subsystem 3 ability-eval)**: medium-large; depends
  on how many shared predicates land. If every event-kind dispatcher
  ends up with 5-10 handlers (per the GPU-everything-scoping doc), CSE
  could save 3-5 ops/event. At 200k+ events/tick, ~1-3 ms/tick.
* **Combined with predicate bitmaps (Area 1)**: the
  `wounded_alive` predicate IS a bitmap. Implementing both together
  collapses to one per-tick derive + one per-rule bit-read. **Cleaner
  than CSE-the-expression.**

### Implementation cost

* **Cascade-dispatcher CSE pass**: medium. Identify shared subtrees
  across handlers in the same dispatcher; hoist into the dispatcher
  body; pass as fn args. ~500 LoC + signature plumbing through
  `emit_dispatcher_call` and the per-rule emitters.

### Risks

* Cross-handler hoisting breaks the per-rule encapsulation — a rule
  with a bug in its predicate now needs the dispatcher to be re-emitted
  too. Surface area for "did the hoist match the body" parity bugs.
* Some predicates have side effects (state writes). Hoisting must be
  pure-only. The `agents.alive(t)` check is pure; `agents.set_*` calls
  cannot be hoisted. The IR already distinguishes these.

### Priority recommendation

**2/5** today; **3/5** if subsystem 3's ability-eval cascade lands as
planned (then the fan-out fattens and CSE compounds). Worth doing
**after** Area 1's predicate bitmaps; the bitmap pattern subsumes most
of what CSE would save.

---

## Area 10 — Generated-code "shape" emitter quality

### What it would do

Improve the readability + debuggability of emitted code:

* WGSL: line-number annotations pointing back to `.sim` source lines.
* CPU Rust: rustfmt-clean output (already mostly there but parens are
  noisy — see `damage.rs:14` `(shield).min(a)` or `chronicle_wound.rs:14`
  `(max_hp > 0.0)`).
* Inline DSL source comments showing which DSL clause produced each
  block. Makes the generated code self-documenting.

### Concrete instances

* **Generated CPU files are mostly clean**, but excess parens persist
  in conditions (e.g. `if (max_hp > 0.0)` in `chronicle_wound.rs:14`).
  Rustfmt would strip them; the emitter could too.
* **No source-line cross-references**: `chronicle_wound.rs` says
  "GENERATED by dsl_compiler from assets/sim/physics.sim" but doesn't
  cite the line range for the rule. When debugging a rule, you scan
  the file. ~20 lines per rule wouldn't be a big add.
* **WGSL source maps are nonexistent** — the generated WGSL is one
  long string with no annotations. Hard to debug a parity drift when
  you don't know which DSL clause produced which WGSL block.

### Estimated impact

* **Zero perf impact.** This is purely developer ergonomics.
* **Compounds**: every future emitter feature is cheaper to debug if
  the output is structured.

### Implementation cost

* **CPU paren cleanup**: trivial (~20 LoC). Could just run rustfmt as
  a post-pass.
* **WGSL line-comment annotations**: small (~150 LoC across the WGSL
  emitters).
* **Source-map JSON sidecar files**: medium (~400 LoC) if we want
  proper line-to-line tracking.

### Risks

* None on the perf path. Documentation-style comments slightly bloat
  WGSL source string size; negligible.

### Priority recommendation

**2/5** — non-urgent but always positive. Land alongside any major
emitter refactor (every refactor benefits from better debug output).

---

## Top-10 ranked compiler work (next quarter)

Ranked by `(estimated impact at N=100k) × (confidence) ÷ (implementation
cost)`. Items with ⚠ have notable risk that could change the ranking.

| Rank | Area | Item | Est. impact | Est. cost | Confidence |
|---:|------|------|-------------|-----------|------------|
| **1** | 3 | **Spatial-hash candidate consumption in scoring kernel** (replace `0..N` walks with mask-emitted candidate lists) | ~10+ ms/tick at N=100k | Large | High (decomp doc proves rows are bounded by candidate walks, not row math) |
| **2** | 4 | **K=4 sparsification of pair-keyed views** (threat_level + my_enemies + pack_focus + rally_boost from K=8 → K=4) | 3-4 ms/tick | Small | Medium ⚠ (eviction churn risk, needs adversarial bench) |
| **3** | 1 | **`hp_pct < 0.5` predicate bitmap** + cascade CSE (Area 9) for AgentAttacked fan-out | 1-3 ms/tick | Medium | Medium (extends proven alive-bitmap pattern) |
| **4** | 3 | **Mask kernel split-by-default at large N** (proven 13.6 ms/tick win at N=100k) | 13.6 ms/tick at N=100k, gated on N | Medium | High (measured) |
| **5** | 2 | **Cascade-dispatcher CSE infrastructure** (hoist shared predicates across handlers) | 0.4-3 ms/tick depending on subsystem-3 fan-out | Medium | Medium |
| **6** | 5 | **Access-pattern analysis tool** (compile-time emit of view read counts + K-suggestions) | 0 perf, but informs Area 4 | Small | High |
| **7** | 2 | **Constant folding for default-only config values** (bake `kin_radius=12.0`, `attack_range=2.0`, etc. when no TOML override) | 0.5 ms/tick + BGL slot reclaim | Small | Medium |
| **8** | 1 | **`is_stunned(x)` + `engaged_with != None` predicate bitmaps** (extend bitmap atlas) | 0.5-2 ms/tick combined | Small (per pattern repetition) | Medium |
| **9** | 4/5 | **Remove `engaged_with` from scoring BGL** (DCE — confirmed unused) | 0 ms but reclaims slot for atlas | Trivial | High |
| **10** | 10 | **Source-map annotations + rustfmt-clean CPU output** (developer ergonomics) | 0 perf, compounds future debug speed | Small | High |

### Cross-area meta-observations

1. **The bitmap atlas is the highest-leverage emitter primitive.**
   Items #3 + #8 share infrastructure. If we land a generic predicate-
   bitmap-derive emitter (one binding, one derive kernel per
   threshold), all four predicates fall out cheap. Reframe these as
   "build the bitmap atlas, then 4 instantiations". **The alive
   bitmap was instance one. Make it a pattern.**

2. **Spatial-hash candidate consumption is the single largest knob.**
   It needs a separate plan + spike before commit, but it is the
   ONLY identified item that would shave double-digit ms/tick. Treat
   it as a quarter-defining initiative, not an emitter tweak.

3. **CSE + predicate bitmaps overlap.** Items #3 and #5 share the
   AgentAttacked fan-out. Doing #3 first subsumes most of #5's wins;
   #5 only justifies itself once subsystem 3 ability-eval cascade
   lands and dispatchers grow to 5-10 handlers per kind.

4. **Per-decl emitter complexity is at a manageable level today**
   (~30 KLoC). Two-shape (sync/batch) specialisation (Area 7) and
   AOT pipeline cache (Area 8) are noticeable code adds for small
   wins; defer.

5. **The exhaustive walker fix (commit `4c47dc5f`) is the unsung
   compounder.** Every future emitter feature is more reliable
   because of it. The ranking above doesn't include "more
   exhaustiveness audits" because the work is already done — but
   the principle (replace `_ => {}` catch-alls with explicit arms)
   should govern every future emitter PR.

### Out of scope

* Bevy ECS-side optimisations.
* Runtime profiler improvements (Stage A is mature).
* Self-play / RL training pipeline performance.
* WebGPU-specific tweaks (browser path is not a perf priority).

---

## Honest confidence notes

The impact numbers above are **rough estimates** derived from tonight's
measurements (the alive bitmap's 49% reduction is the only concrete
data point on a single emitter-pattern change). Individual items could
swing 2-3× either direction once measured. The ranking is a
prioritisation tool, not a budget.

Two items that could re-rank if measured:

* **Hot/cold AgentData split (Area 4)**: estimated 3-5 ms/tick but
  could be much smaller if the L1 hit rate on the unified struct is
  already high. Wants a Stage A sub-phase measurement before commit.
  Currently rank-omitted (would slot ~6 if confirmed).
* **K=4 sparsification (#2)**: depends on actual occupancy. If
  threat_level slots are usually 6-8 full in combat clusters, K=4
  loses information and qualitative behaviour drifts. Ship behind a
  feature flag with an A/B parity run on the wolves+humans scenario
  before committing.

The top recommendation (#1, spatial-hash candidate consumption) is
high-confidence in *direction* but high-uncertainty in
*implementation cost*. A 1-week spike + a parity-test plan would
de-risk it before committing the full ~1000 LoC build.
