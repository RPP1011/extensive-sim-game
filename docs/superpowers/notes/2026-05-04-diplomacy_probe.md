# Diplomacy probe — discovery report (2026-05-04)

This is the report from the smallest possible end-to-end probe of
**diplomacy / coalition formation** — the first .sim that combines
ALL of the recently-landed surfaces into a real game-design pattern:

  - 2 `entity X : Group` declarations (Faction + Coalition).
  - 1 `entity Diplomat : Agent`.
  - ToM-shape pair_map u32 view: `view trust(observer, target) -> u32`
    with bit-OR self-update fold (post `51b5853b` landing).
  - Verb cascade with TWO competing verbs (ProposeAlliance + Betray)
    gated on **disjoint Mod-tick predicates** (`world.tick % 3 == 0`
    vs `world.tick % 3 != 0`).
  - Multi-event-kind ring — `Observed`, `AllianceProposed`,
    `Betrayed` — three distinct event kinds folded by three
    separate handlers reading the SAME ring (per-handler tag
    filter, post `cb24fd69`).
  - First fixture exercising `world.tick % N == 0` inside a verb's
    `when` clause (Mod operator, post `7208912f`).

## Outcome

**OUTCOME (a) FULL FIRE.** All three observables match analytical
predictions exactly after 100 ticks at AGENT_COUNT=32:

```
diplomacy_probe_app: starting — seed=0x0D11051AC0A11710 agents=32 ticks=100
diplomacy_probe_app: finished — final tick=100 agents=32 trust.len()=1024 alliances.len()=32 betrayals.len()=32
diplomacy_probe_app: trust — diagonal-set: 32/32 ; off-diagonal-set: 0/992
diplomacy_probe_app: alliances_proposed — min=34.000 mean=34.000 max=34.000 sum=1088.000 zero_slots=0
diplomacy_probe_app: betrayals_committed — min=66.000 mean=66.000 max=66.000 sum=2112.000 zero_slots=0
diplomacy_probe_app: expected per-slot — alliances=34.000 betrayals=66.000 (sum=100.000)
diplomacy_probe_app: OUTCOME = (a) FULL FIRE — all three observables match analytical predictions.
  - pair_map u32 view fold (trust) lights up diagonal cleanly
  - verb cascade with TWO competing verbs gated on disjoint Mod-tick predicates routes correctly
  - per-handler tag filter scales to THREE distinct event kinds in one ring
  - 2 Group entity declarations + 1 Agent entity + 3 views (mixed u32 + 2 f32) compile + dispatch cleanly
diplomacy_probe_app: preview alliances_proposed[0..8] = [34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0]
  preview betrayals_committed[0..8] = [66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0]
diplomacy_probe_app: preview trust[i*N + i] for i in 0..8 = [1, 1, 1, 1, 1, 1, 1, 1]
```

## Files added

- `assets/sim/diplomacy_probe.sim` (~170 LOC) — probe fixture.
  3 events (Observed + AllianceProposed + Betrayed), 1 Agent
  entity (Diplomat), 2 Group entities (Faction + Coalition),
  2 verbs with Mod-tick `when` predicates, 1 per-agent physics
  rule (ObserveAndAct), 3 views (trust pair_map u32 +
  alliances_proposed f32 + betrayals_committed f32).
- `crates/diplomacy_probe_runtime/Cargo.toml` (~25 LOC).
- `crates/diplomacy_probe_runtime/build.rs` (~110 LOC) — mirrors
  `abilities_runtime/build.rs`.
- `crates/diplomacy_probe_runtime/src/lib.rs` (~480 LOC) — full
  per-tick dispatch chain (10 rounds: clears → ObserveAndAct →
  fused mask → scoring → 2 chronicles → seed → 3 folds).
- `crates/sim_app/src/diplomacy_probe_app.rs` (~150 LOC) — harness
  driving 100 ticks; reports trust diagonal/off-diagonal counts,
  alliances/betrayals per slot, OUTCOME line.
- `Cargo.toml` (workspace) — added `diplomacy_probe_runtime` member.
- `crates/sim_app/Cargo.toml` — added the dep + `[[bin]]` entry.
- `crates/dsl_compiler/tests/stress_fixtures_compile.rs` — added
  `diplomacy_probe_compile_gate` test (passing) — locks the
  structural surface (2 MaskPredicate + 3 PhysicsRule + 3 ViewFold +
  2 scoring rows), the `(tick % 3u)` reference in the mask kernel,
  the `atomicOr` reference in fold_trust, and naga validation
  cleanliness.

Net LOC added: ~935 source lines + ~65 doc lines (above the
~700 budget; the runtime lib.rs explanatory comments dominate.
The compile-gate test alone is ~95 LOC since it pins each emitted
surface against the discovered shape).

## Compiler topology — what got emitted

The compiler lowered the program to **14 ComputeOps**:

```
OpId(0):  PerAgent  :: MaskPredicate(MaskId(0))   — ProposeAlliance's `when world.tick % 3 == 0`
OpId(1):  PerAgent  :: MaskPredicate(MaskId(1))   — Betray's          `when world.tick % 3 != 0`
OpId(2):  PerEvent  :: ViewFold(trust,                on=Observed)
OpId(3):  PerEvent  :: ViewFold(alliances_proposed,   on=AllianceProposed)
OpId(4):  PerEvent  :: ViewFold(betrayals_committed,  on=Betrayed)
OpId(5):  PerAgent  :: PhysicsRule(ObserveAndAct, on_event=None — Tick)
OpId(6):  PerEvent  :: PhysicsRule(verb_chronicle_ProposeAlliance, on=ActionSelected)
OpId(7):  PerEvent  :: PhysicsRule(verb_chronicle_Betray,          on=ActionSelected)
OpId(8):  PerAgent  :: ScoringArgmax(2 rows)
OpId(9):  OneShot   :: Plumbing(UploadSimCfg)
OpId(10): PerAgent  :: Plumbing(PackAgents)
OpId(11): OneShot   :: Plumbing(SeedIndirectArgs(ring=0))
OpId(12): PerAgent  :: Plumbing(UnpackAgents)
OpId(13): OneShot   :: Plumbing(KickSnapshot)
```

The scheduler emitted **13 kernels** — the two MaskPredicate ops
fused into one (`fused_mask_verb_ProposeAlliance`); everything
else stayed standalone (no cross-domain fusion gap surfaced):

```
fused_mask_verb_ProposeAlliance         ← op#0 + op#1 fused (PerAgent siblings)
fold_trust                              ← op#2 standalone (atomicOr u32)
fold_alliances_proposed                 ← op#3 standalone (f32 RMW)
fold_betrayals_committed                ← op#4 standalone (f32 RMW)
physics_ObserveAndAct                   ← op#5 standalone
physics_verb_chronicle_ProposeAlliance  ← op#6 standalone
physics_verb_chronicle_Betray           ← op#7 standalone
scoring                                 ← op#8 (with 2 mask gates)
upload_sim_cfg / pack_agents / seed_indirect_0 / unpack_agents / kick_snapshot ← admin
```

Per the build.rs `cargo:warning=[diplomacy emit-stats]` lines:
all 13 kernels naga-validate cleanly.

### Mask kernel — both predicates lower correctly

```wgsl
// op#0 (mask_predicate)
{
    let mask_0_value: bool = ((tick % 3u) == 0u);
    if (mask_0_value) {
        let mask_0_word = agent_id >> 5u;
        let mask_0_bit  = 1u << (agent_id & 31u);
        atomicOr(&mask_0_bitmap[mask_0_word], mask_0_bit);
    }
}

// op#1 (mask_predicate)
{
    let mask_1_value: bool = ((tick % 3u) != 0u);
    if (mask_1_value) {
        let mask_1_word = agent_id >> 5u;
        let mask_1_bit  = 1u << (agent_id & 31u);
        atomicOr(&mask_1_bitmap[mask_1_word], mask_1_bit);
    }
}
```

`world.tick % 3 == 0` lowers to `(tick % 3u) == 0u` — confirms the
post-`7208912f` Mod operator landing AND the `tick = cfg.tick` PerAgent
preamble both work inside a verb's `when` clause.

### Scoring — mask gates filter rows

```wgsl
// row 0: action=#0 (ProposeAlliance, score=1.0)
{
    let mask_0_word_for_row_0 = agent_id >> 5u;
    let mask_0_bit_for_row_0  = 1u << (agent_id & 31u);
    let mask_0_loaded_for_row_0 = atomicLoad(&mask_0_bitmap[mask_0_word_for_row_0]);
    if ((mask_0_loaded_for_row_0 & mask_0_bit_for_row_0) != 0u) {
        let utility_0: f32 = config_2;
        if (utility_0 > best_utility) {
            best_utility = utility_0;
            best_action = 0u;
            ...
        }
    }
}
// row 1: action=#1 (Betray, score=0.5) — symmetric, gated on mask_1.
```

EXACTLY ONE mask is set per agent per tick (predicates are
disjoint), so EXACTLY ONE row contributes to argmax, and that
verb is chosen. The scoring kernel correctly emits one
`ActionSelected{ actor=agent_id, action_id=best_action,
target=NO_TARGET }` per agent per tick.

## Discovered gaps

### Gap #1 (LOW) — bare `tick` identifier doesn't lower in verb `when` clauses

**Severity:** LOW (workaround documented in fixture).

**Location:** `crates/dsl_compiler/src/cg/lower/expr.rs:589`
(`IrExpr::Namespace` arm of expr lowering rejects the bare token
with `LoweringError::UnsupportedAstNode { ast_label: "Namespace" }`).

**Characterization:** A bare `tick` token in a verb's `when` clause
parses to `IrExpr::Namespace(tick)` (since `tick` is reserved as
a namespace-style identifier the AST resolver has shadowed for the
`world.tick` field). The CG expr lowering pass routes the bare
`Namespace` through an unsupported-variant arm, accumulates a
`LoweringError`, and silently DROPS the mask predicate — no
MaskPredicate op is added to the compilation. The verb then
becomes "always passes the mask" by virtue of having no mask gate
at all, which means the higher-scoring verb wins argmax every tick
and the lower-scoring verb never fires.

**Workaround:** Use the fully-qualified `world.tick` instead of
`tick`. The `world.tick` form lowers cleanly through the
`(NamespaceId::World, tick)` registry arm at `expr.rs:2597` and
produces a kernel-preamble local that the BinOp::Mod path
consumes correctly (verified end-to-end by this probe).

**Discovery path:** Initially wrote `when (tick % 3 == 0)`; the
discovery test logged `LOWER DIAG: lowering: AST variant Namespace
at <span> is not yet supported` AND emitted only 12 kernels (no
mask kernel — one fewer than the verified 13). Switching to
`world.tick` immediately produced the missing mask kernel +
all-clean naga + the correct per-tick alternation.

**Spec status:** `tick` is documented as a per-tick local in the
spec (`docs/spec/dsl.md`'s tick preamble section); the inconsistency
is that it works inside `physics ... { ... }` bodies but not inside
verb `when` predicates. The cooldown_probe physics body uses
`world.tick >= ready_at` (the qualified form) — no fixture today
relies on the unqualified `tick` inside a `when` clause.

**Follow-up:** Either (a) extend `IrExpr::Namespace` lowering to
recognise `tick` and route it through the same kernel-preamble
local resolution path that `world.tick` uses, or (b) reject
`tick` as a bare token at the AST resolve layer with a typed
"use `world.tick` instead" diagnostic so the gap surfaces upstream
of CG lowering. Option (b) is the minimum-surface fix; option (a)
is the user-friendly fix.

### Gap #2 (MEDIUM) — chronicle event_count must over-count when sharing a ring

**Severity:** MEDIUM (runtime gotcha; emerges when multiple
producers fire BEFORE a chronicle dispatches in the same tick).

**Location:** `crates/diplomacy_probe_runtime/src/lib.rs` —
chronicle event_count was initially set to `agent_count`
(matching the abilities_probe pattern); had to be raised to
`agent_count * 2` for both verb chronicles to fire correctly.

**Characterization:** The diplomacy probe's per-tick chain has
the ObserveAndAct physics rule emit one `Observed` event per
agent FIRST (step 2). The scoring kernel then emits one
`ActionSelected` per agent (step 4). When a chronicle dispatches
with `event_count = agent_count`, it iterates slots 0..N — but
those are all `Observed` events (kind 1u), not `ActionSelected`
(kind 4u) which lives at slots N..2N. The chronicle's
per-handler tag filter correctly rejects all the `Observed`
slots, but the workgroup early-returns before ever reaching the
ActionSelected slots. Result: NO chronicle emits, alliances +
betrayals views stay at 0.0.

**Workaround:** Size chronicle event_count generously to cover
the upper bound of all events emitted by prior step phases.
For diplomacy, `agent_count * 2` (Observed + ActionSelected)
suffices; for fixtures with more producers, the bound grows.

**Why this didn't bite abilities_probe:** abilities_probe's
chronicle dispatches AFTER scoring (the only producer in that
fixture), so `event_count = agent_count` exactly matches the
ActionSelected count and the scoring slots are at indices 0..N.
This probe is the FIRST fixture where a chronicle competes with
prior producers for ring slots within a tick.

**Spec status:** Not documented. The chronicle dispatch sizing is
host-side runtime concern (the kernel itself is per-event-tag
filtered).

**Follow-up:** Either (a) document the chronicle event_count
sizing rule in the runtime-pattern doc with a reference to this
probe's gap, or (b) extend the scheduler to emit a per-chronicle
`ring_slot_count` upper bound that accounts for prior producers
in the same tick. Option (a) is the minimum-surface fix; option
(b) prevents the next probe author from re-discovering this.

### Gap #3 (LOW) — `config.<ns>.<u32_field>` resolves to f32 in expression position

**Severity:** LOW (easy literal workaround; surfaces a config-field
type-resolution corner the spec doesn't audit yet).

**Location:** the type inference for `config.<namespace>.<field>`
references in the verb `when` predicate's arithmetic expression
context. Diagnostic from the lower pass: `lowering: binary 'Mod' at
<span> has mismatched operands — lhs is u32, rhs is f32`. The lhs
(`world.tick`) correctly resolves to u32 via the world-namespace
registry; the rhs (`config.diplomacy.observation_tick_mod`)
resolves to f32 even though the field is declared `u32 = 3` in the
config block.

**Characterization:** Same compilation has the field's type
correctly recorded in the config-namespace registry as `u32`, but
the expression-context lowering of `config.<ns>.<field>` returns
the value as `f32` (likely the path is going through a
generic-numeric defaulting arm that doesn't consult the field's
declared type before binding the operand type for downstream
arithmetic). Result: any `world.tick % config.<ns>.<u32_field>`
fails the BinaryOperandTyMismatch check and the mask predicate
drops with the same silent-DROP gap as Gap #1 (no MaskPredicate
op added).

**Workaround:** Use a literal `3` in the verb `when` predicate's
Mod expression. The fixture keeps the `observation_tick_mod: u32 = 3`
config field declared (declaration-only) so a future config-type-
resolution fix can reactivate the live reference with one swap.

**Why this didn't bite tom_probe / trade_market_probe:** Neither
fixture consumes a u32 config field in expression position. Both
inject u32 values via runtime cfg uniforms (e.g., trade_market's
`observation_bit` is declared but never read inside a sim rule;
tom_probe's `fact_bit: 1` is a literal in the emit body).

**Spec status:** `docs/spec/dsl.md` lists config field types but
doesn't audit how typed config-field access lowers to typed CG
expressions. The mismatch surfaces only in arithmetic contexts
where operand types must agree — direct usage in `score
config.<ns>.<f32_field>` or `score config.<ns>.<f32_field>` works
because both sides are f32 and the expression context is a
single-operand utility return.

**Follow-up:** Trace `config.<ns>.<field>` lowering through
`crates/dsl_compiler/src/cg/lower/expr.rs` and check whether the
field's declared type from `comp.config_blocks` is consulted
before binding the CgTy of the expression result. The minimum-
surface fix is propagating the field's declared type through the
lowering arm; the cleaner fix would also tighten the spec-side
type rules around config-field access.

### Notes on what DID work first try

- **Mod operator inside verb `when` (post-`7208912f`):** lowered
  cleanly via the `(BinOp::Mod, CgTy::Tick)` arm at `expr.rs:1553`
  once `world.tick` was used instead of bare `tick`.
- **3-event-kind ring + 3 separate fold handlers:** the per-handler
  tag filter (`cb24fd69`) scaled cleanly — `fold_trust` only
  matches kind 1u, `fold_alliances_proposed` only matches kind 2u,
  `fold_betrayals_committed` only matches kind 3u. Three folds
  reading the same ring with no cross-contamination.
- **pair_map u32 view + 2 f32 views in same compilation:** the
  mixed-storage path composed cleanly. `fold_trust` uses
  `atomicOr` on a `u32` storage; the two f32 folds use the
  CAS-add loop on `f32` storage. No binding mis-routing.
- **2 Group entity declarations:** the entity_field_catalog walks
  both Faction + Coalition without conflict (auction_market has 1
  Group; this probe walks past index 1).
- **Verb cascade with disjoint mask predicates:** the scoring
  kernel correctly handles "exactly one verb's row passes the
  mask gate" — the unmasked row early-exits before contributing
  to argmax, and the masked row wins.

## Reproducer

```bash
cargo build -p diplomacy_probe_runtime              # clean
cargo run -p sim_app --bin diplomacy_probe_app      # OUTCOME (a) FULL FIRE
cargo test -p dsl_compiler --test stress_fixtures_compile diplomacy_probe_compile_gate -- --nocapture
```

## Constitution touch-points

- **P1 Compiler-First:** no compiler changes. Both gaps are
  documented as follow-ups (Gap #1 is a small lowering extension;
  Gap #2 is a runtime-pattern doc addition). Probe surfaces them
  as data, not panics.
- **P9:** closing with verified commit (runtime builds clean,
  workspace tests pass — 22 stress-fixture tests including the new
  diplomacy_probe_compile_gate, the existing 14 fixture apps
  unchanged, OUTCOME (a) FULL FIRE on the new probe).
- **P11:** atomic primitives reused — the runtime composes
  `EventRing`, `ViewStorage`, `GpuContext` plus a local pair_map
  storage allocation pattern lifted verbatim from
  `tom_probe_runtime`. No new GPU helpers.
