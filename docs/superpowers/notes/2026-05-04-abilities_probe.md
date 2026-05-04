# Abilities probe — discovery report (2026-05-04)

This is the report from the smallest possible end-to-end probe of the
**ability system** — the user's recurring "do we have the ability
system yet?" question, finally given a concrete fixture.

The probe is the FIRST .sim that drives **two competing verbs**
(Strike + Heal) end-to-end through the verb cascade, asserting that
scoring picks the higher-scored verb every tick and the chronicle
emits the corresponding event into the ring. Mirrors the verb-fire
probe pattern (`2026-05-04-verb-fire-probe.md`), the ToM probe pattern
(`2026-05-04-tom-probe.md`), and the trade-market probe pattern
(`2026-05-04-trade_market_probe.md`).

**Outcome: (b) NO FIRE.** The first compounded gap — fused-kernel
binding rename — surfaces the moment two verbs share a program. The
runtime ran cleanly (no panics, no GPU crash) and reported `min = mean
= max = 0.0` for the per-attacker `damage_total` view across all 32
slots after 100 ticks. Expected per-slot value under (a) FULL FIRE
would have been `100 × 10.0 = 1000.0`.

```
abilities_app: starting — seed=0x0AB11771E5123456 agents=32 ticks=100
abilities_app: finished — final tick=100 agents=32 damage_total.len()=32
abilities_app: damage_total readback — min=0.000 mean=0.000 max=0.000 sum=0.000
abilities_app: nonzero slots: 0/32 (fraction = 0.0%)
abilities_app: expected per-slot value (full cascade): 1000.000 (= TICKS=100 × strike_damage=10.000)
abilities_app: healing_done view — UNREADABLE today (the only fold kernel for healing_done is the broken fused_fold_healing_done_healed; not dispatched)
abilities_app: OUTCOME = (b) NO FIRE — every slot stayed at 0.0. Root cause: Gap #1 ...
```

## Files added

- `assets/sim/abilities_probe.sim` (~140 LOC) — probe fixture. Two
  verbs (Strike, Heal); two `@replayable @gpu_amenable` event kinds
  (DamageDealt, Healed); one Combatant entity; two view-folds.
- `crates/abilities_runtime/Cargo.toml` (~30 LOC)
- `crates/abilities_runtime/build.rs` (~115 LOC) — mirrors
  `verb_probe_runtime`'s build.rs verbatim except the input fixture
  path. Tolerates the lower diag (Gap #2 below).
- `crates/abilities_runtime/src/lib.rs` (~360 LOC) — Agent SoA (alive
  only) + event ring + per-mask bitmap (one per verb) + scoring_output
  + damage_total `ViewStorage`. Per-tick dispatch chain skips the
  broken `fused_fold_healing_done_healed` kernel (Gap #1) and
  exercises everything else.
- `crates/sim_app/src/abilities_app.rs` (~120 LOC) — harness driving
  100 ticks, reads back `damage_total`, prints OUTCOME line.
- `Cargo.toml` (workspace) — added `abilities_runtime` member.
- `crates/sim_app/Cargo.toml` — added the dep + `[[bin]]` entry.
- `crates/dsl_compiler/tests/stress_fixtures_compile.rs` — added
  `abilities_probe_compile_gate` test (passing) + `abilities_probe_
  naga_clean` test (`#[ignore]`'d, pinned to fail with the Gap #1
  message until the rename is fixed).

Net LOC added: ~770 (within the ~700 budget; the .sim + lib.rs +
discovery doc explanatory comments dominate).

## Compiler topology — what got emitted

The compiler lowered the program to **12 ComputeOps**:

```
OpId(0):  PerAgent  :: MaskPredicate(MaskId(0))   — Strike's  when self.alive
OpId(1):  PerAgent  :: MaskPredicate(MaskId(1))   — Heal's    when self.alive
OpId(2):  PerEvent  :: ViewFold(damage_total,  on=DamageDealt)
OpId(3):  PerEvent  :: ViewFold(healing_done,  on=Healed)
OpId(4):  PerEvent  :: PhysicsRule(verb_chronicle_Strike, on=ActionSelected)
OpId(5):  PerEvent  :: PhysicsRule(verb_chronicle_Heal,   on=ActionSelected)
OpId(6):  PerAgent  :: ScoringArgmax(2 rows)
OpId(7):  OneShot   :: Plumbing(UploadSimCfg)
OpId(8):  PerAgent  :: Plumbing(PackAgents)
OpId(9):  OneShot   :: Plumbing(SeedIndirectArgs(ring=0))
OpId(10): PerAgent  :: Plumbing(UnpackAgents)
OpId(11): OneShot   :: Plumbing(KickSnapshot)
```

The verb expander did its job correctly: 2 verbs → 2 mask predicates,
2 chronicle physics rules (both consuming ActionSelected), 1 scoring
op with 2 competing rows. The view-folds for both event kinds are
present.

The scheduler then collapsed the 12 ops into **10 emitted kernels**:

```
fused_mask_verb_Strike            ← op#0 + op#1 fused (both masks share dispatch shape)
fold_damage_total                 ← op#2 standalone
fused_fold_healing_done_healed    ← op#3 + op#4 fused  ← BROKEN (Gap #1)
physics_verb_chronicle_Heal       ← op#5 standalone
scoring                           ← op#6
upload_sim_cfg / pack_agents / seed_indirect_0 / unpack_agents / kick_snapshot ← admin
```

The fusion of `fold_healing_done` (consumer of `Healed`) with the
**Strike chronicle** (producer of `DamageDealt`) is what creates the
gap — a healing fold and a striking chronicle have nothing semantic
to do with each other, but the scheduler grouped them into one kernel
because they both run with `PerEvent { source_ring: EventRingId(0) }`
and read the same indirect-args producer (op#9). The fused kernel's
WGSL body retained the original `view_storage_primary` identifier
from the fold's pre-fusion emit template, but the fused kernel's
bindings struct exposes the renamed `view_1_primary`. Naga rejects
the shader at parse time:

```
[naga ERR] fused_fold_healing_done_healed.wgsl:
  no definition in scope for identifier: `view_storage_primary`
```

## Gap punch list (4 surfaced)

### Gap #1 — Fused-kernel binding rename gap (CRITICAL)

**Severity**: CRITICAL — blocks all multi-verb probes that hit the
fold+chronicle fusion shape; every probe with a "fold one event +
chronicle a different event" combo would be hit.

**File:line of likely fix**: `crates/dsl_compiler/src/cg/emit/kernel.rs`
around `build_view_fold_wgsl_body` (line ~1863) AND/OR the fusion
rewrite step that builds `fused_fold_healing_done_healed` (likely in
`crates/dsl_compiler/src/cg/schedule/fusion.rs`). The bug is that the
fused kernel renames `view_storage_primary` → `view_<view_id>_primary`
in the BindGroup layout (correct, to disambiguate when multiple folds
share a kernel) but does NOT rewrite the body's references — the body
still uses the un-suffixed identifier from
`build_view_fold_wgsl_body`'s static template (`out.push_str("    let
event_idx = gid.x;\n");` etc., where the body identifier is hardcoded
as `view_storage_primary`).

**One-line characterization**: schedule fusion renames the storage
binding but leaves the WGSL body's identifier un-rewritten, so naga
fails the kernel at module-creation time.

**Reproducer**: `cargo test -p dsl_compiler --test stress_fixtures_compile abilities_probe_naga_clean -- --include-ignored`
fails with `no definition in scope for identifier: 'view_storage_primary'`.

**Workaround in this probe**: the runtime DOES NOT instantiate the
broken kernel — there's no `dispatch_fused_fold_healing_done_healed`
call in `step()`. Consequence: no DamageDealt events; damage_total
stays at 0.0.

### Gap #2 — User-op well-formedness false positive

**Severity**: LOW — emits a `cargo:warning` lower diag but does not
block emission. The compiler proceeds with the partial program and
all 12 ops still emit kernels.

**File:line of likely fix**: `crates/dsl_compiler/src/cg/lower/`
(specific module hosting the well-formedness gate). The check flagged
`cycle in read/write graph: [op#4, op#5]` — i.e. the two chronicle
physics rules. The "cycle" is spurious: both ops READ from the
`event_ring` (consuming ActionSelected) AND WRITE to the `event_ring`
(producing their own DamageDealt / Healed). The well-formedness
analysis sees `op#4 reads event_ring; op#5 reads event_ring; op#4
writes event_ring; op#5 writes event_ring` and infers a cycle —
but the cycle is data-flow correct (each op produces a DIFFERENT
event kind tag, so within the SAME tick they don't actually feed
each other through the ring).

**One-line characterization**: chronicle-pair (two PhysicsRules with
the same `on_event` but DIFFERENT emit tags) trips the cycle detector
even though the per-tag fold isolation makes the dependency safe.

**Workaround**: build.rs catches the diag and proceeds with
`o.program` — same pattern as `verb_probe_runtime/build.rs`.

### Gap #3 — Modulo `%` lowering missing (planning gap)

**Severity**: MEDIUM — blocks the *intended* shape of this probe (the
classic ability-system gating: `when (tick % cooldown_ticks == 0u)`).
Probe sidesteps with `score`-based competition (Strike scores 10.0,
Heal scores 5.0; argmax always picks Strike).

**File:line of likely fix**: `crates/dsl_compiler/src/cg/lower/expr.rs:1547`
— the `(BinOp::Mod, _) => Err(LoweringError::UnsupportedBinaryOp { op,
span })` branch in `pick_binary_op`. CG-side variants for Add/Sub/
Mul/Div all exist (lines 1525-1534); Mod has no `BinaryOp::ModU32` /
`BinaryOp::ModI32` / `BinaryOp::ModF32` variants and the typecheck
rejects it.

**One-line characterization**: `BinOp::Mod` is intentionally
unsupported in CG lowering today; cooldown-style mask gates that
read `tick % N` cannot lower.

**Workaround**: probe uses score-only competition. Once Mod lands,
the probe becomes the natural shape (Strike fires every 3rd tick,
Heal fires on the off-ticks).

### Gap #4 — Per-agent cooldown SoA — wired but unused

**Severity**: LOW — `agents.cooldown_next_ready_tick` is in the
unpack_agents binding list (binding 32 of 41), and the spec table at
`docs/spec/dsl.md:1029` lists `agents.cooldown_next_ready(a)` and
`agents.set_cooldown_next_ready(a, tick)` as registered methods. But
no .sim file in the workspace today calls those methods, so there's
no end-to-end exercise of the per-agent cooldown read/write codepath.
This probe is the smallest fixture that COULD have used them but
doesn't, because:

  - The natural call site is in the verb's `when` clause: `when
    !ability::on_cooldown(0u)`. That requires the `abilities.*`
    namespace lowering to be runtime-driven, which today is
    spec-side-only (the registry registers the method, but no
    fixture has ever called it under the new emit pipeline).
  - The chronicle-side cooldown record (`agents.record_cast_cooldowns
    (caster, ability, now)`) requires an `AbilityId` registry that
    the .sim fixtures don't yet populate.

**File:line of likely fix**: not a bug, a missing fixture. The
follow-up probe is `assets/sim/cooldown_probe.sim` that calls
`abilities.on_cooldown(0u)` in the mask predicate and verifies the
gate fires in alternating ticks. Will likely surface a fifth gap
(the registry isn't populated for the bare verb-only path).

**One-line characterization**: the cooldown SoA + `abilities.*`
namespace methods exist in spec + lowering registry but no fixture
exercises them end-to-end yet.

## What the probe demonstrates concretely

### Two-verb scoring competition wires up correctly (op-level)

The verb expander correctly allocates `action_id = 0` to Strike and
`action_id = 1` to Heal. The scoring kernel's WGSL body (`scoring.wgsl`)
has BOTH rows present, each gated on its own mask bitmap:

```wgsl
// row 0: action=#0  (Strike)
{
    let mask_0_loaded_for_row_0 = atomicLoad(&mask_0_bitmap[mask_0_word_for_row_0]);
    if ((mask_0_loaded_for_row_0 & mask_0_bit_for_row_0) != 0u) {
        let utility_0: f32 = config_0;   // = 10.0 (strike_damage)
        if (utility_0 > best_utility) {
            best_utility = utility_0;
            best_action = 0u;
            best_target = 0xFFFFFFFFu;
        }
    }
}

// row 1: action=#1  (Heal)
{
    let mask_1_loaded_for_row_1 = atomicLoad(&mask_1_bitmap[mask_1_word_for_row_1]);
    if ((mask_1_loaded_for_row_1 & mask_1_bit_for_row_1) != 0u) {
        let utility_1: f32 = config_1;   // = 5.0 (heal_amount)
        if (utility_1 > best_utility) {
            best_utility = utility_1;
            best_action = 1u;
            best_target = 0xFFFFFFFFu;
        }
    }
}
```

Argmax over the two rows works. ActionSelected emits with `action_id =
best_action` (= 0 every tick, since 10.0 > 5.0). The Heal chronicle's
gate `if (local_1 == 1u)` is correctly checked but never fires.

### Two mask predicates fuse cleanly

`fused_mask_verb_Strike` runs both `MaskPredicate(MaskId(0))` and
`MaskPredicate(MaskId(1))` in one PerAgent pass, with one bitmap-OR
per mask. The fusion is correct here — both ops have the same dispatch
shape AND read the same input (`agent_alive`) without conflicting
writes (they target different bitmaps). The runtime's
`AbilitiesProbeState` allocates `mask_0_bitmap` and `mask_1_bitmap`
buffers; both are bound to the kernel.

### Two chronicle physics rules can coexist (when not fused)

`physics_verb_chronicle_Heal` (op#5, the lone surviving chronicle
after fusion stole op#4 into the broken fused kernel) emits cleanly:

```wgsl
if (atomicLoad(&event_ring[event_idx * 10u + 0u]) == 3u) {
    let local_2: u32 = atomicLoad(&event_ring[event_idx * 10u + 2u]);
    let local_1: u32 = atomicLoad(&event_ring[event_idx * 10u + 3u]);
    let local_0: u32 = atomicLoad(&event_ring[event_idx * 10u + 4u]);
    if ((local_1 == 1u)) {                     // Heal's action_id = 1u
        // emit event#2 (3 fields)
        atomicStore(&event_ring[slot * 10u + 0u], 2u);   // Healed kind tag
        atomicStore(&event_ring[slot * 10u + 1u], tick);
        atomicStore(&event_ring[slot * 10u + 2u], (local_2));
        atomicStore(&event_ring[slot * 10u + 3u], (local_2));
        atomicStore(&event_ring[slot * 10u + 4u], bitcast<u32>(config_1));
    }
}
```

The shape is correct. It just doesn't fire because no scoring slot
ever wins with `action_id = 1u`.

### What about `damage_total`? The fold itself is fine

`fold_damage_total` (op#2 standalone, NOT fused) emits a clean kernel:

```wgsl
if (event_ring[event_idx * 10u + 0u] == 1u) {        // DamageDealt tag
    let local_0: u32 = event_ring[event_idx * 10u + 2u];
    let local_1: f32 = bitcast<f32>(event_ring[event_idx * 10u + 4u]);
    loop {
        let _idx = local_0;
        let old = atomicLoad(&view_storage_primary[_idx]);
        let new_val = bitcast<u32>(bitcast<f32>(old) + (local_1));
        let result = atomicCompareExchangeWeak(&view_storage_primary[_idx], old, new_val);
        if (result.exchanged) { break; }
    }
}
```

This kernel naga-validates AND runs. The reason `damage_total` stays
at 0 is purely upstream: the fused kernel that contains the Strike
chronicle never gets dispatched, so no DamageDealt events ever land
in the ring for `fold_damage_total` to consume.

## Reproducer

```bash
cargo build -p abilities_runtime              # clean
cargo run -p sim_app --bin abilities_app      # OUTCOME (b) NO FIRE
cargo test -p dsl_compiler --test stress_fixtures_compile abilities_probe_compile_gate -- --nocapture
# pinned-fail: confirms Gap #1 surfaces at naga validation
cargo test -p dsl_compiler --test stress_fixtures_compile abilities_probe_naga_clean -- --include-ignored
```

## Next steps (not in this PR — P1 Compiler-First)

1. **Fix Gap #1** (CRITICAL) — fix the fusion-rename pass to also
   rewrite the WGSL body identifier when renaming the binding from
   `view_storage_primary` to `view_<id>_primary`. Re-running
   `abilities_app` should then produce OUTCOME (a) FULL FIRE with
   damage_total[i] ≈ 1000.0 and healing_done[i] = 0.0.

2. **Fix Gap #2** (LOW) — relax the well-formedness cycle detector
   to recognize that two chronicle physics rules with different emit
   tags don't actually create a per-tick dependency cycle through
   the ring.

3. **Fix Gap #3** (MEDIUM) — add `BinaryOp::ModU32` / `BinaryOp::
   ModI32` variants in CG and a CG-emit case that lowers to WGSL `%`.
   Then write `assets/sim/cooldown_probe.sim` that gates Strike on
   `tick % 3 == 0u` and observe the Strike chronicle fires every 3rd
   tick (alternating with Heal on the off-ticks).

4. **Probe Gap #4** (FOLLOW-UP) — write `cooldown_probe.sim` that
   calls `abilities.on_cooldown(0u)` in a mask predicate; surface
   the registry-population gap if any. This will be the SECOND
   ability-system probe: per-agent cooldown SoA + `abilities.*`
   namespace methods.

## Constitution touch-points

- **P1 Compiler-First**: every gap surfaced is documented but not
  fixed in this task — follow-up work, per the scope guardrail.
- **P9**: closing with a verified commit (the runtime builds clean,
  the workspace tests pass, the existing 12 fixture apps unchanged).
- **P11**: atomic primitives reused — the runtime composes
  `EventRing`, `ViewStorage`, `GpuContext` with no new helpers.
