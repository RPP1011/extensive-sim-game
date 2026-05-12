# Gaps surfaced by `pirate_fleet`

Adversarial naval-skirmish fixture (16 Pirates vs 16 Navy, 500 ticks)
designed to surface the **mutating creature_type mid-sim** surface.
Companion files:

- `assets/sim/pirate_fleet.sim`
- `assets/ability_test/pirate_fleet/{Board,Cannons}.ability`
- `crates/sims/tests/pirate_fleet_pin.rs`

The fixture build emits 17 kernels and steps 500 ticks without panic;
the pin reports per-faction populations, treasure conservation, kills,
and ownership transfers. Test PASSES with the gaps documented below.

## Gap 1 — `agents.set_creature_type` not in setter allowlist (NEW class)

**Severity:** spec/contract — caller-visible feature gap. Silently
downgrades the consumer rule body.

**Where:**
`crates/dsl_compiler/src/cg/lower/physics.rs::agents_setter_field`
(lines 916–960). The match arm enumerates the 22 allowlisted setters
(`set_pos`, `set_vel`, `set_hp`, `set_alive`, `set_mana`, `set_shield_hp`,
the 6 status-expires-at-tick fields, the 4 buff multipliers, the disguise
pair, the 4 busy-state slots, `set_hunger`). `set_creature_type` is NOT
among them.

`AgentFieldId::CreatureType` exists in
`crates/dsl_compiler/src/cg/data_handle.rs:286`, has type
`OptEnumU32`, lowers to `agent_creature_type` buffer with
`AccessMode::AtomicStorage` (kernel.rs:1462). The SoA column is
fully wired on the read side (`self.creature_type == Pirate`
predicates work) and on host-write side (`agent_creature_type_buf`
is exposed on `GeneratedRuntime` for seeding) — only the DSL setter
arm is missing.

**Symptom:**

```
warning: sims@0.1.0: [pirate_fleet lower diag] physics#6 body at \
    10227..10258 contains AST statement `Expr` which has no \
    CG-statement equivalent yet
```

The `AttemptOwnershipFlip` rule body — a single expression statement
`agents.set_creature_type(t, nt)` — surfaces as
`UnsupportedPhysicsStmt { ast_label: "Expr" }`. The build_helper
tolerates lower errors, so the **entire rule body is dropped** and
no `physics_AttemptOwnershipFlip` kernel appears in emit-stats.

The pin reports `ownership_transfers = 0` slots flipped, despite 500
ticks of boarding cadence + a pair-keyed boarding view that records
each `Boarded` event firing.

**Why this is the right adversarial surface:** every prior fixture
treats `creature_type` as a constant set at host-seed time and
referenced only in `where` predicates (e.g. `self.creature_type ==
Enemy`). No fixture has mutated it mid-sim. The setter machinery has
never been exercised because nobody asked.

**Fix sketch:** add `"set_creature_type" => Some(&AgentFieldId::CreatureType)`
to `agents_setter_field`. The downstream `lower_agents_setter` arm
(lines 976+) already routes `AgentRef::Target(<expr_id>)` for non-self
writes, so a single-line allowlist edit may suffice. Two follow-up
verifications needed:
1. **Atomic-store semantics for u32 column** — `agent_creature_type`
   is `AtomicStorage` (kernel.rs:1462), so the `CgStmt::Assign`
   lowering needs to emit an `atomicStore` not a plain SoA store.
   Today's hp/mana writes are `f32` plain stores; mana is
   `ReadWriteStorage`. The status `expires_at_tick` writers are u32 —
   if they go through `atomicStore` already, this lifts onto the
   same path; if they're plain stores, a parallel u32-atomic store
   path is needed.
2. **Schema-hash invariance** — the engine schema_hash test pins
   the SoA column layout (`crates/engine/tests/schema_hash.rs`).
   Adding a setter arm doesn't change the column set, so the hash
   should not bump; verify with `cargo test -p engine schema_hash`.

A `creature_type` write from a chronicle consumer is the canonical
"transfer-of-ownership" gameplay primitive (boarding actions, mind
control, possession spells, faction swap on diplomatic events).
Without it, every such mechanic has to be implemented host-side via
chronicle drain + CPU mutation, which breaks the
"single-DSL-emit-fully-on-GPU" property the rest of the system aims for.

## Gap 2 — Chronicle-consumer Indirect dispatch (KNOWN, re-surfaced)

**Severity:** P0 for any apply_ability-using fixture. Already documented
in `crates/sims/tests/hill_raid_pin.rs` ("zero damage flowed —
chronicle-consumer Indirect-dispatch gap from commit 353527e6").
This fixture ALSO trips it.

**Where:** `crates/dsl_compiler/src/build_helper.rs`'s synthesized
`step()` body. The hill_raid pin docs it as: "apply_ability records
enqueue but consumers don't fire from synthesized step()."

**Symptom (this fixture):**

```
  damage:  total=0.0  by_pirates=0.0  by_navy=0.0
  treasure delta (all slots):  0.00
  boarding attempts (sum of pair-keyed view): 0.0
```

The fleet engagement positions are correct (Pirates start at y=-8 with
vel.y=+0.04, Navy at y=+8 with vel.y=-0.04, lines meet at tick ~200,
cross during ticks 200..400 within `cannon_range = 8.0` AND
`board_range = 1.5` of each other). All three views recorded zero
events: damage_dealt (single-key per source), boarding_attempts
(pair-keyed). EffectDamageApplied / EffectGoldTransfer chronicle
records aren't reaching the post-phase consumers.

**This re-confirms the gap class:** ANY mega-crate fixture that uses
`apply_ability` for damage / gold transfer / status effects will see
zero engagement until the chronicle-consumer Indirect dispatch wiring
lands in the synthesized step(). This is independent of pirate_fleet's
adversarial design — wave_defense / duel_25v25 / mass_battle_100v100
on the mega-crate would all show the same.

## Gap 3 — Cone AOE Path B (untested at fleet scale; LATENT)

**Severity:** unverified. **Could not measure** because Gap 2 prevents
any chronicle records reaching the consumer. The Cone WGSL emit branch
exists (`cg/emit/wgsl_body.rs:2728`+) and 4+ lol_heroes .ability files
declare `damage X in cone(4.0, 60.0)` patterns, but no end-to-end pin
asserts cone-shaped damage actually lands on the right targets.

`pirate_fleet`'s `Cannons.ability`:
```
damage 6.0 in cone(8.0, 90.0) [PHYSICAL: 100]
```
generates a Cone-shape AOE. The `aoe_dispatch=true` flag is enabled
(`[pirate_fleet ability-corpus] 2 .ability files, aoe_dispatch=true`),
so the AOE Path B walk should fire — but the chronicle records can't
be observed without Gap 2 closing.

**Suggested follow-up:** once Gap 2 closes, this fixture is the
natural place to pin cone AOE coverage:
- Ships in cone arc (forward 90° half-angle from caster) take damage,
  ships outside the arc do not.
- Per-cast damage scales with in-cone neighbour density (compare
  Cannons-only damage vs Spread-area variant on identical topology).

## Gap 4 — @runtime config setter per-tick coalescing (LATENT)

**Severity:** unverified perf concern. Adversarial probe DESIGNED for
this — pin re-writes `wind_strength` BEFORE EVERY `step()` (vs other
fixtures that write once at try_new). Test passed with no asserted
change in behaviour, but no per-tick perf measurement was taken.

The codegen path
(`build_helper.rs::synthesize_runtime_core_a2`, lines 1245–1277) emits
one `gpu.queue.write_buffer` call per kernel that references the field
— at 17 kernels in pirate_fleet, that's 17 buffer-write calls per
tick from `set_config_fleet_wind_strength`. Suggested follow-up:
batch-write all `@runtime` config field updates into a single
end-of-tick uniform-buffer flush rather than per-setter scatter
writes. Will matter at fixtures with > 50 kernels OR > 5 @runtime
fields each setter-spammed at full tick rate.

## Verified non-issues

- **`damage X in cone(R, A)` parses cleanly** — `assets/ability_test/
  pirate_fleet/Cannons.ability` parses and lowers without
  diagnostics; 17 kernels emit (vs 16 in a circle variant — the
  Cone arm doesn't add a kernel, just a different walk in the AOE
  dispatcher body). Unblocks Gap 3 follow-up.
- **`transfer_gold N` parses cleanly** — `Board.ability` declares
  `transfer_gold 5` and the registry build doesn't complain
  (`[pirate_fleet ability-corpus] 2 .ability files, aoe_dispatch=true`).
  Whether the EffectGoldTransfer chronicle record is actually
  written remains unverified due to Gap 2.
- **Ability slot ordering matches alphabetical** — Board (1) <
  Cannons (2). `apply_ability 1` and `apply_ability 2` literals in
  the .sim resolve to the registered programs.
- **decl-order alphabetical discriminants** — Navy = 0, Pirate = 1.
  Pin verifies no slot ends up with a third value (would indicate a
  bogus mid-sim creature_type write); count holds at exactly 16/16.
- **No host-side panics across 500 ticks** — `step()` runs 500x with
  per-tick wind setter calls without GPU validation errors,
  buffer-overflow asserts, or NaN propagation.
- **Treasure conservation INVARIANT trivially holds** — sum-of-mana
  across all slots stays at 800 gold (the seed value) because Gap 2
  prevented any transfer events from firing. Cannot verify the
  EffectGoldTransfer drain rule's correctness until Gap 2 closes.

## Workspace hygiene

- `cargo build --workspace` clean.
- `RUST_MIN_STACK=33554432 cargo test -p sims --release --test pirate_fleet_pin` passes.
- `dsl_compiler::resolve_one pirate_fleet` reports `RESOLVE OK`.
- 17 kernels emit (15 fully-fired, 1 dropped at lower time —
  AttemptOwnershipFlip — and 1 hill_raid-class chronicle-consumer
  blocked downstream).
