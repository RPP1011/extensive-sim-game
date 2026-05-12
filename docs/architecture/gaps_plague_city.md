# Plague City — Gap Discovery Log

The Plague City fixture is one of seven adversarial fixtures spawned in
parallel to surface compiler gaps that the toy `*_runtime` migrations
hid. The fixture stacks: per-class entity routing, lifecycle (deaths +
burials), pair-keyed N² ToM beliefs about infection, @decay belief
forgetting, multi-row scoring on healers (most-likely-sick target),
spatial proximity contagion, rng.action() stochastic transmission,
cross-event invariants (SIR-style population conservation), and a
500-tick equilibrium horizon.

This document is the running gap log. Entries are appended as gaps surface
during fixture authoring; each entry has the .sim line, the diagnostic
text emitted by `cargo build -p sims`, the gap class, and the workaround
adopted to keep the fixture moving.

---

## GAP P-A: `agents.set_<custom_field>` is hard-coded to a closed set

**Surface:** `agents.set_infected(p, 1);` in `ApplySickness`
(plague_city.sim ~L194); `agents.set_infected_since_tick(p, world.tick);`
likewise; the corresponding `agents.infected(self)` reader on every
mask-side gate referencing the field.

**Diag (build.rs):**

```
warning: sims@0.1.0: [plague_city lower diag] lowering: `self.infected`
  at 7973..7994 does not name an agent field
warning: sims@0.1.0: [plague_city lower diag] physics#1 body at
  8764..8789 contains AST statement `Expr` which has no CG-statement
  equivalent yet
```

**Gap class:** Closed-enum agent SoA. `crates/dsl_compiler/src/cg/lower/
physics.rs::agents_setter_field` matches a fixed set of `set_<name>`
strings (set_pos / set_vel / set_hp / set_alive / set_mana /
set_shield_hp / set_stun_expires_at_tick / set_slow_expires_at_tick /
set_slow_factor_q8 / set_root_expires_at_tick /
set_silence_expires_at_tick / set_fear_expires_at_tick /
set_taunt_expires_at_tick / set_lifesteal_frac_q8 /
set_lifesteal_expires_at_tick / set_damage_taken_mult_q8 /
set_damage_taken_mult_expires_at_tick / set_disguise_expires_at_tick /
set_disguise_fake_type / set_busy_until_tick / set_busy_with_ability_id /
set_busy_started_at_tick / set_busy_target_slot / set_hunger). Reader
side: `crates/dsl_compiler/src/cg/lower/expr.rs::lower_field` ->
`AgentFieldId::from_snake` is also a closed enum
(`crates/dsl_compiler/src/cg/data_handle.rs::AgentFieldId`, ~24 variants
spanning vitals, needs, statuses, busy SoA). New per-agent fields can't
be added without adding an enum variant + setter mapping + ty()
classification + storage allocation.

**Why this matters:** Fixtures that need new per-agent boolean state
(infected? cursed? recruited? thirsty-for-blood?) hit the same wall.
foraging_real worked around it by repurposing the `Hunger` field for
"ant energy"; this fixture has to do the same trick (overload `Hunger`
to mean "infection severity"; the `infected_since_tick` field is
dropped because there's no spare u32-typed status field that doesn't
have semantic baggage from elsewhere).

**Workaround in fixture:** Repurpose `hunger` (f32). Encoding:
`hunger > 0.0` ↔ infected. The contagion producer sets
`hunger = 100.0` as the initial infection load; SicknessProgresses
drains it at `sickness_rate` per tick. When the citizen's `hp <= 0`
they die; cure pulses heal `hp` AND zero `hunger`.

**Fix sketch:** Either (a) extend `AgentFieldId` to a registry the
.sim can populate (`field infected: u32`), or (b) wire the existing
`agents.set_<runtime_field>` path through the auto-emitted runtime
field allocator. The build_helper already auto-allocates per-agent
columns for unrecognised binding names — the lowering pass is the
bottleneck.

---

## GAP P-B: `@host_callable` on a fixture-defined event with no
engine-aliased kind id is silently dropped

**Surface:** `@host_callable event Infected { patient, source }`
declared in `plague_city.sim` for host-side initial-outbreak
seeding.

**Diag:**

```
warning: sims@0.1.0: [plague_city host_callable] event `Infected` has
  no engine kind id; skipping codegen
```

**Gap class:** `@host_callable` codegen is gated on the event having an
engine-aliased kind id (per `crates/dsl_ast::engine_events::
engine_event_kind_id_for_name`). The aliased set is closed
(EffectDamageApplied=26, EffectHealApplied=27, ...). Fixture-defined
event names route through dynamic kind allocation but the host-callable
injector codegen doesn't follow them; the warning fires and codegen is
silently skipped.

**Why this matters:** The host needs a way to inject events that DON'T
have an engine-side dispatcher (Infected isn't an EffectOp — it's just
a chronicle ferry from "host says agent X is sick at t=0" into the GPU
event ring). Today the only way to bootstrap such a state from CPU is
to (a) write the per-agent SoA column directly via wgpu::Queue::
write_buffer, OR (b) pre-stamp the field at agent_count construction
via `init { hunger: ... }` (but `init` only supports literal defaults,
not per-slot variation).

**Workaround:** Bootstrap infection by host-side direct write of the
`agent_hunger_buf` column on the runtime struct AFTER `try_new()`.
Same shape hill_raid uses for seeding `agent_pos_buf` /
`agent_creature_type_buf`. The Infected event still exists in the
.sim for the in-engine producer (ContagionScan), but the event is
NOT host-callable in practice.

**Fix sketch:** Auto-emit a typed injector for any `@host_callable`
event with a fixture-allocated kind id, not just engine-aliased ones.
The dispatcher already handles arbitrary kinds via the event ring;
the codegen gate is overly narrow.

---

## GAP P-C: Lower errors silently drop physics rules; only views
emit

**Surface:** With GAP P-A unresolved, every physics rule referencing
`agents.infected(...)` or `agents.set_infected(...)` fails to lower.
The build emit-stats line says "12 kernels, schedule has 12 stages" —
all 12 are folds/decays/upload/seed/pack — NO physics_* kernels and
NO spatial_build_hash kernels.

**Diag:** Multiple `[plague_city lower diag]` warnings, but the build
SUCCEEDS overall — the schedule just runs no per-tick logic.

**Gap class:** `LowerOpts` outcome handling is lossy. When a rule's
body fails to lower, the diagnostic is logged at warn-level; the rule
disappears from the schedule. There's no fail-fast option, no
"partial-lower" diagnostic that flags the missing rule downstream.
`runtime_core::step()` happily runs the (much-shrunken) schedule
each tick.

**Why this matters:** A pin author might think the fixture is wired
when it builds; only by inspecting the kernel count + emit-stats can
you tell that ContagionScan / ApplySickness / SicknessProgresses /
CitizenObserve / DoctorTriage / HealerTriage / PriestRites /
ApplyHealOrCure / ApplyLastRites are ALL silently absent from the
schedule. The emit-stats line is the only signal.

**Workaround:** The pin reads back the affected views and asserts
zero motion (NOTE comment, not assert) — same shape hill_raid's pin
uses for `total_kill_pressure == 0.0` (the chronicle-consumer
Indirect-dispatch gap). When the gap closes the pin's NOTE upgrades
to a load-bearing assertion.

**Fix sketch:** Add a build.rs flag that promotes `[lower diag]`
warnings to compile errors for fixtures opting in
(`#[deny(plague_city::missing_physics)]`-style). Or add a per-fixture
"required rule names" manifest that build.rs cross-checks against the
emitted schedule.

---

---

## GAP P-D: `agents.set_alive(self, false)` inside a fused PerAgent rule doesn't flip the alive bit — RESOLVED (downstream of T5)

**Status:** Resolved. The fusion-side hypothesis was structurally
falsified during the post-T5 audit (`d1207fca`). The actual symptom
described below was a downstream effect of the spatial-build-after-
consumer schedule cycle (Gap T5 / `gaps_observed.md`); after that
cycle was broken, the SicknessProgresses kernel runs alone (the
original `_and_ContagionScan` fusion changed shape) and the original
6 host-seeded citizens die from the alive flip exactly as expected
(`D=6` in the pin output). The fused-emit path itself preserves
conditional alive writes — pinned by
`crates/dsl_compiler/tests/fused_set_alive_conditional_emit.rs` and
the post-T5 `plague_city_pin.rs` `dead >= INITIAL_INFECTED`
assertion. The investigation log below is preserved for reference.

**Surface:** `SicknessProgresses` body:

```
let old_hp = agents.hp(self);
let new_hp = old_hp - config.plague.sickness_rate;
agents.set_hp(self, new_hp);
if (old_hp > 0.0 && new_hp <= 0.0) {
  agents.set_alive(self, false);
  emit Died { victim: self }
}
```

The hp write lands (the readback shows `min_hp = -500.00` after 500
ticks with `sickness_rate = 1.0`), but `set_alive(self, false)`
doesn't propagate — the test pin reports `D=0` final dead even though
every Citizen has accumulated hundreds of negative hp.

**Diag:** None at compile time. Runtime symptom only.

**Gap class:** Fusion-shape gap. `physics_ContagionScan_and_Sickness
Progresses` is one fused kernel (per the emit-stats). The fused body
performs both the per-pair-candidate hunger write (ContagionScan) AND
the conditional `set_alive` write (SicknessProgresses) in the same
thread invocation. The `set_alive` write is gated on `old_hp > 0 &&
new_hp <= 0` which is true only on the EXACT crossing tick — and
apparently the conditional write doesn't materialize.

Likely root cause is one of:
  1. The conditional branch's `set_alive` Assign gets dropped at
     fusion time (the alive write is only emitted from one of the
     two original rules, but the fusion's PerCell write tracker
     doesn't know which arm produced it).
  2. The `emit Died` immediately following the `set_alive` clobbers
     the write through the per-event-ring scratch (the emit's payload
     write paths through atomic adds may shadow the alive bit).
  3. The kernel-namer's `_and_` fusion drops the conditional Stmt's
     entire body when the two rules' write sets disagree.

**Why this matters:** Lifecycle is the killer feature. duel_25v25
has a clean alive flip in `ApplyDamage @phase(post)` (PerEvent over
Damaged) — this fixture tries to flip alive from a `@phase(per_agent)`
rule that ALSO contains a sibling per-agent rule writing the same
column. The fusion compromises one of them.

**Workaround:** Move the alive flip into a downstream `@phase(post)`
PerEvent rule that consumes a `Died` event. The producer just emits
`Died` from the per_agent rule (no `set_alive` in the per_agent
body); the consumer flips alive on receipt. Same shape duel_25v25's
ApplyDamage / Defeated cascade. Out of scope for this commit — the
NOTE in the pin tracks it.

**Fix sketch:** Audit the per-agent fused-kernel body emit for
conditional writes that touch the alive bit; ensure the write
survives the alphabetised `_and_` merge. A regression test in
`crates/dsl_compiler/tests/` would pin the precise shape.

**Resolution audit (post-T5):**

  - `body_ops_have_set_alive_false` in `cg/emit/kernel.rs` walks
    every body op of a fused kernel and recursively scans nested
    `If` / `Match` / `ForEachNeighborBody` bodies via
    `stmt_list_contains_set_alive_false` (`cg/emit/wgsl_body.rs`).
    A `set_alive(self, false)` nested inside a conditional in any
    sub-body of any fused op is detected and triggers the
    `agent_alive` AtomicStorage upgrade.
  - The per-stmt emit path (`is_alive_cas_site` branch in
    `cg/emit/wgsl_body.rs`) emits the
    `atomicCompareExchangeWeak(&agent_alive[idx], 1u, 0u)` CAS
    regardless of nesting depth — the conditional wrapper around
    it is preserved verbatim. An inspection of the (now-stale)
    fused WGSL artifact `physics_ContagionScan_and_Sickness
    Progresses.wgsl` from a pre-T5 build confirmed the alive CAS
    was emitted correctly inside the conditional even pre-fix.
  - Post-T5 the schedule fuses ContagionScan with CitizenObserve
    instead of SicknessProgresses (the topology shifted when the
    spatial-build chain moved earlier in the order); the gap's
    original kernel name `physics_ContagionScan_and_Sickness
    Progresses` is no longer in the schedule.
  - Pre-T5 the contagion didn't transmit (the spatial walk hit an
    empty grid on every tick), so citizens never accumulated
    negative hp; the original 6 DID die from the alive flip but
    the pin's NOTE described it as "0 deaths" because of a
    separate condition-coupling bug in the original soft NOTE.
    Post-T5 the pin reports `D=6` and the alive-flip assertion
    is load-bearing.

**Regression coverage:**

  - `crates/dsl_compiler/tests/fused_set_alive_conditional_emit.rs`
    asserts the fused WGSL body upgrades `agent_alive` to atomic
    storage AND emits the kill CAS inside the conditional, for a
    minimal-repro `Bleeder + Sickness` two-rule fusion.
  - `crates/sims/tests/plague_city_pin.rs` asserts `dead >=
    INITIAL_INFECTED` (load-bearing) — a regression that breaks
    the conditional alive flip would trip exactly that.

---

## ~~GAP P-E~~ — RESOLVED: Multiple `@materialized` views share one view_storage_primary_buf

**Closed by:** `fix(build_helper): per-view storage buffers (6-fixture aliasing gap)`.
Per-view rename allocates one `view_storage_<view>_primary_buf` per
declared view; plague_city's 5 views each get their own backing buffer.
Pin: `crates/dsl_compiler/tests/per_view_storage_distinct.rs`.

---

## (historical context) GAP P-E

**Surface:** plague_city declares 5 `@materialized` views:
beliefs_flags, beliefs_confidence, contagion_pressure, cure_count,
death_count. The auto-emitted runtime allocates ONE
`view_storage_primary_buf` of size `agent_count * agent_count * 4`
bytes; ALL 5 fold kernels bind it as their primary storage at slot 2
(per the comments at the top of `runtime_core.rs`).

**Diag:** None at compile time. The aliasing is silent; readbacks
show last-writer-wins semantics across the fold passes per tick.

**Gap class:** View-storage allocator collapsing. Today's
`build_helper.rs` allocates one shared `view_storage_primary_buf`
sized for the LARGEST view (here the pair-keyed N×N cells); every
other view writes into the same physical buffer at offsets the fold
kernels compute by hand. The N²-sized allocation is correct for the
pair-keyed views (beliefs_*) but the per-agent rollups
(contagion_pressure / cure_count / death_count, all size N) write
overlapping into the same memory.

**Why this matters:** Pin authors can't read a per-agent view's
contents reliably — by the time the readback completes, the prior
fold's data has been clobbered. Forces the pin to derive
"recovered count" / "death count" from agent SoA columns directly
(hp/hunger/alive states) rather than the materialized rollups.

**Workaround:** Pin uses agent SoA columns (hp / hunger / alive) for
all SIR breakdown — these are unambiguous (one buffer per column).
The view-derived metrics (cure_count, death_count) are not read.

**Fix sketch:** Allocate one `view_storage_*_buf` per view (or at
minimum group by storage shape — pair_map vs per-agent — and
sub-allocate within each). Same shape per_view buffer naming
that tom_probe_runtime predates the mega-crate move toward.

---

(More entries appended as additional gaps surface during fixture
iteration. Currently 5 entries covering one rebuild cycle and one
test-pin run.)
