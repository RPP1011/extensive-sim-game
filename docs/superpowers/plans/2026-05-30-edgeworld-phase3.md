# Edgeworld Phase 3 — Reproduction (Living Population) Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Well-fed survivors reproduce → a living, oscillating population that grows when food is plentiful, gets culled by starvation/wolves, and recovers — instead of a static remnant.

**Architecture (race-free reproduction):** Pre-link each "breeder" survivor 1:1 to a dead "offspring" slot (via the `engaged_with` column, the `trade_caravans` heir pattern). A well-fed breeder on a birth cooldown whose offspring slot is dead emits `Born { child }`; a `@phase(post) BornRevive` consumer does `set_alive(child, true)` + positions it at the parent + resets hunger. Because each breeder owns a unique offspring slot, there is NO allocation race (two parents never revive the same slot). A breeder re-revives its slot whenever it dies → sustained reproduction.

**Tech stack:** `.sim` DSL; proven `set_alive(_, true)` revive (trade_caravans), `engaged_with` link, Born event + `@phase(post)` consumer.

---

## AIS (P8)
- P1: DSL only. P2: reuse `engaged_with` column for the parent→offspring link, no new SoA column. P3/P5/P11: revive is a boolean `set_alive(child,true)` to a UNIQUE per-parent slot — no N-to-1 race. Birth cadence via `world.tick % cooldown` (deterministic), or keyed RNG if probabilistic. P7: `Born` flagged `@replayable`.
- **Risk:** reviving a dead slot from a rule is proven (trade_caravans `BornRevive`), but confirm (a) the revived slot's position/hunger/creature_type seed correctly so it behaves as a survivor next tick, and (b) `engaged_with` is free to repurpose in edgeworld (it's not currently used). Reviving + setting the child's `pos` (vec3) needs the DCE anchor (a scalar self/child write in the same rule).

## Reference
- `assets/sim/trade_caravans.sim:365-397` — `OnDied`→`Born`→`BornRevive` (`set_alive(heir,true)` + `set_hunger(heir,0)`), heir via `agents.engaged_with(a)`. THE template.
- `assets/sim/edgeworld.sim` — survivors (type 1, hunger on `hunger`), the existing physics rules, world bounds, fear column on `shield_hp`.

---

## Task 1: Core reproduction — well-fed breeders revive their offspring slot

**Files:** `assets/sim/edgeworld.sim`, `crates/sims/tests/edgeworld_pin.rs`, `crates/sims/tests/edgeworld_common/mod.rs`.

- [ ] **Step 1 — seed model + link.** Extend `seed_world` (or a new seeder) so survivors split into K breeders (alive) + K offspring slots (dead, `alive=0`, type 1), with each breeder's `engaged_with` set to its unique offspring slot index (write `agent_engaged_with_buf` — confirm the field name; grep the runtime/trade_caravans pin). Confirm `engaged_with` is unused by edgeworld today.
- [ ] **Step 2 — config + Reproduce rule.** Add config `birth_hunger_max: f32 = 0.3` (must be well-fed), `birth_cooldown: u32 = 60`. Add:
```
@replayable @gpu_amenable
event Born { parent: AgentId, child: AgentId }

physics Reproduce @phase(per_agent) {
  on Tick {} where (self.alive
                    && self.creature_type == 1
                    && agents.hunger(self) < config.edgeworld.birth_hunger_max
                    && (world.tick % config.edgeworld.birth_cooldown == 0)) {
    let child = agents.engaged_with(self);
    // only birth if the offspring slot is currently dead — guard inside body if needed
    emit Born { parent: self, child: child }
  }
}

physics BornRevive @phase(post) {
  on Born { parent: p, child: c } {
    agents.set_alive(c, true);
    agents.set_hunger(c, 0.0);
    agents.set_pos(c, agents.pos(p));   // place newborn at the parent (anchor: set_hunger above counts)
    agents.set_creature_type(c, 1);
  }
}
```
Verify against trade_caravans: the "only if child dead" guard — trade_caravans births on death (child always dead). Here a breeder fires repeatedly; ensure reviving an ALREADY-alive child is harmless (set_alive(true) on alive = no-op; but the hunger/pos reset would teleport a living child — so GATE the birth on the child being dead). If a rule can't read `agents.alive(child)` for the neighbour-child in the guard, gate in the BornRevive body (`if (agents.alive(c) == false) { ...revive... }`) or accept the reset as "the breeder only re-cradles a dead child." Resolve empirically; document.
- [ ] **Step 3 — test population growth.** Seed K=8 breeders (well-fed, hunger 0) + 8 dead offspring + ample food, no wolves. Step ~200 ticks. Assert alive survivor count GREW above 8 (births happened) toward 16. Confirm newborns forage (their hunger rises/falls like adults).
- [ ] **Step 4 — all pins green, commit** `feat(edgeworld): reproduction — well-fed survivors revive offspring slots`.

## Task 2: Tune the living oscillation + render births + coexistence

**Files:** `assets/sim/edgeworld.sim` (tuning), `crates/sims/tests/edgeworld_render.rs`, `edgeworld_pin.rs`.

- [ ] **Step 1 — render.** Newborns/all survivors render as before (amber, fear-tinted); optionally flash a brief birth marker. Ensure the dynamic-viewport render handles the growing/shrinking population.
- [ ] **Step 2 — tune for a LIVING CYCLE.** With reproduction + food limits + wolves, tune `birth_hunger_max`, `birth_cooldown`, food params, wolf count so the population OSCILLATES over 600+ ticks: grows when food is plentiful, overshoots, gets culled (starvation + predation), recovers via births. The target is a non-flat population trace (visible ups AND downs), not monotonic growth or instant extinction. Replace/extend the coexistence pin with an **oscillation pin**: assert the survivor population has both a high-water mark well above the start AND a subsequent dip (a real cycle), AND survivors+wolves both alive at 600.
- [ ] **Step 3 — VISUAL self-verify** (no waiting): open early/mid/late frames; confirm population visibly grows then culls then recovers, wolves persist, agents bounded. Iterate.
- [ ] **Step 4 — all pins + render green, commit** `feat(edgeworld): tuned reproductive oscillation; Phase 3 complete`. Report 3 sign-off frames + the population trace.

## Self-review
- Race-free: unique per-breeder offspring slot → no two parents revive the same slot.
- The offspring (revived) slots are full survivors (forage/flee/fear/die). If breeders all die, population can collapse — realistic; note it.
- Oscillation pin must show genuine ups-and-downs, not a flat line (the honest bar this whole project has been held to).
