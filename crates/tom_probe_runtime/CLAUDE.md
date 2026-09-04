# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What tom_probe is

The fixture is `assets/sim/tom_probe.sim` — the Theory-of-Mind (ToM) belief
read/write end-to-end probe. It is a **discovery probe**, not a gameplay
fixture: its purpose is to exercise the belief-state machinery end-to-end and
surface compiler gaps as data, not to model a real game system.

- `entity Knower : Agent` is the only entity. Every alive Knower emits one
  `BeliefAcquired { observer: self, subject: self, fact_bit: 1 }` event per
  tick (`physics WhatIBelieve`).
- `belief beliefs_flags(observer, subject) -> u32` is a `pair_map`-storage
  fold (`self |= b`) that atomically ORs `fact_bit` into
  `view_storage_primary[observer * agent_cap + subject]` — a u32 `atomicOr`,
  P11-trivial (no CAS retry). After the first tick the diagonal
  `beliefs_flags(i, i)` is `1u` for every agent and every off-diagonal cell
  stays `0u` forever ("FULL FIRE"), because no rule ever emits an event with
  `observer != subject`.
- The other 5 BeliefState columns from the spec (`last_known_pos`,
  `last_known_creature_type`, `last_seen_tick`, `confidence`, `suspicion`)
  are **not** `@materialized` views in this fixture — they're plain SoA
  buffers allocated and owned by `tom_probe_runtime`, not by the compiler's
  view machinery. The .sim only declares `physics @phase(post)` consumer
  rules (`ApplyObserveBeliefUpdate`, `ApplyScryBeliefUpdate`,
  `ApplyRevealBeliefUpdate`, `ApplyDecoyBeliefUpdate`,
  `ApplyEraseBeliefUpdate`, `ApplyDisguise`) that read a synthetic chronicle
  event and write those columns via `agents.set_beliefs_<field>(...)`.
- The runtime's `observe()` / `scry()` / `reveal()` / `decoy()` /
  `erase_belief()` / `disguise()` methods on `TomProbeState` are the host
  side of that contract: each hand-packs a 10-word chronicle record (see the
  per-method doc comments in `src/lib.rs` for the exact slot layout — it
  must match `dsl_compiler::cpu_chronicle_reference` exactly), appends it via
  `EventRing::append_chronicle_record`, and synchronously dispatches the
  matching compiler-emitted consumer kernel **outside** the per-tick
  `step()` encoder. `step()` itself only drives the base producer/fold pair
  (`physics_WhatIBelieve` + `fold_beliefs_flags`); the 6 consumer kernels are
  still listed in `step()`'s schedule but dispatched with `event_count = 0`
  as a no-op — this exists purely to keep the pipelines warm in the kernel
  cache and pinned to a stable submission order, not to do work per tick.
- `decay_step()` is the one remaining hand-rolled (non-.sim) per-tick sweep:
  it decrements `beliefs_confidence` toward 0 based on staleness. Its WGSL
  comes from `dsl_compiler::belief_decay_wgsl::decay_kernel_wgsl()`, not from
  `.sim`-authored rules or `OUT_DIR/generated.rs`.

Per the crate's `Cargo.toml` header comment: there is intentionally **no**
smoke test asserting belief-read (`agents.beliefs(o, s).<field>`) values are
observable, because that read-accessor path is rejected at CG-lower time
today (see `docs/superpowers/notes/2026-05-04-tom-probe.md`). The .sim
sidesteps the gap by routing every belief mutation through
events + chronicle consumers instead of a direct read/write AST surface.

## Commands and the sim_app data flow

```bash
cargo build -p tom_probe_runtime
cargo test -p tom_probe_runtime          # tests/*.rs pin suite (see below)
```

`sim_app`'s `tom_probe_app` binary is the runnable harness:

```bash
cargo run -p sim_app --bin tom_probe_app --features bin-tom_probe_app
```

`crates/sim_app/Cargo.toml` pulls `tom_probe_runtime` in as an `optional`
dependency gated by the `bin-tom_probe_app` feature (so it isn't compiled
into default `sim_app` builds). `crates/sim_app/src/tom_probe_app.rs` is a
thin driver: it constructs `TomProbeState::new(seed, 32)`, calls `sim.step()`
100 times, reads back `beliefs_flags()`, and classifies the result as (a)
FULL FIRE (diagonal all `1u`, off-diagonal all `0u` — the belief path lights
up end-to-end), (b) NO FIRE (every cell `0u` — a rule dropped at lower time),
or (c) PARTIAL FIRE (a cfg/offset mismatch between compiler emit and runtime
cfg uniform), exiting nonzero on anything but FULL FIRE. **`tom_probe_app`
only exercises the base per-tick producer/fold loop** — it never calls
`observe`/`scry`/`reveal`/`decoy`/`erase_belief`/`disguise`/`decay_step`;
those verbs are exercised exclusively by this crate's own `tests/*_pin.rs`
files, not by the `sim_app` binary. `tom_probe_runtime` is not wired into
`viz_app`'s `SIMS` table — `CompiledSim::positions()` always returns `&[]`,
so there's nothing to visualize; it's a headless correctness probe.

## build.rs vs the `sims` mega-crate pattern

`build.rs` is one line: `dsl_compiler::build_helper::emit("tom_probe")`. That
calls `build_helper::emit_with_strategy` → `emit_into`, which writes
`generated.rs` (and the per-kernel `.wgsl` files) directly into `OUT_DIR`.
`src/lib.rs` pulls it in at crate root with
`include!(concat!(env!("OUT_DIR"), "/generated.rs"));`, so the generated
`crate::`-qualified paths resolve correctly as-is.

The `sims` mega-crate's `build.rs` instead calls
`build_helper::emit_namespaced(fixture)` once per discovered `.sim` file,
which writes each fixture's generated files into `OUT_DIR/<fixture>/` and
then does a literal text rewrite (`crate::` → `super::`) across every
generated `.rs` file in that subdirectory — necessary because `sims` wraps
each fixture's generated code in `pub mod <fixture> { include!(...) }`, so
`crate::`-qualified paths would resolve to the wrong module without the
rewrite. `tom_probe_runtime`, being its own crate, needs none of this.

The build.rs's own doc comment notes it used to hand-flip
`LowerOpts.belief_state: true` before calling
`lower_compilation_to_cg_with_opts`; the 2026-05-11 belief-state
auto-detect (which walks every physics rule body for
`agents.set_beliefs_<field>(...)` calls) made that unnecessary, so this
build.rs is now structurally identical boilerplate to any other
per-fixture runtime's build.rs — the only difference from the mega-crate
path is `emit()` vs `emit_namespaced()`.

## Non-obvious things

- **Rule-order cycle is tolerated, not fixed.** `lower_compilation_to_cg_with_opts`
  detects an ordering cycle between the `WhatIBelieve` producer and its
  consumers; the build helper emits it as a `cargo:warning` and continues
  with the partial program (same as every other fixture) unless
  `SIM_REQUIRE_ALL_RULES=1` is set in the build environment, in which case it
  becomes a hard build error.
- **`agent_creature_type` storage width changed under you.** Pre-Phase-3.8 it
  was u8-packed; Phase 3.8 flipped the underlying GPU buffer to one `u32` per
  agent so the compiler-emitted `.sim` consumer's `array<u32>` binding type
  matches. The public `seed_agent_creature_type()` / `agent_creature_type()`
  API still takes/returns `&[u8]` — it zero-extends on write and truncates
  the low byte on read internally.
- **Two independent confidence-decay implementations coexist by construction.**
  The `.sim` declares a speculative `@decay(per = tick, mode = sub, by = 1,
  gate = BeliefStillFresh) @storage(packed_q8) view confidence(...)` that
  the compiler auto-emits a per-word packed-byte decay kernel for — but per
  the `.sim`'s own comment this is "WGSL-side parity only": that kernel
  writes to the view's own `view_storage_primary` buffer, a buffer separate
  from the runtime-allocated `beliefs_confidence_primary` that
  `set_beliefs_confidence` actually targets. The two don't conflict only
  because they touch physically different buffers. The runtime's real decay
  path is still the hand-rolled `decay_step()` / `belief_decay_wgsl` kernel;
  don't assume editing the `.sim`'s `confidence` view changes production
  decay behavior.
- **Pin tests skip, they don't fail, without a GPU adapter.** Every test in
  `tests/*.rs` constructs state via `std::panic::catch_unwind(|| TomProbeState::new(...))`
  and returns early (prints a skip message) if that panics — the same
  convention `GpuContext::new_blocking()` failures trigger across the
  workspace's other `*_runtime` pin suites.
- **Belief-state indexing is always row-major `observer * agent_count +
  subject`**, consistently across the fold kernel, the 6 SoA columns, and
  every consumer rule — including the "6 columns" set (`beliefs_flags`,
  `beliefs_pos`, `beliefs_type`, `beliefs_tick`, `beliefs_confidence`,
  `beliefs_suspicion`), each stored as an independent `(primary, staging,
  host-side cache, dirty flag)` quadruple on `TomProbeState`, lazily read
  back to `cache` only when `dirty` is set.
