# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Binaries in this crate

Only two source files are actually wired up as `[[bin]]` targets in `Cargo.toml`:

- **`viz_app.rs`** — universal terminal sim visualizer. Looks up a sim name in its `SIMS` table
  (a `&[(&str, Factory, u32, &str)]` of name / factory-fn / default-agent-count / description),
  calls the factory to get a `Box<dyn engine::CompiledSim>`, then steps it in a loop (up to
  `MAX_TICKS = 2000`, one tick every `FRAME_MS = 80ms`) rendering an ASCII frame after each step
  via `viz::render_sim_auto`. Right now `SIMS` has exactly **one** entry: `"tom_probe"` →
  `tom_probe_runtime::make_sim`. The comment in the table notes `boids` was retired from here
  because the `sims` mega-crate doesn't yet expose a `make_sim()` / `Box<dyn CompiledSim>` /
  populated `positions()` surface — that's the blocker for wiring any mega-crate fixture into
  `viz_app`, not a naming gap.
- **`tom_probe_app.rs`** — non-interactive correctness harness (not a visualizer) for
  `tom_probe_runtime`. Runs `TomProbeState` for a fixed `SEED`/`AGENT_COUNT=32`/`TICKS=100`, reads
  back the `beliefs` (agent_count² pair-map) buffer, and checks the diagonal is all `1u` and the
  off-diagonal all `0u`. Classifies the result into three named outcomes — (a) FULL FIRE, (b) NO
  FIRE, (c) PARTIAL FIRE — printed with a one-line diagnosis of which compiler stage likely broke,
  and `exit(1)`s on (b)/(c). This is a discovery-probe pattern repeated across several fixtures in
  the workspace: run to a fixed tick count, dump/assert on a specific buffer shape, classify by
  outcome letter.

Two more files sit in `src/` with no `[[bin]]` entry and no matching crate in the workspace:

- **`disease_spread_app.rs`** — SIR-epidemic trace harness for a `disease_spread_runtime` crate.
- **`objective_capture_app.rs`** — 10v10 objective-capture trace harness for an
  `objective_capture_10v10_runtime` crate.

Neither `disease_spread_runtime` nor `objective_capture_10v10_runtime` is a workspace member
(check `members = [...]` in the root `Cargo.toml`) or a dependency of `sim_app` — **these two
files do not compile today and are not reachable via any `cargo build`/`cargo run` invocation.**
Treat them as orphaned harnesses (probably pre-mega-crate-consolidation leftovers or written
ahead of a runtime crate that never landed) rather than working examples. If revived, they'd need
their own runtime crate added to the workspace, a `bin-<name>` feature + `[[bin]]` entry here, and
an `optional = true` dependency — follow the `tom_probe_app` pattern below.

## Commands

```bash
cargo run -p sim_app --bin viz_app --features bin-viz_app -- <sim_name> [seed] [agent_count]
cargo run -p sim_app --bin viz_app --features bin-viz_app             # no args: lists SIMS table and exits 1
cargo run -p sim_app --bin tom_probe_app --features bin-tom_probe_app # no CLI args; SEED/AGENT_COUNT/TICKS are consts
```

`viz_app` args, positionally: `<name>` (required, must match a `SIMS` entry) → `[seed]` (u64,
defaults to `0xC0FFEE_DEC1DE_42`) → `[agent_count]` (u32, defaults to that sim's table default).
Ctrl-C to quit; it otherwise stops itself at `MAX_TICKS`.

### Feature-gating structure (`Cargo.toml`)

- Every per-fixture runtime dependency is declared `optional = true` (currently only
  `tom_probe_runtime` is un-commented; the file is full of commented-out `dep = { path = ...,
  optional = true }` lines documenting fixtures that were once wired in and got pulled back out —
  read them for history, don't uncomment them without checking the fixture still exists).
- Each `[[bin]]` has `required-features = ["bin-<name>"]`, and a matching `bin-<name> =
  ["dep:<runtime_crate>"]` feature turns on only the dependency that binary needs. `cargo build -p
  sim_app` with no feature flags builds no binaries.
- `bin-viz_app` is the exception: because `viz_app`'s `SIMS` table can reference any fixture
  runtime crate by name, its feature enables **every** optional runtime dependency unconditionally
  (currently just `dep:tom_probe_runtime`, but the intent per the Cargo.toml comment is "all N
  runtime deps go here" as more get wired into `SIMS`) rather than gating per-row.
- `default = ["test-fixtures"]`, and `test-fixtures = []` is currently an empty feature group —
  it's a no-op kept only so `cargo test -p sim_app --features test-fixtures` doesn't fail; the
  fixtures it used to pull for `tests/cross_fixture_determinism.rs` all migrated to the `sims`
  mega-crate and that test file is now a stub.

**To add a new bin-gated binary:** add `runtime_crate = { path = "../runtime_crate", optional =
true }` under `[dependencies]`, add `bin-<name> = ["dep:runtime_crate"]` under `[features]`, and
append a `[[bin]] name = "<name>" path = "src/<name>.rs" required-features = ["bin-<name>"]`
block. If the binary should also be reachable through `viz_app`, add its factory fn to the `SIMS`
table in `src/viz_app.rs` and add its dep to the `bin-viz_app` feature list too.

## Architecture: `viz_app` SIMS table → renderer

`SIMS: &[(&str, Factory, u32, &str)]` where `Factory = fn(u64, u32) -> Box<dyn engine::CompiledSim>`.
`main()` linear-scans `SIMS` for the row matching `args[1]`, calls `factory(seed, agent_count)` to
build the sim, then loop-steps it and calls `viz::render_sim_auto(&mut *sim, ...)` each iteration.
There is no dynamic discovery of `sims::<fixture>::GeneratedRuntime` — every reachable sim is a
manually-added row whose factory fn must already exist and return `Box<dyn CompiledSim>` (which is
why mega-crate fixtures aren't reachable yet: they don't expose that constructor shape).

`src/viz.rs` is a **terminal ASCII renderer**, Qud/CogMind-style — not logs, not a GUI:
- `render_sim_auto` pulls an `engine::AgentSnapshot` via `sim.snapshot()` (positions +
  creature_types + alive bits); if that's empty it falls back to `sim.positions()` and synthesizes
  an all-alive, all-creature_type-0 snapshot so at least single-species sims render out of the box.
  If even `positions()` is empty, it returns `""` and `viz_app::main` treats that as "sim hasn't
  implemented viz" and exits with code 3.
- Viewport: uses the sim's `default_viewport()` if it implements one, else auto-fits a bounding
  box from observed agent positions (`auto_viewport`, 10% padding, min span 2.0).
- Glyphs: uses the sim's `glyph_table()` if implemented, else `default_glyphs` assigns single
  ASCII letters (`a`, `b`, `c`, ... wrapping after 26) keyed by `creature_type` id, with colors
  cycled from an 8-entry ANSI-256 palette.
- Each frame is built as one big string with ANSI escapes (`\x1b[2J\x1b[H` clear+home, `\x1b[38;5;
  Nm` 256-color glyphs) — a bordered `width × height` grid (`VIEW_W=80 × VIEW_H=24` from
  `viz_app.rs`), a title line, and status lines (auto tick/alive-count/per-creature_type counts,
  plus whatever `extra_status` the caller passes — `viz_app` passes seed and a "Ctrl-C to quit"
  hint). Multiple agents landing in the same terminal cell render as the highest-`creature_type`
  glyph (so e.g. predators visually dominate prey in a shared cell).
- Both `Viewport` methods and `render_sim_frame` are marked `#[allow(dead_code)]` — they're a
  lower-level API (explicit viewport, no snapshot fallback) not currently called by `viz_app`
  itself but kept as the primitives `render_sim_auto` composes, for any future caller that wants
  more control.

## Non-obvious

- `disease_spread_app.rs` and `objective_capture_app.rs` are dead code today (see above) — don't
  assume they build just because they're in `src/`; always check `[[bin]]` entries in `Cargo.toml`
  before treating a `src/*.rs` file as a real binary.
- `viz_app`'s single live `SIMS` row (`tom_probe`) drives a *belief-tracking probe*, not anything
  visually interesting — `tom_probe_runtime::make_sim`'s agents are point creatures with no
  meaningful spatial behavior, so running `viz_app tom_probe` mostly demonstrates the renderer
  plumbing rather than showing an interesting sim.
- `tests/cross_fixture_determinism.rs` is a stub (per its own doc comment and the `test-fixtures`
  feature comment) — don't expect it to exercise real fixture determinism; the fixtures it used to
  cover moved to `crates/sims`.
- The `Cargo.toml` file has ~180 lines of comments documenting fixtures that were once dependencies
  here and were removed as they migrated to the `sims` mega-crate — treat that comment block as a
  historical log, not as a hint that those deps still need restoring.
