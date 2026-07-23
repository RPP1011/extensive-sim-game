# The Webband port — MOVED

The game this plan built ("Mount & Blade: Webband", a RimWorld-style
settlement story generator) left this repository on 2026-07-23. It lives at
**https://github.com/RPP1011/webband** (private), which depends on this repo
as a git dependency pinned to a rev.

The full plan document and its verbatim data extract — every slice report,
every engine finding, every honest gaps list — moved with it and are now
`docs/webband-port.md` and `docs/webband-port-data.md` in that repo. This stub
exists because source comments across `crates/dsl_ast`, `crates/dsl_compiler`
and `crates/engine_play` cite this path when explaining *why* a defect was
fixed; those citations still resolve, they just resolve to a signpost.

**What stayed here, and why.** Every engine fix the port found is engine code
and never moved: the pair-fold dispatch fix, the user-event / engine-alias
kind-id reservation, the scheduler's consumer-before-producer defect, the
field/annotation boundary, the render-block subkind range defect, the wide
pair-map dispatch, and the `DebugTimings` wiring. So did
`assets/sim/many_events_ability.sim` + `assets/ability_test/many_events_ability/`
+ `crates/sims/tests/many_events_ability_pin.rs` — a purpose-built synthetic
fixture that keeps a large-event-count subject in the corpus now that the
game's 60-event fixture is gone.

**What left.** `assets/sim/webband_*.sim`,
`assets/ability_test/webband_colony/`, `dataset/abilities/webband/`,
`crates/webband_{app,bridge,play}`, the four `crates/sims/tests/webband_*.rs`
integration tests, the two `crates/sims/examples/webband_*.rs` harnesses,
`crates/dsl_compiler/tests/webband_abilities.rs`, and the recorded
determinism digests under `crates/sims/target/webband_*/`.

The move was proved behaviour-neutral: both recorded digests reproduced from
the new repo, in fresh processes, bit for bit.
